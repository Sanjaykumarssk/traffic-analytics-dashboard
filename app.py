"""
FastAPI backend for traffic route analysis.
Provides endpoints for autocomplete, route analysis, and serving the frontend.
"""

import os
import json
from typing import Optional, Union
from fastapi import FastAPI, HTTPException, Query, Depends, status, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
import io
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
import joblib
from datetime import datetime, UTC, timedelta
from typing import Optional, Union, List
import secrets
import uuid

# Import logging and rate limiting
from logging_config import setup_logging, get_logger
from rate_limiter import RateLimitMiddleware

# Setup logging
setup_logging()
logger = get_logger(__name__)

from utils import (
    tomtom_geocode,
    tomtom_autocomplete,
    tomtom_route,
    summarize_route,
    compute_route_cost,
    haversine_m
)
from db import (
    init_db, get_session, save_analysis, AnalysisResult,
    User, SavedRoute, RouteRating, Notification
)
from sqlalchemy.orm import Session
from auth import (
    verify_password, get_password_hash, create_access_token,
    get_current_user, get_current_active_user, get_current_admin_user,
    authenticate_user, create_user, get_user_by_username, Token, UserCreate as AuthUserCreate, UserResponse,
    get_optional_user
)
from analytics import (
    get_peak_hours_analysis, get_day_of_week_analysis,
    get_seasonal_trends, calculate_route_reliability, predict_future_congestion
)
from export_utils import export_to_csv, export_to_excel, export_to_pdf
from notifications import (
    create_notification, check_traffic_alerts,
    suggest_best_time_to_leave, check_congestion_warnings
)
from cache_utils import cached, clear_cache, get_cache_stats
from realtime_utils import get_traffic_incidents, auto_refresh_route, monitor_route_changes
from notifications import get_user_notifications, mark_notification_read
from analytics import get_traffic_hotspots

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Route Analysis API",
    description="Real-time traffic congestion analysis with ML predictions",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Initialize database
init_db()

# Load ML model if available
ML_MODEL = None
MODEL_PATH = os.getenv("MODEL_PATH", "rf_model.pkl")
if os.path.exists(MODEL_PATH):
    try:
        ML_MODEL = joblib.load(MODEL_PATH)
        logger.info(f"✅ Loaded ML model from {MODEL_PATH}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load ML model: {e}")

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Add PWA manifest route
@app.get("/static/manifest.json")
async def get_manifest():
    """Serve PWA manifest."""
    manifest_path = os.path.join("static", "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/json")
    return JSONResponse({"error": "Manifest not found"}, status_code=404)


class RouteAnalysisRequest(BaseModel):
    """Request model for route analysis."""
    origin: Union[str, dict] = Field(..., description="Origin as place name or {lat, lon}")
    destination: Union[str, dict] = Field(..., description="Destination as place name or {lat, lon}")
    maxAlternatives: int = Field(3, ge=1, le=5, description="Maximum route alternatives")
    alpha: float = Field(1.0, description="Weight for travel time in cost calculation")
    beta: float = Field(0.5, description="Weight for delay in cost calculation")
    gamma: float = Field(0.001, description="Weight for distance in cost calculation")
    avoid_tolls: bool = Field(False, description="Avoid toll roads")
    avoid_ferries: bool = Field(False, description="Avoid ferries")
    avoid_highways: bool = Field(False, description="Avoid highways")


class UserCreate(BaseModel):
    """User registration model."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str


class SavedRouteCreate(BaseModel):
    """Create saved route model."""
    route_name: str
    origin: Union[str, dict]
    destination: Union[str, dict]
    route_preferences: Optional[dict] = None


class RouteRatingCreate(BaseModel):
    """Create route rating model."""
    route_id: str
    rating: int = Field(..., ge=1, le=5)
    review: Optional[str] = None


class ShareRouteRequest(BaseModel):
    """Share route request model."""
    route_id: str
    route_index: Optional[int] = None


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """
    Decode Google-style polyline string to list of (lat, lon) tuples.
    
    Args:
        encoded: Encoded polyline string
        
    Returns:
        List of (latitude, longitude) tuples
    """
    if not encoded:
        return []
    
    # TomTom uses a similar encoding, but may also return coordinates directly
    # For now, if we get coordinates from TomTom route geometry, use them directly
    # This is a simplified decoder - full implementation would handle TomTom's specific encoding
    try:
        # If encoded is already a list of coordinates, return it
        if isinstance(encoded, list):
            return [(float(p["lat"]), float(p["lon"])) for p in encoded if "lat" in p and "lon" in p]
    except:
        pass
    
    # Basic polyline decoder (simplified - may need adjustment for TomTom format)
    coordinates = []
    index = 0
    lat = 0
    lon = 0
    
    try:
        while index < len(encoded):
            # Decode latitude
            shift = 0
            result = 0
            while index < len(encoded):
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat
            
            # Check if we have enough characters for longitude
            if index >= len(encoded):
                break
                
            # Decode longitude
            shift = 0
            result = 0
            while index < len(encoded):
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            dlon = ~(result >> 1) if (result & 1) else (result >> 1)
            lon += dlon
            
            coordinates.append((lat / 1e5, lon / 1e5))
    except (IndexError, ValueError) as e:
        # If decoding fails, return empty list
        logger.error(f"Polyline decoding error: {e}")
        return []
    
    return coordinates


def extract_route_geometry(route_json: dict) -> list[tuple[float, float]]:
    """
    Extract route geometry from TomTom route JSON.
    
    Args:
        route_json: Route object from TomTom API
        
    Returns:
        List of (lat, lon) tuples for route path
    """
    # TomTom routes may have geometry in different formats
    # Check for legs and points
    geometry = []
    
    try:
        legs = route_json.get("legs", [])
        for leg in legs:
            points = leg.get("points", [])
            for point in points:
                if isinstance(point, dict):
                    # Check for latitude/longitude fields
                    if "latitude" in point and "longitude" in point:
                        try:
                            geometry.append((float(point["latitude"]), float(point["longitude"])))
                        except (ValueError, TypeError):
                            continue
                    # Alternative format: lat/lon
                    elif "lat" in point and "lon" in point:
                        try:
                            geometry.append((float(point["lat"]), float(point["lon"])))
                        except (ValueError, TypeError):
                            continue
        
        # If no geometry found, try to get from guidance instructions
        if not geometry:
            guidance = route_json.get("guidance", {})
            instructions = guidance.get("instructions", [])
            for instruction in instructions:
                point = instruction.get("point", {})
                if "latitude" in point and "longitude" in point:
                    try:
                        geometry.append((float(point["latitude"]), float(point["longitude"])))
                    except (ValueError, TypeError):
                        continue
    except Exception as e:
        logger.error(f"Error extracting route geometry: {e}")
        return []
    
    return geometry


def predict_congestion(features: dict) -> Optional[float]:
    """
    Predict congestion using ML model if available.
    
    Args:
        features: Dictionary with feature values
        
    Returns:
        Predicted congestion level or None
    """
    if ML_MODEL is None:
        return None
    
    try:
        # Map features to model input
        # Model expects: hour, weekday, is_weekend, distance_km, route_index,
        # travel_time_s, no_traffic_s, delay_s, rolling_mean_congestion, rolling_std_congestion
        import pandas as pd
        import numpy as np
        
        now = datetime.now(UTC)
        feature_dict = {
            "hour": now.hour,
            "weekday": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            "distance_km": features.get("distance_km", 0),
            "route_index": features.get("route_index", 0),
            "travel_time_s": features.get("travel_time_s", 0),
            "no_traffic_s": features.get("no_traffic_s", 0),
            "delay_s": features.get("delay_s", 0),
            "rolling_mean_congestion": features.get("rolling_mean_congestion", 1.0),
            "rolling_std_congestion": features.get("rolling_std_congestion", 0.0)
        }
        
        # Create DataFrame with single row
        df = pd.DataFrame([feature_dict])
        
        # Get feature order from model (if available)
        # For simplicity, use all numeric features
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        X = df[numeric_cols].fillna(0)
        
        prediction = ML_MODEL.predict(X)[0]
        return float(prediction)
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main frontend HTML page."""
    index_path = os.path.join("templates", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>Traffic Route Analysis API</h1><p>Frontend not found. Please create templates/index.html</p>",
        status_code=404
    )


@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    """Serve the login/registration page."""
    login_path = os.path.join("templates", "login.html")
    if os.path.exists(login_path):
        with open(login_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>Login</h1><p>Login page not found.</p>",
        status_code=404
    )


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    """Serve the admin dashboard page."""
    admin_path = os.path.join("templates", "admin.html")
    if os.path.exists(admin_path):
        with open(admin_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>Admin Dashboard</h1><p>Admin page not found.</p>",
        status_code=404
    )


@app.get("/account", response_class=HTMLResponse)
async def serve_account():
    """Serve the user account page."""
    account_path = os.path.join("templates", "account.html")
    if os.path.exists(account_path):
        with open(account_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>My Account</h1><p>Account page not found.</p>",
        status_code=404
    )


@app.get("/autocomplete")
async def autocomplete(q: str = Query(..., description="Search query")):
    """
    Get autocomplete suggestions from TomTom API.
    
    Args:
        q: Search query string
        
    Returns:
        List of suggestion objects
    """
    try:
        suggestions = tomtom_autocomplete(q)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autocomplete failed: {str(e)}")


@app.post("/analyze-route")
async def analyze_route(
    request: RouteAnalysisRequest,
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Analyze route alternatives with cost calculation and ML prediction.
    
    Args:
        request: RouteAnalysisRequest with origin, destination, and parameters
        current_user: Optional authenticated user
        
    Returns:
        JSON with analyzed routes and best route recommendation
    """
    try:
        # Parse origin
        if isinstance(request.origin, str):
            o_lat, o_lon = tomtom_geocode(request.origin)
            origin_data = {"name": request.origin, "lat": o_lat, "lon": o_lon}
        else:
            o_lat = float(request.origin.get("lat"))
            o_lon = float(request.origin.get("lon"))
            origin_data = request.origin
        
        # Parse destination
        if isinstance(request.destination, str):
            d_lat, d_lon = tomtom_geocode(request.destination)
            dest_data = {"name": request.destination, "lat": d_lat, "lon": d_lon}
        else:
            d_lat = float(request.destination.get("lat"))
            d_lon = float(request.destination.get("lon"))
            dest_data = request.destination
        
        # Fetch routes from TomTom
        route_json = tomtom_route(
            o_lat, o_lon, d_lat, d_lon,
            maxAlternatives=request.maxAlternatives
        )
        
        routes = route_json.get("routes", [])
        if not routes:
            raise HTTPException(status_code=404, detail="No routes found")
        
        # Analyze each route
        analyzed_routes = []
        route_id = f"{origin_data.get('name', f'{o_lat},{o_lon}')}→{dest_data.get('name', f'{d_lat},{d_lon}')}"
        
        for idx, route in enumerate(routes):
            summary = summarize_route(route)
            
            # Calculate distance if not in summary
            if summary["length_m"] == 0:
                summary["length_m"] = haversine_m(o_lat, o_lon, d_lat, d_lon)
            
            # Calculate cost
            cost = compute_route_cost(
                summary["travel_time_s"],
                summary["no_traffic_s"],
                summary["delay_s"],
                summary["length_m"],
                alpha=request.alpha,
                beta=request.beta,
                gamma=request.gamma
            )
            
            # ML prediction
            ml_predicted = predict_congestion({
                "distance_km": summary["length_m"] / 1000.0,
                "route_index": idx,
                "travel_time_s": summary["travel_time_s"],
                "no_traffic_s": summary["no_traffic_s"],
                "delay_s": summary["delay_s"]
            })
            
            # Calculate congestion ratio
            congestion_ratio = (
                summary["travel_time_s"] / summary["no_traffic_s"]
                if summary["no_traffic_s"] and summary["no_traffic_s"] > 0
                else None
            )
            
            # Extract geometry
            geometry = extract_route_geometry(route)
            
            # Always calculate delay from travel_time and no_traffic_time
            # Don't rely on API-provided delay_s which might be 0
            calculated_delay = 0
            if summary["travel_time_s"] and summary["no_traffic_s"]:
                calculated_delay = max(0, summary["travel_time_s"] - summary["no_traffic_s"])
            elif summary.get("delay_s"):
                calculated_delay = summary["delay_s"]
            
            analyzed_route = {
                "route_index": idx,
                "travel_time_s": summary["travel_time_s"],
                "no_traffic_s": summary["no_traffic_s"],
                "delay_s": calculated_delay,
                "length_m": summary["length_m"],
                "congestion_ratio": congestion_ratio,
                "calculated_cost": cost,
                "ml_predicted_congestion": ml_predicted,
                "geometry": geometry
            }
            analyzed_routes.append(analyzed_route)
            
            # Save to database
            try:
                session = get_session()
                save_analysis(session, {
                    "route_id": f"{route_id}_route{idx}",
                    "origin": origin_data,
                    "destination": dest_data,
                    "travel_time_s": summary["travel_time_s"],
                    "no_traffic_s": summary["no_traffic_s"],
                    "delay_s": summary["delay_s"],
                    "length_m": summary["length_m"],
                    "calculated_cost": cost,
                    "ml_predicted": ml_predicted,
                    "raw_json": route,
                    "user_id": current_user.id if current_user else None
                })
                session.close()
            except Exception as e:
                logger.error(f"Database save error: {e}")
        
        # Find best route (lowest cost)
        best_route = min(analyzed_routes, key=lambda x: x["calculated_cost"])
        
        return {
            "origin": origin_data,
            "destination": dest_data,
            "analyzed_routes": analyzed_routes,
            "best_route_index": best_route["route_index"],
            "best_route": best_route,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route analysis failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": ML_MODEL is not None,
        "timestamp": datetime.now(UTC).isoformat()
    }


@app.get("/analysis-report")
async def serve_analysis_report():
    """Serve the analysis report HTML page."""
    report_path = os.path.join("templates", "analysis_report.html")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>Analysis Report</h1><p>Report page not found.</p>",
        status_code=404
    )


@app.get("/api/route-analysis/{route_id}")
async def get_route_analysis(route_id: str, route_index: Optional[int] = None):
    """
    Get detailed analysis data for a specific route.
    
    Args:
        route_id: Route identifier (e.g., "Origin→Destination")
        route_index: Optional route index to filter specific route variant
        
    Returns:
        Analysis data with historical trends and statistics
    """
    try:
        import json
        from fastapi.responses import JSONResponse
        
        session = get_session()
        
        # Build query - route_id format is "Origin→Destination_route{idx}"
        # Remove URL encoding if present
        route_id = route_id.replace('%E2%86%92', '→')
        
        if route_index is not None:
            # Match specific route: "Origin→Destination_route{route_index}"
            query = session.query(AnalysisResult).filter(
                AnalysisResult.route_id == f"{route_id}_route{route_index}"
            )
        else:
            # Match all routes for this origin→destination
            query = session.query(AnalysisResult).filter(
                AnalysisResult.route_id.like(f"{route_id}_route%")
            )
        
        # Get all results
        results = query.order_by(AnalysisResult.timestamp.desc()).all()
        session.close()
        
        if not results:
            raise HTTPException(status_code=404, detail="No analysis data found for this route")
        
        # Parse results
        analysis_data = []
        for r in results:
            try:
                origin = json.loads(r.origin) if r.origin.startswith('{') else {"name": r.origin}
                dest = json.loads(r.destination) if r.destination.startswith('{') else {"name": r.destination}
            except:
                origin = {"name": r.origin}
                dest = {"name": r.destination}
            
            # Calculate delay if not provided
            delay_val = r.delay_s
            if delay_val is None or delay_val == 0:
                if r.travel_time_s and r.no_traffic_s:
                    delay_val = max(0, r.travel_time_s - r.no_traffic_s)
                else:
                    delay_val = 0
            
            analysis_data.append({
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "route_id": r.route_id,
                "origin": origin,
                "destination": dest,
                "travel_time_s": r.travel_time_s,
                "no_traffic_s": r.no_traffic_s,
                "delay_s": delay_val,
                "length_m": r.length_m,
                "calculated_cost": r.calculated_cost,
                "ml_predicted": r.ml_predicted,
                "congestion_ratio": (r.travel_time_s / r.no_traffic_s) if r.no_traffic_s and r.no_traffic_s > 0 else None
            })
        
        # Calculate statistics
        if analysis_data:
            travel_times = [d["travel_time_s"] for d in analysis_data if d["travel_time_s"]]
            
            # Calculate delays - use provided delay_s or calculate from travel_time - no_traffic_time
            delays = []
            for d in analysis_data:
                delay_val = d.get("delay_s")
                if delay_val is None or delay_val == 0:
                    # Calculate delay manually if not provided
                    if d.get("travel_time_s") and d.get("no_traffic_s"):
                        delay_val = max(0, d["travel_time_s"] - d["no_traffic_s"])
                    else:
                        delay_val = 0
                if delay_val > 0:
                    delays.append(delay_val)
            
            costs = [d["calculated_cost"] for d in analysis_data if d.get("calculated_cost")]
            congestion_ratios = [d["congestion_ratio"] for d in analysis_data if d.get("congestion_ratio")]
            
            stats = {
                "avg_travel_time": sum(travel_times) / len(travel_times) if travel_times else 0,
                "avg_delay": sum(delays) / len(delays) if delays else 0,
                "avg_cost": sum(costs) / len(costs) if costs else 0,
                "avg_congestion": sum(congestion_ratios) / len(congestion_ratios) if congestion_ratios else 0,
                "min_travel_time": min(travel_times) if travel_times else 0,
                "max_travel_time": max(travel_times) if travel_times else 0,
                "total_records": len(analysis_data)
            }
        else:
            stats = {}
        
        # Create response with no-cache headers to ensure fresh data
        response_data = {
            "route_id": route_id,
            "route_index": route_index,
            "analysis_data": analysis_data,
            "statistics": stats,
            "latest": analysis_data[0] if analysis_data else None,
            "fetched_at": datetime.now(UTC).isoformat()  # Add timestamp when data was fetched
        }
        
        # Return with no-cache headers
        response = JSONResponse(content=response_data)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")


@app.post("/api/refresh-route")
async def refresh_route_analysis(
    request: RouteAnalysisRequest,
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Refresh route analysis with latest data from TomTom API.
    Similar to analyze_route but returns data in format suitable for analysis report.
    """
    try:
        # Reuse analyze_route logic
        if isinstance(request.origin, str):
            o_lat, o_lon = tomtom_geocode(request.origin)
            origin_data = {"name": request.origin, "lat": o_lat, "lon": o_lon}
        else:
            o_lat = float(request.origin.get("lat"))
            o_lon = float(request.origin.get("lon"))
            origin_data = request.origin
        
        if isinstance(request.destination, str):
            d_lat, d_lon = tomtom_geocode(request.destination)
            dest_data = {"name": request.destination, "lat": d_lat, "lon": d_lon}
        else:
            d_lat = float(request.destination.get("lat"))
            d_lon = float(request.destination.get("lon"))
            dest_data = request.destination
        
        route_json = tomtom_route(
            o_lat, o_lon, d_lat, d_lon,
            maxAlternatives=request.maxAlternatives
        )
        
        routes = route_json.get("routes", [])
        if not routes:
            raise HTTPException(status_code=404, detail="No routes found")
        
        analyzed_routes = []
        route_id = f"{origin_data.get('name', f'{o_lat},{o_lon}')}→{dest_data.get('name', f'{d_lat},{d_lon}')}"
        
        for idx, route in enumerate(routes):
            summary = summarize_route(route)
            
            if summary["length_m"] == 0:
                summary["length_m"] = haversine_m(o_lat, o_lon, d_lat, d_lon)
            
            cost = compute_route_cost(
                summary["travel_time_s"],
                summary["no_traffic_s"],
                summary["delay_s"],
                summary["length_m"],
                alpha=request.alpha,
                beta=request.beta,
                gamma=request.gamma
            )
            
            ml_predicted = predict_congestion({
                "distance_km": summary["length_m"] / 1000.0,
                "route_index": idx,
                "travel_time_s": summary["travel_time_s"],
                "no_traffic_s": summary["no_traffic_s"],
                "delay_s": summary["delay_s"]
            })
            
            congestion_ratio = (
                summary["travel_time_s"] / summary["no_traffic_s"]
                if summary["no_traffic_s"] and summary["no_traffic_s"] > 0
                else None
            )
            
            # Always calculate delay from travel_time and no_traffic_time
            # Don't rely on API-provided delay_s which might be 0
            calculated_delay = 0
            if summary["travel_time_s"] and summary["no_traffic_s"]:
                calculated_delay = max(0, summary["travel_time_s"] - summary["no_traffic_s"])
            elif summary.get("delay_s"):
                calculated_delay = summary["delay_s"]
            
            analyzed_route = {
                "route_index": idx,
                "travel_time_s": summary["travel_time_s"],
                "no_traffic_s": summary["no_traffic_s"],
                "delay_s": calculated_delay,
                "length_m": summary["length_m"],
                "congestion_ratio": congestion_ratio,
                "calculated_cost": cost,
                "ml_predicted_congestion": ml_predicted
            }
            analyzed_routes.append(analyzed_route)
            
            # Save to database
            try:
                session = get_session()
                save_analysis(session, {
                    "route_id": f"{route_id}_route{idx}",
                    "origin": origin_data,
                    "destination": dest_data,
                    "travel_time_s": summary["travel_time_s"],
                    "no_traffic_s": summary["no_traffic_s"],
                    "delay_s": summary["delay_s"],
                    "length_m": summary["length_m"],
                    "calculated_cost": cost,
                    "ml_predicted": ml_predicted,
                    "raw_json": route,
                    "user_id": current_user.id if current_user else None
                })
                session.close()
            except Exception as e:
                logger.error(f"Database save error: {e}")
        
        best_route = min(analyzed_routes, key=lambda x: x["calculated_cost"])
        
        return {
            "origin": origin_data,
            "destination": dest_data,
            "route_id": route_id,
            "analyzed_routes": analyzed_routes,
            "best_route_index": best_route["route_index"],
            "best_route": best_route,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route refresh failed: {str(e)}")


# ============================================================================
# USER AUTHENTICATION & MANAGEMENT (Feature 1)
# ============================================================================

@app.post("/api/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_session)):
    """Register a new user."""
    try:
        # Validate password before creating user
        if len(user_data.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
        
        # Maximum password length check (reasonable limit to prevent abuse)
        MAX_PASSWORD_LENGTH = 10000
        if len(user_data.password) > MAX_PASSWORD_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password is too long. Maximum {MAX_PASSWORD_LENGTH} characters allowed."
            )
        
        user = create_user(db, AuthUserCreate(**user_data.dict()))
        return UserResponse.model_validate(user)
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Handle any remaining password-related errors
        if "password" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation error: {error_msg}"
            )
        raise HTTPException(status_code=500, detail=f"Registration failed: {error_msg}")


@app.post("/api/auth/login", response_model=Token)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_session)):
    """Login and get access token."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30 * 24 * 60)  # 30 days
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return UserResponse.model_validate(current_user)


# ============================================================================
# SAVED ROUTES (Feature 1)
# ============================================================================

@app.post("/api/saved-routes")
async def create_saved_route(
    route_data: SavedRouteCreate,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_session)
):
    """Save a route for the current user (optional auth)."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please login to save routes. Login is optional but required for this feature."
        )
    import json
    
    origin_str = json.dumps(route_data.origin) if isinstance(route_data.origin, dict) else route_data.origin
    dest_str = json.dumps(route_data.destination) if isinstance(route_data.destination, dict) else route_data.destination
    
    saved_route = SavedRoute(
        user_id=current_user.id,
        route_name=route_data.route_name,
        origin=origin_str,
        destination=dest_str,
        route_preferences=route_data.route_preferences,
        is_favorite=False,
        share_token=secrets.token_urlsafe(16)
    )
    db.add(saved_route)
    db.commit()
    db.refresh(saved_route)
    return saved_route


@app.get("/api/saved-routes")
async def get_saved_routes(
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_session),
    favorites_only: bool = Query(False)
):
    """Get saved routes for current user (optional auth)."""
    if not current_user:
        return []  # Return empty list if not logged in
    query = db.query(SavedRoute).filter(SavedRoute.user_id == current_user.id)
    if favorites_only:
        query = query.filter(SavedRoute.is_favorite == True)
    routes = query.order_by(SavedRoute.last_used.desc()).all()
    return routes


@app.put("/api/saved-routes/{route_id}/favorite")
async def toggle_favorite(
    route_id: int,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_session)
):
    """Toggle favorite status of a saved route (optional auth)."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required for this feature")
    route = db.query(SavedRoute).filter(
        SavedRoute.id == route_id,
        SavedRoute.user_id == current_user.id
    ).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    route.is_favorite = not route.is_favorite
    db.commit()
    return {"is_favorite": route.is_favorite}


@app.delete("/api/saved-routes/{route_id}")
async def delete_saved_route(
    route_id: int,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_session)
):
    """Delete a saved route (optional auth)."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required for this feature")
    route = db.query(SavedRoute).filter(
        SavedRoute.id == route_id,
        SavedRoute.user_id == current_user.id
    ).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    db.delete(route)
    db.commit()
    return {"message": "Route deleted"}


@app.get("/api/share-route/{share_token}")
async def get_shared_route(share_token: str, db: Session = Depends(get_session)):
    """Get a shared route by token."""
    route = db.query(SavedRoute).filter(SavedRoute.share_token == share_token).first()
    if not route:
        raise HTTPException(status_code=404, detail="Shared route not found")
    return route


# ============================================================================
# ADVANCED ANALYTICS (Feature 2)
# ============================================================================

@app.get("/api/analytics/peak-hours/{route_id}")
async def get_peak_hours(
    route_id: str,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_session)
):
    """Get peak hours analysis for a route."""
    return get_peak_hours_analysis(db, route_id, days)


@app.get("/api/analytics/day-of-week/{route_id}")
async def get_day_analysis(
    route_id: str,
    days: int = Query(90, ge=1, le=365),
    db: Session = Depends(get_session)
):
    """Get day of week analysis."""
    return get_day_of_week_analysis(db, route_id, days)


@app.get("/api/analytics/seasonal/{route_id}")
async def get_seasonal_analysis(
    route_id: str,
    months: int = Query(12, ge=1, le=24),
    db: Session = Depends(get_session)
):
    """Get seasonal trends."""
    return get_seasonal_trends(db, route_id, months)


@app.get("/api/analytics/reliability/{route_id}")
async def get_reliability(
    route_id: str,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_session)
):
    """Get route reliability score."""
    return calculate_route_reliability(db, route_id, days)


@app.get("/api/analytics/predict/{route_id}")
async def get_prediction(
    route_id: str,
    hours_ahead: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_session)
):
    """Predict future congestion."""
    return predict_future_congestion(db, route_id, hours_ahead)


@app.get("/api/analytics/hotspots")
async def get_hotspots(
    days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_session)
):
    """Get traffic hotspots."""
    return get_traffic_hotspots(db, days)


# ============================================================================
# EXPORT & REPORTING (Feature 5)
# ============================================================================

@app.get("/api/export/csv/{route_id}")
async def export_csv(
    route_id: str,
    db: Session = Depends(get_session)
):
    """Export route data to CSV."""
    csv_content = export_to_csv(db, route_id)
    return StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=route_{route_id}_{datetime.now(UTC).strftime('%Y%m%d')}.csv"}
    )


@app.get("/api/export/excel/{route_id}")
async def export_excel(
    route_id: str,
    db: Session = Depends(get_session)
):
    """Export route data to Excel."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        export_to_excel(db, route_id, tmp.name)
        return FileResponse(
            tmp.name,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"route_{route_id}_{datetime.now(UTC).strftime('%Y%m%d')}.xlsx"
        )


@app.get("/api/admin/export/users/csv")
async def export_users_csv(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session)
):
    """Export all users to CSV (admin only). Saves to exports folder and streams for download."""
    import csv
    import io
    
    users = db.query(User).order_by(User.created_at.desc()).all()
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        'id', 'email', 'username', 'full_name', 'is_active', 
        'is_admin', 'created_at', 'last_login'
    ])
    writer.writeheader()
    
    for user in users:
        writer.writerow({
            'id': user.id,
            'email': user.email,
            'username': user.username,
            'full_name': user.full_name or '',
            'is_active': 'Yes' if user.is_active else 'No',
            'is_admin': 'Yes' if user.is_admin else 'No',
            'created_at': user.created_at.isoformat() if user.created_at else '',
            'last_login': user.last_login.isoformat() if user.last_login else ''
        })
    
    csv_content = output.getvalue()
    output.close()
    
    # Save to exports folder in project directory
    exports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    filename = f"users_export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(exports_dir, filename)
    
    # Save file to exports folder
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        f.write(csv_content)
    
    logger.info(f"Exported users CSV saved to: {filepath}")
    
    # Also stream for download
    return StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/admin/export/routes/csv")
async def export_routes_csv(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session)
):
    """Export all route analyses to CSV (admin only). Saves to exports folder and streams for download."""
    import csv
    import io
    import json
    
    routes = db.query(AnalysisResult).order_by(AnalysisResult.timestamp.desc()).all()
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        'id', 'route_id', 'timestamp', 'user_id', 'origin', 'destination',
        'travel_time_min', 'delay_min', 'distance_km', 'congestion_ratio',
        'cost', 'ml_prediction'
    ])
    writer.writeheader()
    
    for route in routes:
        # Parse origin and destination JSON
        origin_str = ''
        dest_str = ''
        try:
            if route.origin:
                origin_data = json.loads(route.origin) if isinstance(route.origin, str) else route.origin
                origin_str = origin_data.get('name', '') or f"{origin_data.get('lat', '')}, {origin_data.get('lon', '')}"
            if route.destination:
                dest_data = json.loads(route.destination) if isinstance(route.destination, str) else route.destination
                dest_str = dest_data.get('name', '') or f"{dest_data.get('lat', '')}, {dest_data.get('lon', '')}"
        except:
            origin_str = str(route.origin) if route.origin else ''
            dest_str = str(route.destination) if route.destination else ''
        
        writer.writerow({
            'id': route.id,
            'route_id': route.route_id,
            'timestamp': route.timestamp.isoformat() if route.timestamp else '',
            'user_id': route.user_id or '',
            'origin': origin_str,
            'destination': dest_str,
            'travel_time_min': round(route.travel_time_s / 60, 2) if route.travel_time_s else '',
            'delay_min': round(route.delay_s / 60, 2) if route.delay_s else '',
            'distance_km': round(route.length_m / 1000, 2) if route.length_m else '',
            'congestion_ratio': round(route.travel_time_s / route.no_traffic_s, 2) if route.no_traffic_s and route.no_traffic_s > 0 else '',
            'cost': round(route.calculated_cost, 2) if route.calculated_cost else '',
            'ml_prediction': round(route.ml_predicted, 2) if route.ml_predicted else ''
        })
    
    csv_content = output.getvalue()
    output.close()
    
    # Save to exports folder in project directory
    exports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    filename = f"routes_export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(exports_dir, filename)
    
    # Save file to exports folder
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        f.write(csv_content)
    
    logger.info(f"Exported routes CSV saved to: {filepath}")
    
    # Also stream for download
    return StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/admin/export/system/json")
async def export_system_json(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session)
):
    """Export system statistics to JSON (admin only)."""
    import json
    
    # Get admin stats directly
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    admin_users = db.query(User).filter(User.is_admin == True).count()
    total_routes = db.query(AnalysisResult).count()
    total_saved_routes = db.query(SavedRoute).count()
    total_ratings = db.query(RouteRating).count()
    total_notifications = db.query(Notification).count()
    
    # Get cache stats
    from cache_utils import get_cache_stats
    cache_stats = get_cache_stats()
    
    # Get recent analyses
    recent_analyses = db.query(AnalysisResult).order_by(AnalysisResult.timestamp.desc()).limit(10).all()
    
    # Compile comprehensive system report
    system_report = {
        "export_timestamp": datetime.now(UTC).isoformat(),
        "system_statistics": {
            "total_users": total_users,
            "active_users": active_users,
            "admin_users": admin_users,
            "total_routes": total_routes,
            "total_analyses": total_routes,
            "total_saved_routes": total_saved_routes,
            "total_ratings": total_ratings,
            "total_notifications": total_notifications
        },
        "cache_statistics": cache_stats,
        "database_info": {
            "database_type": "SQLite",
            "database_path": os.getenv("DB_PATH", "traffic_analysis.db")
        },
        "ml_model_info": {
            "model_loaded": ML_MODEL is not None,
            "model_path": MODEL_PATH if ML_MODEL else None
        },
        "recent_activity": [
            {
                "route_id": r.route_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "travel_time": r.travel_time_s,
                "cost": r.calculated_cost
            }
            for r in recent_analyses
        ]
    }
    
    json_content = json.dumps(system_report, indent=2, ensure_ascii=False)
    
    # Save to exports folder in project directory
    exports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    filename = f"system_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(exports_dir, filename)
    
    # Save file to exports folder
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_content)
    
    logger.info(f"Exported system report JSON saved to: {filepath}")
    
    # Also stream for download
    return StreamingResponse(
        io.BytesIO(json_content.encode('utf-8')),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/export/pdf/{route_id}")
async def export_pdf(
    route_id: str,
    db: Session = Depends(get_session)
):
    """Export route data to PDF."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        export_to_pdf(db, route_id, tmp.name)
        return FileResponse(
            tmp.name,
            media_type="application/pdf",
            filename=f"route_{route_id}_{datetime.now(UTC).strftime('%Y%m%d')}.pdf"
        )


# ============================================================================
# NOTIFICATIONS (Feature 4)
# ============================================================================

@app.get("/api/notifications")
async def get_notifications(
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=100),
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_session)
):
    """Get user notifications (optional auth)."""
    if not current_user:
        return []  # Return empty if not logged in
    return get_user_notifications(db, current_user.id, unread_only, limit)


@app.put("/api/notifications/{notification_id}/read")
async def mark_read(
    notification_id: int,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_session)
):
    """Mark notification as read (optional auth)."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required")
    success = mark_notification_read(db, notification_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Notification marked as read"}


@app.post("/api/notifications/check-alerts")
async def check_alerts(
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_session)
):
    """Check for traffic alerts on saved routes (optional auth)."""
    if not current_user:
        return {"alerts": 0, "notifications": []}
    alerts = check_traffic_alerts(db, current_user.id)
    return {"alerts": len(alerts), "notifications": alerts}


# ============================================================================
# REAL-TIME FEATURES (Feature 6)
# ============================================================================

@app.get("/api/realtime/incidents")
async def get_incidents(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    radius: int = Query(5000, ge=100, le=50000)
):
    """Get traffic incidents near a location."""
    return get_traffic_incidents(lat, lon, radius)


@app.post("/api/realtime/monitor/{route_id}")
async def monitor_route(
    route_id: str,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_session)
):
    """Monitor route for changes (works without login)."""
    change = monitor_route_changes(db, route_id)
    if change:
        # Create notification if user is logged in
        if current_user:
            from notifications import create_notification
            create_notification(
                db, current_user.id, 'traffic_alert',
                f"Route Change: {route_id}",
                f"Route travel time changed by {change['change_percent']}%",
                route_id
            )
    return change or {"message": "No significant changes detected"}


# ============================================================================
# ROUTE RATINGS & SOCIAL (Feature 9)
# ============================================================================

@app.post("/api/ratings")
async def create_rating(
    rating_data: RouteRatingCreate,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_session)
):
    """Rate a route (optional auth - works without login)."""
    # Allow anonymous ratings by using a default user_id of 0 or None
    user_id = current_user.id if current_user else None
    rating = RouteRating(
        user_id=user_id,
        route_id=rating_data.route_id,
        rating=rating_data.rating,
        review=rating_data.review
    )
    db.add(rating)
    db.commit()
    db.refresh(rating)
    return rating


@app.get("/api/ratings/{route_id}")
async def get_ratings(route_id: str, db: Session = Depends(get_session)):
    """Get ratings for a route."""
    ratings = db.query(RouteRating).filter(RouteRating.route_id == route_id).all()
    if not ratings:
        return {"average_rating": 0, "count": 0, "ratings": []}
    
    avg_rating = sum(r.rating for r in ratings) / len(ratings)
    return {
        "average_rating": round(avg_rating, 2),
        "count": len(ratings),
        "ratings": ratings
    }


# ============================================================================
# ADMIN DASHBOARD (Feature 11)
# ============================================================================

@app.get("/api/admin/stats")
async def get_admin_stats(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session)
):
    """Get admin statistics (admin only)."""
    total_users = db.query(User).count()
    total_routes = db.query(AnalysisResult).count()
    total_saved_routes = db.query(SavedRoute).count()
    total_ratings = db.query(RouteRating).count()
    
    # Get recent activity
    recent_routes = db.query(AnalysisResult).order_by(AnalysisResult.timestamp.desc()).limit(10).all()
    recent_users = db.query(User).order_by(User.created_at.desc()).limit(5).all()
    
    return {
        "total_users": total_users,
        "total_route_analyses": total_routes,
        "total_saved_routes": total_saved_routes,
        "total_ratings": total_ratings,
        "cache_stats": get_cache_stats(),
        "recent_activity": {
            "routes": len(recent_routes),
            "new_users": len(recent_users)
        }
    }


@app.get("/api/admin/route-analysis")
async def get_all_route_analyses(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session),
    filter_period: Optional[str] = Query(None, alias="filter"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get all route analyses with optional filtering (admin only)."""
    import json
    from datetime import datetime, timedelta, UTC
    
    try:
        # Build query
        query = db.query(AnalysisResult)
        
        # Apply time filter if specified
        if filter_period:
            now = datetime.now(UTC)
            if filter_period == "today":
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(AnalysisResult.timestamp >= start_date)
            elif filter_period == "week":
                start_date = now - timedelta(days=7)
                query = query.filter(AnalysisResult.timestamp >= start_date)
            elif filter_period == "month":
                start_date = now - timedelta(days=30)
                query = query.filter(AnalysisResult.timestamp >= start_date)
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply pagination and ordering
        routes = query.order_by(AnalysisResult.timestamp.desc()).offset(skip).limit(limit).all()
        
        # Format response
        route_data = []
        for r in routes:
            try:
                origin = json.loads(r.origin) if isinstance(r.origin, str) and r.origin.startswith('{') else {"name": str(r.origin) if r.origin else ""}
                dest = json.loads(r.destination) if isinstance(r.destination, str) and r.destination.startswith('{') else {"name": str(r.destination) if r.destination else ""}
            except:
                origin = {"name": str(r.origin) if r.origin else ""}
                dest = {"name": str(r.destination) if r.destination else ""}
            
            route_name = f"{origin.get('name', '')} → {dest.get('name', '')}"
            
            # Calculate delay if not provided or is zero
            delay_val = r.delay_s
            if delay_val is None or delay_val == 0:
                if r.travel_time_s and r.no_traffic_s:
                    delay_val = max(0, r.travel_time_s - r.no_traffic_s)
                else:
                    delay_val = 0
            
            route_data.append({
                "id": r.id,
                "route": route_name,
                "route_id": r.route_id,
                "travel_time_s": r.travel_time_s,
                "delay_s": delay_val,
                "length_m": r.length_m,
                "calculated_cost": r.calculated_cost,
                "ml_predicted": r.ml_predicted,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "origin": origin,
                "destination": dest
            })
        
        # Calculate statistics from filtered results
        all_routes_for_stats = query.all()
        
        if all_routes_for_stats:
            travel_times = [r.travel_time_s for r in all_routes_for_stats if r.travel_time_s is not None]
            # Calculate delays - use provided delay_s or calculate from travel_time - no_traffic_time
            delays = []
            for r in all_routes_for_stats:
                delay_val = r.delay_s
                if delay_val is None or delay_val == 0:
                    # Calculate delay manually if not provided
                    if r.travel_time_s and r.no_traffic_s:
                        delay_val = max(0, r.travel_time_s - r.no_traffic_s)
                    else:
                        delay_val = 0
                if delay_val > 0:
                    delays.append(delay_val)
            costs = [r.calculated_cost for r in all_routes_for_stats if r.calculated_cost is not None]
            
            stats = {
                "total": total_count,
                "avg_travel_time": sum(travel_times) / len(travel_times) if travel_times else 0,
                "avg_delay": sum(delays) / len(delays) if delays else 0,
                "avg_cost": sum(costs) / len(costs) if costs else 0
            }
        else:
            stats = {
                "total": 0,
                "avg_travel_time": 0,
                "avg_delay": 0,
                "avg_cost": 0
            }
        
        return {
            "routes": route_data,
            "stats": stats,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching route analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch route analyses: {str(e)}")


@app.get("/api/admin/users")
async def get_all_users(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get all users (admin only)."""
    users = db.query(User).offset(skip).limit(limit).all()
    return [UserResponse.model_validate(u) for u in users]


@app.put("/api/admin/users/{user_id}/activate")
async def toggle_user_status(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session)
):
    """Activate/deactivate a user (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_active = not user.is_active
    db.commit()
    return {"is_active": user.is_active, "message": f"User {'activated' if user.is_active else 'deactivated'}"}


@app.put("/api/admin/users/{user_id}/admin")
async def toggle_admin_status(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session)
):
    """Grant/revoke admin privileges (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot change your own admin status")
    
    user.is_admin = not user.is_admin
    db.commit()
    return {"is_admin": user.is_admin, "message": f"Admin privileges {'granted' if user.is_admin else 'revoked'}"}


class UserUpdate(BaseModel):
    """User update model for admin editing."""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8)


@app.put("/api/admin/users/{user_id}")
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session)
):
    """Update user details (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Note: We'll check self-demotion later when actually updating is_admin
    
    # Update username if provided
    if user_update.username is not None:
        # Check if username is already taken by another user
        existing_user = db.query(User).filter(User.username == user_update.username, User.id != user_id).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")
        user.username = user_update.username
    
    # Update email if provided
    if user_update.email is not None:
        # Check if email is already taken by another user
        existing_user = db.query(User).filter(User.email == user_update.email, User.id != user_id).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already taken")
        user.email = user_update.email
    
    # Update full name if provided
    if user_update.full_name is not None:
        user.full_name = user_update.full_name
    
    # Update active status if provided
    if user_update.is_active is not None:
        user.is_active = user_update.is_active
    
    # Update admin status if provided
    if user_update.is_admin is not None:
        # Prevent self-demotion
        if user.id == current_user.id and not user_update.is_admin:
            raise HTTPException(status_code=400, detail="Cannot remove your own admin privileges")
        user.is_admin = user_update.is_admin
    
    # Update password if provided and not empty
    if user_update.password is not None and user_update.password.strip() != "":
        if len(user_update.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        user.hashed_password = get_password_hash(user_update.password)
    
    try:
        db.commit()
        db.refresh(user)
        return UserResponse.model_validate(user)
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")


@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_session)
):
    """Delete a user (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}


@app.get("/api/user/stats")
async def get_user_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_session)
):
    """Get user-specific statistics."""
    # Count user's saved routes
    saved_routes_count = db.query(SavedRoute).filter(SavedRoute.user_id == current_user.id).count()
    
    # Count user's route analyses
    analyses_count = db.query(AnalysisResult).filter(AnalysisResult.user_id == current_user.id).count()
    
    # Count user's ratings
    ratings_count = db.query(RouteRating).filter(RouteRating.user_id == current_user.id).count()
    
    # Get recent analyses
    recent_analyses = db.query(AnalysisResult).filter(
        AnalysisResult.user_id == current_user.id
    ).order_by(AnalysisResult.timestamp.desc()).limit(10).all()
    
    return {
        "saved_routes": saved_routes_count,
        "analyses": analyses_count,
        "ratings": ratings_count,
        "recent_analyses": [
            {
                "route_id": r.route_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "travel_time": r.travel_time_s,
                "cost": r.calculated_cost
            }
            for r in recent_analyses
        ]
    }


# ============================================================================
# CACHE MANAGEMENT (Feature 10)
# ============================================================================

@app.post("/api/cache/clear")
async def clear_route_cache(
    pattern: Optional[str] = None,
    current_user: User = Depends(get_current_admin_user)
):
    """Clear route cache (admin only)."""
    clear_cache(pattern=pattern)
    return {"message": "Cache cleared"}


@app.get("/api/cache/stats")
async def get_cache_statistics():
    """Get cache statistics."""
    return get_cache_stats()



# Integration endpoints
# Integration endpoints to append to app.py

@app.get("/api/integration/navigation/{route_id}")
async def get_navigation_links(
    route_id: str,
    route_index: int = Query(0),
    db: Session = Depends(get_session)
):
    """Get navigation app links (Google Maps, Waze)."""
    result = db.query(AnalysisResult).filter(
        AnalysisResult.route_id.like(f"{route_id}%")
    ).order_by(AnalysisResult.timestamp.desc()).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Route not found")
    
    import json
    try:
        origin = json.loads(result.origin) if result.origin.startswith('{') else {"name": result.origin}
        dest = json.loads(result.destination) if result.destination.startswith('{') else {"name": result.destination}
    except:
        origin = {"name": result.origin}
        dest = {"name": result.destination}
    
    origin_lat = origin.get('lat', 0)
    origin_lon = origin.get('lon', 0)
    dest_lat = dest.get('lat', 0)
    dest_lon = dest.get('lon', 0)
    
    google_maps = f"https://www.google.com/maps/dir/{origin_lat},{origin_lon}/{dest_lat},{dest_lon}"
    waze = f"https://waze.com/ul?ll={dest_lat},{dest_lon}&navigate=yes"
    
    return {
        "google_maps": google_maps,
        "waze": waze,
        "origin": origin,
        "destination": dest
    }

