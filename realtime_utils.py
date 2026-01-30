"""
Real-time traffic monitoring and updates.
"""

import asyncio
from datetime import datetime, UTC, timedelta
from typing import Optional, Dict, List
from sqlalchemy.orm import Session
import requests
import os

from db import AnalysisResult, SavedRoute
from utils import tomtom_route, summarize_route
from logging_config import get_logger

logger = get_logger(__name__)
TOMTOM_KEY = os.getenv("TOMTOM_KEY")


def get_traffic_incidents(lat: float, lon: float, radius: int = 5000) -> List[Dict]:
    """Get traffic incidents near a location."""
    if not TOMTOM_KEY:
        return []
    
    try:
        url = f"https://api.tomtom.com/traffic/services/4/incidentDetails"
        params = {
            "key": TOMTOM_KEY,
            "point": f"{lat},{lon}",
            "radius": radius,
            "language": "en",
            "projection": "EPSG4326"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        incidents = []
        for incident in data.get("incidents", []):
            incidents.append({
                "id": incident.get("id"),
                "type": incident.get("type"),
                "severity": incident.get("properties", {}).get("iconCategory"),
                "description": incident.get("properties", {}).get("description"),
                "location": incident.get("geometry", {}).get("coordinates"),
                "start_time": incident.get("properties", {}).get("startTime"),
                "end_time": incident.get("properties", {}).get("endTime")
            })
        
        return incidents
    except Exception as e:
        logger.error(f"Error fetching traffic incidents: {e}")
        return []


async def auto_refresh_route(
    db: Session,
    route_id: str,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    interval_minutes: int = 15
) -> Dict:
    """Auto-refresh route data at intervals."""
    try:
        route_json = tomtom_route(origin_lat, origin_lon, dest_lat, dest_lon, maxAlternatives=1)
        routes = route_json.get("routes", [])
        
        if not routes:
            return {"status": "error", "message": "No routes found"}
        
        route = routes[0]
        summary = summarize_route(route)
        
        # Save to database
        from db import save_analysis
        save_analysis(db, {
            "route_id": route_id,
            "origin": {"lat": origin_lat, "lon": origin_lon},
            "destination": {"lat": dest_lat, "lon": dest_lon},
            "travel_time_s": summary["travel_time_s"],
            "no_traffic_s": summary["no_traffic_s"],
            "delay_s": summary["delay_s"],
            "length_m": summary["length_m"],
            "calculated_cost": 0,  # Will be calculated if needed
        })
        
        return {
            "status": "success",
            "travel_time_min": summary["travel_time_s"] / 60,
            "delay_min": summary["delay_s"] / 60,
            "timestamp": datetime.now(UTC).isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def monitor_route_changes(
    db: Session,
    route_id: str,
    threshold_percent: float = 20.0
) -> Optional[Dict]:
    """Monitor route for significant changes."""
    # Get last two analyses
    results = db.query(AnalysisResult).filter(
        AnalysisResult.route_id.like(f"{route_id}%")
    ).order_by(AnalysisResult.timestamp.desc()).limit(2).all()
    
    if len(results) < 2:
        return None
    
    latest = results[0]
    previous = results[1]
    
    # Check for significant change
    if latest.travel_time_s and previous.travel_time_s:
        change_percent = abs((latest.travel_time_s - previous.travel_time_s) / previous.travel_time_s) * 100
        
        if change_percent >= threshold_percent:
            return {
                "route_id": route_id,
                "change_percent": round(change_percent, 2),
                "previous_time": previous.travel_time_s / 60,
                "current_time": latest.travel_time_s / 60,
                "timestamp": latest.timestamp.isoformat() if latest.timestamp else None
            }
    
    return None
