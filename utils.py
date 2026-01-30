"""
Utility functions for TomTom API integration and route analysis.
Provides unified functions for geocoding, routing, distance calculation, and route summarization.
"""

import os
import math
import time
import requests
from requests.adapters import HTTPAdapter, Retry
from dotenv import load_dotenv

load_dotenv()

TOMTOM_KEY = os.getenv("TOMTOM_KEY")

# Session with retry logic for network calls
_session = requests.Session()
_retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=["GET", "POST"]
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))


def tomtom_geocode(query: str, timeout: int = 10, country_set: str = "IN") -> tuple[float, float]:
    """
    Geocode a place name to latitude and longitude using TomTom Search API.
    
    Args:
        query: Place name or address to geocode
        timeout: Request timeout in seconds
        country_set: Country code to prioritize (default: "IN" for India)
        
    Returns:
        Tuple of (latitude, longitude)
        
    Raises:
        ValueError: If no results found or API key missing
        requests.RequestException: If API call fails
    """
    if not TOMTOM_KEY:
        raise ValueError("TOMTOM_KEY not set in environment")
    
    # Use search API for better global results
    url = "https://api.tomtom.com/search/2/search/.json"
    try:
        # Add country set to prioritize India, but allow global results
        params = {
            "key": TOMTOM_KEY,
            "query": query,
            "limit": 5,  # Get more results to find the best match
            "countrySet": country_set
        }
        resp = _session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if not results:
            raise ValueError(f"No geocode result for '{query}'")
        
        # Prefer results from India (country code IN)
        best_result = None
        for result in results:
            address = result.get("address", {})
            country_code = address.get("countryCode", "").upper()
            if country_code == "IN":
                best_result = result
                break
        
        # If no India result, use first result but validate coordinates
        if not best_result:
            best_result = results[0]
        
        pos = best_result.get("position", {})
        if not pos or "lat" not in pos or "lon" not in pos:
            raise ValueError(f"No position data for '{query}'")
        
        lat = float(pos["lat"])
        lon = float(pos["lon"])
        
        # Validate coordinates are reasonable for India (rough bounds)
        # India is approximately: lat 6.5 to 37.1, lon 68.1 to 97.4
        if country_set == "IN":
            if lat < 6 or lat > 38 or lon < 65 or lon > 100:
                # Coordinates seem wrong for India, try without country restriction
                if len(results) > 1:
                    # Try second result
                    pos2 = results[1].get("position", {})
                    if pos2 and "lat" in pos2 and "lon" in pos2:
                        lat2 = float(pos2["lat"])
                        lon2 = float(pos2["lon"])
                        if 6 <= lat2 <= 38 and 65 <= lon2 <= 100:
                            return lat2, lon2
                # If still wrong, log warning but return anyway
                import logging
                logging.warning(f"Coordinates {lat}, {lon} seem outside India bounds for query '{query}'")
        
        return lat, lon
    except requests.RequestException as e:
        raise requests.RequestException(f"Geocoding failed: {str(e)}")


def tomtom_autocomplete(q: str, timeout: int = 10) -> list[dict]:
    """
    Get autocomplete suggestions from TomTom Search API.
    
    Args:
        q: Search query string
        timeout: Request timeout in seconds
        
    Returns:
        List of suggestion dictionaries with 'text', 'address', 'position' keys
        
    Raises:
        requests.RequestException: If API call fails
    """
    if not TOMTOM_KEY:
        raise ValueError("TOMTOM_KEY not set in environment")
    
    url = "https://api.tomtom.com/search/2/search/.json"
    try:
        # Use search API with proper parameters for global search
        # Simplified parameters to avoid 400 errors
        params = {
            "key": TOMTOM_KEY,
            "query": q,
            "limit": 10
        }
        resp = _session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        suggestions = []
        for r in results:
            pos = r.get("position", {})
            # Get the best display name
            address = r.get("address", {})
            poi = r.get("poi", {})
            
            # Try multiple fields for the display text
            display_text = (
                address.get("freeformAddress") or
                poi.get("name") or
                address.get("municipality") or
                address.get("municipalitySubdivision") or
                f"{address.get('municipality', '')}, {address.get('countrySubdivision', '')}".strip(", ")
            )
            
            # Add more context if available
            if address.get("municipality") and address.get("countrySubdivision"):
                if display_text and address.get("municipality") not in display_text:
                    display_text = f"{display_text}, {address.get('municipality')}, {address.get('countrySubdivision')}"
            
            if display_text:
                suggestions.append({
                    "text": display_text,
                    "address": address,
                    "position": {"lat": pos.get("lat"), "lon": pos.get("lon")},
                    "poi": poi
                })
        return suggestions
    except requests.RequestException as e:
        raise requests.RequestException(f"Autocomplete failed: {str(e)}")


def tomtom_route(
    olat: float,
    olon: float,
    dlat: float,
    dlon: float,
    maxAlternatives: int = 3,
    timeout: int = 15
) -> dict:
    """
    Fetch route alternatives from TomTom Routing API.
    
    Args:
        olat: Origin latitude
        olon: Origin longitude
        dlat: Destination latitude
        dlon: Destination longitude
        maxAlternatives: Maximum number of route alternatives (default: 3)
        timeout: Request timeout in seconds
        
    Returns:
        JSON response from TomTom Routing API
        
    Raises:
        ValueError: If API key missing
        requests.RequestException: If API call fails
    """
    if not TOMTOM_KEY:
        raise ValueError("TOMTOM_KEY not set in environment")
    
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{olat},{olon}:{dlat},{dlon}/json"
    params = {
        "key": TOMTOM_KEY,
        "maxAlternatives": maxAlternatives,
        "computeTravelTimeFor": "all"
    }
    try:
        resp = _session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise requests.RequestException(f"Routing failed: {str(e)}")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in meters.
    Uses the Haversine formula.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        Distance in meters
    """
    # Earth radius in meters
    R = 6371000
    
    # Convert to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (
        math.sin(delta_phi / 2) ** 2 +
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def summarize_route(route_json: dict) -> dict:
    """
    Extract summary metrics from a TomTom route JSON object.
    
    Args:
        route_json: A single route object from TomTom API response
        
    Returns:
        Dictionary with keys: travel_time_s, no_traffic_s, delay_s, length_m
        
    Raises:
        KeyError: If required fields are missing
    """
    summary = route_json.get("summary", {})
    travel_time_s = summary.get("travelTimeInSeconds")
    no_traffic_s = summary.get("noTrafficTravelTimeInSeconds")
    
    # Get delay from API, or calculate it if not provided
    delay_s = summary.get("trafficDelayInSeconds")
    if delay_s is None or delay_s == 0:
        # Calculate delay manually: delay = travel_time - no_traffic_time
        if travel_time_s is not None and no_traffic_s is not None:
            delay_s = max(0, travel_time_s - no_traffic_s)
        else:
            delay_s = 0
    
    length_m = summary.get("lengthInMeters", 0)
    
    return {
        "travel_time_s": travel_time_s,
        "no_traffic_s": no_traffic_s,
        "delay_s": delay_s,
        "length_m": length_m
    }


def compute_route_cost(
    travel_time_s: float,
    no_traffic_s: float,
    delay_s: float,
    distance_m: float,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.001,
    use_fuel_prices: bool = True
) -> float:
    """
    Compute route cost using fuel prices (INR) or weighted factors.
    
    If use_fuel_prices=True (default):
        Cost is calculated based on current petrol/diesel prices in INR,
        plus time and delay penalties.
        
    If use_fuel_prices=False:
        Uses legacy weighted formula:
        cost = alpha * (travel_time_s / 60) + beta * (delay_s / 60) + gamma * (distance_m / 1000)
    
    Args:
        travel_time_s: Total travel time in seconds
        no_traffic_s: Travel time without traffic in seconds
        delay_s: Traffic delay in seconds
        distance_m: Route distance in meters
        alpha: Weight for travel time (legacy mode, default: 1.0)
        beta: Weight for delay/congestion (legacy mode, default: 0.5)
        gamma: Weight for distance (legacy mode, default: 0.001)
        use_fuel_prices: If True, use fuel price-based calculation (default: True)
        
    Returns:
        Calculated cost value in INR (if use_fuel_prices=True) or unitless (if False)
    """
    if use_fuel_prices:
        try:
            from fuel_price import calculate_route_cost_with_fuel
            
            distance_km = distance_m / 1000.0
            travel_time_min = travel_time_s / 60.0
            delay_min = delay_s / 60.0
            
            # Calculate cost based on fuel prices
            cost = calculate_route_cost_with_fuel(
                distance_km=distance_km,
                travel_time_min=travel_time_min,
                delay_min=delay_min,
                fuel_type='petrol',  # Default to petrol, can be made configurable
                time_weight=0.1,     # 0.1 INR per minute of travel time
                delay_weight=0.05    # 0.05 INR per minute of delay
            )
            return cost
        except ImportError:
            # Fallback to legacy calculation if fuel_price module not available
            pass
        except Exception as e:
            # Log error and fallback to legacy calculation
            from logging_config import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Error calculating fuel-based cost: {e}. Using legacy calculation.")
    
    # Legacy calculation (weighted factors)
    time_cost = alpha * (travel_time_s / 60.0)  # minutes
    delay_cost = beta * (delay_s / 60.0)  # minutes
    distance_cost = gamma * (distance_m / 1000.0)  # kilometers
    
    return time_cost + delay_cost + distance_cost

