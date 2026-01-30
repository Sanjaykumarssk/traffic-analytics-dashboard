"""
Advanced analytics and insights for route analysis.
"""

from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, extract, Float, case
import pandas as pd
import numpy as np
from db import AnalysisResult


def get_peak_hours_analysis(db: Session, route_id: str, days: int = 30) -> Dict:
    """Analyze peak hours for a route."""
    cutoff_date = datetime.now(UTC) - timedelta(days=days)
    
    results = db.query(
        AnalysisResult.hour_of_day,
        func.avg(AnalysisResult.travel_time_s).label('avg_travel_time'),
        func.avg(AnalysisResult.delay_s).label('avg_delay'),
        func.avg(
            cast(AnalysisResult.travel_time_s, Float) / 
            nullif(cast(AnalysisResult.no_traffic_s, Float), 0)
        ).label('avg_congestion'),
        func.count(AnalysisResult.id).label('count')
    ).filter(
        and_(
            AnalysisResult.route_id.like(f"{route_id}%"),
            AnalysisResult.timestamp >= cutoff_date
        )
    ).group_by(AnalysisResult.hour_of_day).all()
    
    if not results:
        return {"peak_hours": [], "off_peak_hours": [], "data": []}
    
    data = []
    for r in results:
        data.append({
            "hour": r.hour_of_day,
            "avg_travel_time": r.avg_travel_time / 60,  # Convert to minutes
            "avg_delay": r.avg_delay / 60,
            "avg_congestion": r.avg_congestion,
            "count": r.count
        })
    
    # Find peak hours (top 3 hours with highest travel time)
    sorted_data = sorted(data, key=lambda x: x['avg_travel_time'], reverse=True)
    peak_hours = [d['hour'] for d in sorted_data[:3]]
    off_peak_hours = [d['hour'] for d in sorted_data[-3:]]
    
    return {
        "peak_hours": peak_hours,
        "off_peak_hours": off_peak_hours,
        "data": data,
        "best_hour": sorted_data[-1]['hour'] if sorted_data else None,
        "worst_hour": sorted_data[0]['hour'] if sorted_data else None
    }


def get_day_of_week_analysis(db: Session, route_id: str, days: int = 90) -> Dict:
    """Analyze day of week patterns."""
    cutoff_date = datetime.now(UTC) - timedelta(days=days)
    
    results = db.query(
        AnalysisResult.day_of_week,
        func.avg(AnalysisResult.travel_time_s).label('avg_travel_time'),
        func.avg(AnalysisResult.delay_s).label('avg_delay'),
        func.avg(AnalysisResult.calculated_cost).label('avg_cost'),
        func.count(AnalysisResult.id).label('count')
    ).filter(
        and_(
            AnalysisResult.route_id.like(f"{route_id}%"),
            AnalysisResult.timestamp >= cutoff_date
        )
    ).group_by(AnalysisResult.day_of_week).all()
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    data = []
    for r in results:
        data.append({
            "day": day_names[r.day_of_week] if r.day_of_week is not None else 'Unknown',
            "day_index": r.day_of_week,
            "avg_travel_time": r.avg_travel_time / 60,
            "avg_delay": r.avg_delay / 60,
            "avg_cost": r.avg_cost,
            "count": r.count
        })
    
    return {
        "data": data,
        "weekday_avg": sum([d['avg_travel_time'] for d in data if d['day_index'] < 5]) / max(len([d for d in data if d['day_index'] < 5]), 1),
        "weekend_avg": sum([d['avg_travel_time'] for d in data if d['day_index'] >= 5]) / max(len([d for d in data if d['day_index'] >= 5]), 1),
        "best_day": min(data, key=lambda x: x['avg_travel_time'])['day'] if data else None,
        "worst_day": max(data, key=lambda x: x['avg_travel_time'])['day'] if data else None
    }


def get_seasonal_trends(db: Session, route_id: str, months: int = 12) -> Dict:
    """Analyze seasonal/monthly trends."""
    cutoff_date = datetime.now(UTC) - timedelta(days=months * 30)
    
    results = db.query(
        AnalysisResult.month,
        func.avg(AnalysisResult.travel_time_s).label('avg_travel_time'),
        func.avg(AnalysisResult.delay_s).label('avg_delay'),
        func.count(AnalysisResult.id).label('count')
    ).filter(
        and_(
            AnalysisResult.route_id.like(f"{route_id}%"),
            AnalysisResult.timestamp >= cutoff_date
        )
    ).group_by(AnalysisResult.month).all()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    data = []
    for r in results:
        if r.month:
            data.append({
                "month": month_names[r.month - 1],
                "month_index": r.month,
                "avg_travel_time": r.avg_travel_time / 60,
                "avg_delay": r.avg_delay / 60,
                "count": r.count
            })
    
    return {"data": data}


def calculate_route_reliability(db: Session, route_id: str, days: int = 30) -> Dict:
    """Calculate route reliability score (0-100)."""
    cutoff_date = datetime.now(UTC) - timedelta(days=days)
    
    results = db.query(AnalysisResult).filter(
        and_(
            AnalysisResult.route_id.like(f"{route_id}%"),
            AnalysisResult.timestamp >= cutoff_date
        )
    ).all()
    
    if not results or len(results) < 5:
        return {"reliability_score": None, "consistency": None, "data_points": len(results)}
    
    travel_times = [r.travel_time_s / 60 for r in results]
    avg_time = np.mean(travel_times)
    std_time = np.std(travel_times)
    
    # Reliability based on coefficient of variation (lower is better)
    cv = (std_time / avg_time) if avg_time > 0 else 1.0
    reliability_score = max(0, min(100, (1 - cv) * 100))
    
    # Consistency (percentage within 20% of average)
    within_range = sum(1 for t in travel_times if abs(t - avg_time) / avg_time <= 0.2)
    consistency = (within_range / len(travel_times)) * 100
    
    return {
        "reliability_score": round(reliability_score, 2),
        "consistency": round(consistency, 2),
        "avg_travel_time": round(avg_time, 2),
        "std_travel_time": round(std_time, 2),
        "min_time": round(min(travel_times), 2),
        "max_time": round(max(travel_times), 2),
        "data_points": len(results)
    }


def predict_future_congestion(db: Session, route_id: str, hours_ahead: int = 24) -> Dict:
    """Predict future congestion using historical patterns."""
    # Get historical data for the same hour of day
    current_hour = datetime.now(UTC).hour
    target_hour = (current_hour + hours_ahead) % 24
    
    results = db.query(AnalysisResult).filter(
        AnalysisResult.route_id.like(f"{route_id}%"),
        AnalysisResult.hour_of_day == target_hour
    ).order_by(AnalysisResult.timestamp.desc()).limit(50).all()
    
    if not results:
        return {"predicted_congestion": None, "confidence": None}
    
    congestion_ratios = [r.travel_time_s / r.no_traffic_s for r in results if r.no_traffic_s and r.no_traffic_s > 0]
    
    if not congestion_ratios:
        return {"predicted_congestion": None, "confidence": None}
    
    predicted = np.mean(congestion_ratios)
    std = np.std(congestion_ratios)
    confidence = max(0, min(100, (1 - (std / predicted)) * 100)) if predicted > 0 else 0
    
    return {
        "predicted_congestion": round(predicted, 2),
        "confidence": round(confidence, 2),
        "predicted_travel_time": round(predicted * (results[0].no_traffic_s / 60), 2) if results[0].no_traffic_s else None,
        "data_points": len(congestion_ratios)
    }


def get_traffic_hotspots(db: Session, days: int = 7) -> List[Dict]:
    """Identify traffic hotspots (routes with highest congestion)."""
    cutoff_date = datetime.now(UTC) - timedelta(days=days)
    
    results = db.query(
        AnalysisResult.route_id,
        func.avg(AnalysisResult.delay_s).label('avg_delay'),
        func.avg(
            AnalysisResult.travel_time_s / AnalysisResult.no_traffic_s
        ).label('avg_congestion'),
        func.count(AnalysisResult.id).label('count')
    ).filter(
        AnalysisResult.timestamp >= cutoff_date,
        AnalysisResult.no_traffic_s > 0
    ).group_by(AnalysisResult.route_id).having(
        func.count(AnalysisResult.id) >= 5
    ).order_by(
        func.avg(AnalysisResult.delay_s).desc()
    ).limit(10).all()
    
    hotspots = []
    for r in results:
        hotspots.append({
            "route_id": r.route_id,
            "avg_delay_minutes": round(r.avg_delay / 60, 2),
            "avg_congestion": round(r.avg_congestion, 2),
            "analysis_count": r.count
        })
    
    return hotspots
