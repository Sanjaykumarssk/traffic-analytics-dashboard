"""
Notification and alert system.
"""

import os
from datetime import datetime, UTC, timedelta
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import and_
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from db import Notification, SavedRoute, AnalysisResult, User
from analytics import get_peak_hours_analysis, predict_future_congestion
from logging_config import get_logger

logger = get_logger(__name__)


# Email configuration (set in .env)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@trafficdashboard.com")


def create_notification(
    db: Session,
    user_id: int,
    notification_type: str,
    title: str,
    message: str,
    route_id: Optional[str] = None
) -> Notification:
    """Create a notification for a user."""
    notification = Notification(
        user_id=user_id,
        type=notification_type,
        title=title,
        message=message,
        route_id=route_id,
        is_read=False
    )
    db.add(notification)
    db.commit()
    db.refresh(notification)
    return notification


def send_email_notification(user_email: str, subject: str, body: str) -> bool:
    """Send email notification."""
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.debug(f"Email not configured. Would send to {user_email}: {subject}")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = user_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email sent successfully to {user_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {user_email}: {e}")
        return False


def check_traffic_alerts(db: Session, user_id: int) -> List[Notification]:
    """Check for traffic alerts on saved routes."""
    saved_routes = db.query(SavedRoute).filter(
        SavedRoute.user_id == user_id,
        SavedRoute.is_favorite == True
    ).all()
    
    notifications = []
    for route in saved_routes:
        # Get latest analysis
        latest = db.query(AnalysisResult).filter(
            AnalysisResult.route_id.like(f"{route.route_name}%")
        ).order_by(AnalysisResult.timestamp.desc()).first()
        
        if latest:
            # Check if delay is high
            delay_minutes = latest.delay_s / 60 if latest.delay_s else 0
            if delay_minutes > 15:  # Alert if delay > 15 minutes
                notification = create_notification(
                    db, user_id, 'traffic_alert',
                    f"High Traffic Alert: {route.route_name}",
                    f"Your saved route has {delay_minutes:.1f} minutes of delay. Consider alternative routes.",
                    route.route_name
                )
                notifications.append(notification)
                
                # Send email if user has email
                user = db.query(User).filter(User.id == user_id).first()
                if user and user.email:
                    send_email_notification(
                        user.email,
                        f"Traffic Alert: {route.route_name}",
                        f"<h2>Traffic Alert</h2><p>Your saved route <b>{route.route_name}</b> currently has <b>{delay_minutes:.1f} minutes</b> of delay.</p>"
                    )
    
    return notifications


def suggest_best_time_to_leave(db: Session, user_id: int, route_id: str) -> Optional[Notification]:
    """Suggest best time to leave based on historical data."""
    peak_analysis = get_peak_hours_analysis(db, route_id)
    
    if peak_analysis.get('best_hour') is not None:
        best_hour = peak_analysis['best_hour']
        current_hour = datetime.now(UTC).hour
        
        if best_hour != current_hour:
            hours_until_best = (best_hour - current_hour) % 24
            if hours_until_best > 0 and hours_until_best <= 6:
                notification = create_notification(
                    db, user_id, 'best_time',
                    f"Best Time to Leave: {best_hour}:00",
                    f"Based on historical data, the best time to travel this route is at {best_hour}:00 ({hours_until_best} hours from now).",
                    route_id
                )
                return notification
    
    return None


def check_congestion_warnings(db: Session, user_id: int, route_id: str) -> Optional[Notification]:
    """Check for congestion warnings."""
    prediction = predict_future_congestion(db, route_id, hours_ahead=1)
    
    if prediction.get('predicted_congestion') and prediction['predicted_congestion'] > 1.5:
        notification = create_notification(
            db, user_id, 'congestion_warning',
            "High Congestion Predicted",
            f"Route is predicted to have {prediction['predicted_congestion']:.2f}x congestion in the next hour. Consider leaving earlier or taking an alternative route.",
            route_id
        )
        return notification
    
    return None


def get_user_notifications(db: Session, user_id: int, unread_only: bool = False, limit: int = 50) -> List[Notification]:
    """Get notifications for a user."""
    query = db.query(Notification).filter(Notification.user_id == user_id)
    
    if unread_only:
        query = query.filter(Notification.is_read == False)
    
    return query.order_by(Notification.created_at.desc()).limit(limit).all()


def mark_notification_read(db: Session, notification_id: int, user_id: int) -> bool:
    """Mark a notification as read."""
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == user_id
    ).first()
    
    if notification:
        notification.is_read = True
        db.commit()
        return True
    return False
