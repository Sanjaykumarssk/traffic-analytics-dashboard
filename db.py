"""
Database persistence layer using SQLAlchemy with PostgreSQL or SQLite fallback.
"""

import os
import json
from datetime import datetime, UTC
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from logging_config import get_logger

logger = get_logger(__name__)

Base = declarative_base()

# Database configuration - supports PostgreSQL or SQLite
DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()
DB_PATH = os.getenv("DB_PATH", "traffic_analysis.db")

# PostgreSQL configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "traffic_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "traffic_dashboard")


class User(Base):
    """User model for authentication and user management."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    saved_routes = relationship("SavedRoute", back_populates="user", cascade="all, delete-orphan")
    route_ratings = relationship("RouteRating", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")


class SavedRoute(Base):
    """Saved/favorite routes for users."""
    __tablename__ = "saved_routes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    route_name = Column(String, nullable=False)
    origin = Column(Text, nullable=False)  # JSON string
    destination = Column(Text, nullable=False)  # JSON string
    route_preferences = Column(JSON, nullable=True)  # Store route options
    is_favorite = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    last_used = Column(DateTime, default=lambda: datetime.now(UTC))
    usage_count = Column(Integer, default=0)
    share_token = Column(String, unique=True, index=True, nullable=True)  # For sharing
    
    # Relationships
    user = relationship("User", back_populates="saved_routes")


class RouteRating(Base):
    """Route ratings and reviews from users."""
    __tablename__ = "route_ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # Made nullable for anonymous ratings
    route_id = Column(String, nullable=False, index=True)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    review = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    
    # Relationships
    user = relationship("User", back_populates="route_ratings")


class Notification(Base):
    """User notifications and alerts."""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(String, nullable=False)  # 'traffic_alert', 'best_time', 'congestion_warning'
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    route_id = Column(String, nullable=True)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    
    # Relationships
    user = relationship("User", back_populates="notifications")


class AnalysisResult(Base):
    """SQLAlchemy model for analysis results table."""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), index=True)
    route_id = Column(String, index=True)
    origin = Column(String)  # JSON string or text
    destination = Column(String)  # JSON string or text
    travel_time_s = Column(Float)
    no_traffic_s = Column(Float)
    delay_s = Column(Float)
    length_m = Column(Float)
    calculated_cost = Column(Float)
    ml_predicted = Column(Float, nullable=True)
    raw_json = Column(Text, nullable=True)  # Store full route JSON if needed
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # Optional user association
    hour_of_day = Column(Integer, nullable=True, index=True)  # For analytics
    day_of_week = Column(Integer, nullable=True, index=True)  # 0=Monday, 6=Sunday
    month = Column(Integer, nullable=True, index=True)  # For seasonal analysis


def get_engine():
    """
    Create and return SQLAlchemy engine.
    Supports PostgreSQL (production) or SQLite (development).
    """
    if DB_TYPE == "postgresql" or DB_TYPE == "postgres":
        # PostgreSQL connection string
        if not POSTGRES_PASSWORD:
            logger.warning("PostgreSQL password not set, falling back to SQLite")
            return _get_sqlite_engine()
        
        database_url = (
            f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
            f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )
        try:
            engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                echo=False  # Set to True for SQL query logging
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info(f"✅ Connected to PostgreSQL database: {POSTGRES_DB}")
            return engine
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            logger.info("Falling back to SQLite")
            return _get_sqlite_engine()
    else:
        # SQLite (default for development)
        return _get_sqlite_engine()


def _get_sqlite_engine():
    """Create SQLite engine."""
    logger.info(f"Using SQLite database: {DB_PATH}")
    return create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        echo=False
    )


def get_session() -> Session:
    """Create and return a database session."""
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def init_db():
    """Initialize database tables."""
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables initialized successfully")
        
        # Ensure default admin user exists
        try:
            from auth import ensure_admin_user_exists
            session = get_session()
            ensure_admin_user_exists(session)
            session.close()
            logger.info("✅ Default admin user verified/created")
        except Exception as e:
            logger.warning(f"⚠️ Could not create default admin user: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


def save_analysis(session: Session, analysis_object: dict) -> AnalysisResult:
    """
    Save an analysis result to the database.
    
    Args:
        session: SQLAlchemy session
        analysis_object: Dictionary with analysis data containing:
            - route_id: str
            - origin: str or dict
            - destination: str or dict
            - travel_time_s: float
            - no_traffic_s: float
            - delay_s: float
            - length_m: float
            - calculated_cost: float
            - ml_predicted: float (optional)
            - raw_json: str (optional)
            
    Returns:
        Saved AnalysisResult object
    """
    import json
    
    # Convert origin/destination to string if dict
    origin_str = json.dumps(analysis_object.get("origin")) if isinstance(
        analysis_object.get("origin"), dict
    ) else str(analysis_object.get("origin", ""))
    
    dest_str = json.dumps(analysis_object.get("destination")) if isinstance(
        analysis_object.get("destination"), dict
    ) else str(analysis_object.get("destination", ""))
    
    now = datetime.now(UTC)
    result = AnalysisResult(
        timestamp=now,
        route_id=analysis_object.get("route_id", ""),
        origin=origin_str,
        destination=dest_str,
        travel_time_s=analysis_object.get("travel_time_s"),
        no_traffic_s=analysis_object.get("no_traffic_s"),
        delay_s=analysis_object.get("delay_s", 0),
        length_m=analysis_object.get("length_m", 0),
        calculated_cost=analysis_object.get("calculated_cost"),
        ml_predicted=analysis_object.get("ml_predicted"),
        raw_json=json.dumps(analysis_object.get("raw_json")) if analysis_object.get("raw_json") else None,
        user_id=analysis_object.get("user_id"),
        hour_of_day=now.hour,
        day_of_week=now.weekday(),
        month=now.month
    )
    
    session.add(result)
    session.commit()
    session.refresh(result)
    return result

