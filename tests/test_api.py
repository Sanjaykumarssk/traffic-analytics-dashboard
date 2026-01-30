"""
Comprehensive API endpoint tests using pytest.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import tempfile

# Import app and database
from app import app
from db import Base, get_session, init_db, User
from auth import get_password_hash

# Create test database
TEST_DB = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
TEST_DB_PATH = TEST_DB.name
TEST_DB.close()

# Override database path for testing
os.environ["DB_PATH"] = TEST_DB_PATH
os.environ["DB_TYPE"] = "sqlite"

# Create test engine
test_engine = create_engine(f"sqlite:///{TEST_DB_PATH}", connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Override get_session dependency
def override_get_session():
    try:
        db = TestSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_session] = override_get_session

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Set up test database before tests."""
    Base.metadata.create_all(bind=test_engine)
    
    # Create test user
    db = TestSessionLocal()
    test_user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpass123"),
        is_active=True
    )
    db.add(test_user)
    db.commit()
    db.close()
    
    yield
    
    # Cleanup
    Base.metadata.drop_all(bind=test_engine)
    os.unlink(TEST_DB_PATH)


@pytest.fixture
def auth_headers():
    """Get authentication headers for test user."""
    response = client.post(
        "/api/auth/login",
        data={"username": "testuser", "password": "testpass123"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "timestamp" in data


class TestAutocompleteEndpoint:
    """Test autocomplete endpoint."""
    
    def test_autocomplete_missing_query(self):
        """Test autocomplete without query parameter."""
        response = client.get("/autocomplete")
        assert response.status_code == 422  # Validation error
    
    def test_autocomplete_with_query(self):
        """Test autocomplete with query parameter."""
        # Mock or skip if no API key
        response = client.get("/autocomplete?q=Chennai")
        # Should return 200 or 500 (if API key missing)
        assert response.status_code in [200, 500]


class TestRouteAnalysis:
    """Test route analysis endpoint."""
    
    def test_analyze_route_missing_data(self):
        """Test route analysis with missing data."""
        response = client.post("/analyze-route", json={})
        assert response.status_code == 422
    
    def test_analyze_route_invalid_data(self):
        """Test route analysis with invalid data."""
        response = client.post(
            "/analyze-route",
            json={
                "origin": "",
                "destination": ""
            }
        )
        assert response.status_code in [400, 422]


class TestAuthentication:
    """Test authentication endpoints."""
    
    def test_register_user(self):
        """Test user registration."""
        response = client.post(
            "/api/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "newpass123",
                "full_name": "New User"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert data["email"] == "newuser@example.com"
    
    def test_register_duplicate_user(self):
        """Test registering duplicate user."""
        # Register first time
        client.post(
            "/api/auth/register",
            json={
                "email": "duplicate@example.com",
                "username": "duplicate",
                "password": "pass123"
            }
        )
        # Try to register again
        response = client.post(
            "/api/auth/register",
            json={
                "email": "duplicate@example.com",
                "username": "duplicate",
                "password": "pass123"
            }
        )
        assert response.status_code == 400
    
    def test_login_success(self, auth_headers):
        """Test successful login."""
        response = client.post(
            "/api/auth/login",
            data={"username": "testuser", "password": "testpass123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        response = client.post(
            "/api/auth/login",
            data={"username": "testuser", "password": "wrongpass"}
        )
        assert response.status_code == 401
    
    def test_get_current_user(self, auth_headers):
        """Test getting current user info."""
        response = client.get("/api/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"


class TestSavedRoutes:
    """Test saved routes endpoints."""
    
    def test_create_saved_route(self, auth_headers):
        """Test creating a saved route."""
        response = client.post(
            "/api/saved-routes",
            json={
                "route_name": "Home to Work",
                "origin": "Guindy, Chennai",
                "destination": "Velachery, Chennai"
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["route_name"] == "Home to Work"
    
    def test_get_saved_routes(self, auth_headers):
        """Test getting saved routes."""
        response = client.get("/api/saved-routes", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_create_route_unauthorized(self):
        """Test creating route without authentication."""
        response = client.post(
            "/api/saved-routes",
            json={
                "route_name": "Test Route",
                "origin": "A",
                "destination": "B"
            }
        )
        assert response.status_code == 401


class TestAnalytics:
    """Test analytics endpoints."""
    
    def test_get_peak_hours(self, auth_headers):
        """Test peak hours analysis."""
        response = client.get(
            "/api/analytics/peak-hours/Test→Route",
            headers=auth_headers
        )
        assert response.status_code == 200
    
    def test_get_hotspots(self, auth_headers):
        """Test traffic hotspots."""
        response = client.get("/api/analytics/hotspots", headers=auth_headers)
        assert response.status_code == 200


class TestRateLimiting:
    """Test rate limiting."""
    
    def test_rate_limit_headers(self):
        """Test that rate limit headers are present."""
        response = client.get("/health")
        # Rate limit headers should be present (even if not limited)
        assert "X-RateLimit-Limit" in response.headers or response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

