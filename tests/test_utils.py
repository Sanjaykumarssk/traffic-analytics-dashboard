"""
Unit tests for utility functions.
"""

import pytest
from utils import (
    haversine_m,
    compute_route_cost,
    summarize_route
)


class TestHaversine:
    """Test haversine distance calculation."""
    
    def test_haversine_same_point(self):
        """Test distance between same point is zero."""
        distance = haversine_m(13.0827, 80.2707, 13.0827, 80.2707)
        assert distance == 0
    
    def test_haversine_known_distance(self):
        """Test distance calculation for known coordinates."""
        # Chennai to Bangalore approximate distance: ~350 km
        distance = haversine_m(13.0827, 80.2707, 12.9716, 77.5946)
        assert 340000 < distance < 360000  # Within 20km tolerance
    
    def test_haversine_negative_coordinates(self):
        """Test with negative coordinates."""
        distance = haversine_m(-13.0827, -80.2707, -12.9716, -77.5946)
        assert distance > 0


class TestCostCalculation:
    """Test route cost calculation."""
    
    def test_cost_calculation_basic(self):
        """Test basic cost calculation."""
        cost = compute_route_cost(
            travel_time_s=1200,  # 20 minutes
            no_traffic_s=900,     # 15 minutes
            delay_s=300,          # 5 minutes
            length_m=8500,        # 8.5 km
            alpha=1.0,
            beta=0.5,
            gamma=0.001
        )
        # Expected: 1.0 * 20 + 0.5 * 5 + 0.001 * 8.5 = 22.5085
        assert abs(cost - 22.5085) < 0.01
    
    def test_cost_calculation_zero_delay(self):
        """Test cost with no delay."""
        cost = compute_route_cost(
            travel_time_s=600,
            no_traffic_s=600,
            delay_s=0,
            length_m=5000,
            alpha=1.0,
            beta=0.5,
            gamma=0.001
        )
        # Expected: 1.0 * 10 + 0.5 * 0 + 0.001 * 5 = 10.005
        assert abs(cost - 10.005) < 0.01
    
    def test_cost_calculation_custom_weights(self):
        """Test cost with custom weights."""
        cost = compute_route_cost(
            travel_time_s=1200,
            no_traffic_s=900,
            delay_s=300,
            length_m=8500,
            alpha=2.0,  # Higher weight on time
            beta=1.0,   # Higher weight on delay
            gamma=0.002  # Higher weight on distance
        )
        # Should be higher than default weights
        assert cost > 22.5


class TestRouteSummarization:
    """Test route summarization."""
    
    def test_summarize_route_mock_data(self):
        """Test route summarization with mock data."""
        mock_route = {
            "summary": {
                "lengthInMeters": 8500,
                "travelTimeInSeconds": 1200,
                "noTrafficTravelTimeInSeconds": 900
            }
        }
        summary = summarize_route(mock_route)
        
        assert summary["length_m"] == 8500
        assert summary["travel_time_s"] == 1200
        assert summary["no_traffic_s"] == 900
        assert summary["delay_s"] == 300  # 1200 - 900


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

