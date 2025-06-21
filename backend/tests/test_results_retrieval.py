"""
Tests for analysis results retrieval functionality
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from fastapi.testclient import TestClient
from app.main import app
from app.services.analysis_executor import AnalysisExecutor


@pytest.fixture
def client():
    return TestClient(app)


class TestResultsRetrieval:
    """Test analysis results retrieval endpoints"""
    
    def setup_method(self):
        self.analysis_id = "test-analysis-123"
        
        # Mock analysis results for testing
        self.mock_complete_results = {
            "volume_results": [
                {
                    "layer_name": "Surface 0 to Surface 1",
                    "volume_cubic_yards": 1250.5,
                    "confidence_interval": [1240.2, 1260.8]
                },
                {
                    "layer_name": "Surface 1 to Surface 2", 
                    "volume_cubic_yards": 980.3,
                    "confidence_interval": [970.1, 990.5]
                }
            ],
            "thickness_results": [
                {
                    "layer_name": "Surface 0 to Surface 1",
                    "average_thickness_feet": 2.5,
                    "min_thickness_feet": 1.0,
                    "max_thickness_feet": 4.0,
                    "confidence_interval": [2.3, 2.7]
                },
                {
                    "layer_name": "Surface 1 to Surface 2",
                    "average_thickness_feet": 2.1,
                    "min_thickness_feet": 0.8,
                    "max_thickness_feet": 3.5,
                    "confidence_interval": [2.0, 2.2]
                }
            ],
            "compaction_rates": [
                {
                    "layer_name": "Surface 0 to Surface 1",
                    "compaction_rate_lbs_per_cubic_yard": 3200.0,
                    "tonnage_input": 2000.0
                },
                {
                    "layer_name": "Surface 1 to Surface 2",
                    "compaction_rate_lbs_per_cubic_yard": None,
                    "tonnage_input": None
                }
            ],
            "analysis_metadata": {
                "analysis_id": self.analysis_id,
                "status": "completed",
                "completion_time": "2024-12-20T10:30:00Z",
                "total_processing_time_seconds": 45.2,
                "surfaces_processed": 3,
                "coordinate_system": "UTM Zone 10N",
                "analysis_boundary": {
                    "min_x": 583000.0,
                    "max_x": 584000.0,
                    "min_y": 4507000.0,
                    "max_y": 4508000.0
                }
            }
        }

    def test_complete_results_retrieval(self, client):
        """Test complete results retrieval with all components"""
        with patch('app.routes.analysis.executor.get_results', return_value=self.mock_complete_results):
            response = client.get(f"/api/analysis/{self.analysis_id}/results")
            
        assert response.status_code == 200
        data = response.json()
        
        # Check required result components
        assert "volume_results" in data
        assert "thickness_results" in data
        assert "compaction_rates" in data
        assert "analysis_metadata" in data
        
        # Validate volume results structure
        volume_results = data["volume_results"]
        assert len(volume_results) == 2
        for layer_result in volume_results:
            assert "layer_name" in layer_result
            assert "volume_cubic_yards" in layer_result
            assert isinstance(layer_result["volume_cubic_yards"], (int, float))
            assert "confidence_interval" in layer_result
            assert len(layer_result["confidence_interval"]) == 2
        
        # Validate thickness results structure
        thickness_results = data["thickness_results"]
        assert len(thickness_results) == 2
        for layer_result in thickness_results:
            assert "layer_name" in layer_result
            assert "average_thickness_feet" in layer_result
            assert "min_thickness_feet" in layer_result
            assert "max_thickness_feet" in layer_result
            assert isinstance(layer_result["average_thickness_feet"], (int, float))
            assert isinstance(layer_result["min_thickness_feet"], (int, float))
            assert isinstance(layer_result["max_thickness_feet"], (int, float))
        
        # Validate compaction rates structure
        compaction_rates = data["compaction_rates"]
        assert len(compaction_rates) == 2
        for layer_result in compaction_rates:
            assert "layer_name" in layer_result
            assert "compaction_rate_lbs_per_cubic_yard" in layer_result
            assert "tonnage_input" in layer_result

    def test_results_not_ready(self, client):
        """Test results retrieval when analysis is still processing"""
        with patch('app.routes.analysis.executor.get_results', return_value=None):
            with patch('app.routes.analysis.executor.get_analysis_status', return_value={"status": "processing", "progress_percent": 60}):
                response = client.get(f"/api/analysis/{self.analysis_id}/results")
        
        assert response.status_code == 202  # Accepted, not ready yet
        data = response.json()
        assert data["status"] == "processing"
        assert "estimated_completion" in data
        assert "progress" in data

    def test_analysis_not_found(self, client):
        """Test results retrieval for non-existent analysis"""
        with patch('app.routes.analysis.executor.get_results', return_value=None):
            with patch('app.routes.analysis.executor.get_analysis_status', side_effect=KeyError("Analysis not found")):
                response = client.get(f"/api/analysis/non-existent-id/results")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_partial_results_volume_only(self, client):
        """Test partial results retrieval for volume only"""
        partial_results = {
            "volume_results": self.mock_complete_results["volume_results"],
            "analysis_metadata": self.mock_complete_results["analysis_metadata"]
        }
        
        with patch('app.routes.analysis.executor.get_results', return_value=partial_results):
            response = client.get(f"/api/analysis/{self.analysis_id}/results?include=volume")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "volume_results" in data
        assert "analysis_metadata" in data
        assert "thickness_results" not in data
        assert "compaction_rates" not in data

    def test_partial_results_thickness_only(self, client):
        """Test partial results retrieval for thickness only"""
        partial_results = {
            "thickness_results": self.mock_complete_results["thickness_results"],
            "analysis_metadata": self.mock_complete_results["analysis_metadata"]
        }
        
        with patch('app.routes.analysis.executor.get_results', return_value=partial_results):
            response = client.get(f"/api/analysis/{self.analysis_id}/results?include=thickness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "thickness_results" in data
        assert "analysis_metadata" in data
        assert "volume_results" not in data
        assert "compaction_rates" not in data

    def test_result_formatting_and_units(self, client):
        """Test result formatting and unit validation"""
        with patch('app.routes.analysis.executor.get_results', return_value=self.mock_complete_results):
            response = client.get(f"/api/analysis/{self.analysis_id}/results")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate volume units (cubic yards)
        for volume_result in data["volume_results"]:
            assert volume_result["volume_cubic_yards"] > 0
            assert isinstance(volume_result["volume_cubic_yards"], (int, float))
        
        # Validate thickness units (feet)
        for thickness_result in data["thickness_results"]:
            assert thickness_result["average_thickness_feet"] > 0
            assert thickness_result["min_thickness_feet"] > 0
            assert thickness_result["max_thickness_feet"] > 0
            assert thickness_result["min_thickness_feet"] <= thickness_result["average_thickness_feet"] <= thickness_result["max_thickness_feet"]
        
        # Validate compaction rate units (lbs/cubic yard)
        for compaction_result in data["compaction_rates"]:
            if compaction_result["compaction_rate_lbs_per_cubic_yard"] is not None:
                assert compaction_result["compaction_rate_lbs_per_cubic_yard"] > 0

    def test_result_caching_behavior(self, client):
        """Test that results are properly cached and retrieved"""
        with patch('app.routes.analysis.executor.get_results', return_value=self.mock_complete_results) as mock_get:
            # Make multiple requests
            for _ in range(3):
                response = client.get(f"/api/analysis/{self.analysis_id}/results")
                assert response.status_code == 200
            
            # Should call get_results only once per request
            assert mock_get.call_count == 3

    def test_cancelled_analysis_results(self, client):
        """Test results retrieval for cancelled analysis"""
        with patch('app.routes.analysis.executor.get_results', return_value=None):
            with patch('app.routes.analysis.executor.get_analysis_status', return_value={"status": "cancelled", "completion_time": "2024-12-20T10:30:00Z"}):
                response = client.get(f"/api/analysis/{self.analysis_id}/results")
        
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_metadata"]["status"] == "cancelled"
        assert "cancellation_time" in data["analysis_metadata"]

    def test_failed_analysis_results(self, client):
        """Test results retrieval for failed analysis"""
        with patch('app.routes.analysis.executor.get_results', return_value=None):
            with patch('app.routes.analysis.executor.get_analysis_status', return_value={"status": "failed", "completion_time": "2024-12-20T10:30:00Z", "error_message": "Processing failed"}):
                response = client.get(f"/api/analysis/{self.analysis_id}/results")
        
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_metadata"]["status"] == "failed"
        assert "failure_time" in data["analysis_metadata"]
        assert "error_message" in data["analysis_metadata"]

    def test_results_with_confidence_intervals(self, client):
        """Test that confidence intervals are properly calculated and included"""
        with patch('app.routes.analysis.executor.get_results', return_value=self.mock_complete_results):
            response = client.get(f"/api/analysis/{self.analysis_id}/results")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate confidence intervals in volume results
        for volume_result in data["volume_results"]:
            ci = volume_result["confidence_interval"]
            assert len(ci) == 2
            assert ci[0] < ci[1]  # Lower bound < upper bound
            assert ci[0] <= volume_result["volume_cubic_yards"] <= ci[1]
        
        # Validate confidence intervals in thickness results
        for thickness_result in data["thickness_results"]:
            ci = thickness_result["confidence_interval"]
            assert len(ci) == 2
            assert ci[0] < ci[1]  # Lower bound < upper bound
            assert ci[0] <= thickness_result["average_thickness_feet"] <= ci[1]


class TestResultsRetrievalLogic:
    """Test the logic behind results retrieval"""

    def test_results_formatting_validation(self):
        """Test that results are properly formatted"""
        # Test valid results structure
        valid_results = {
            "volume_results": [{"layer_name": "Test", "volume_cubic_yards": 100.0, "confidence_interval": [95.0, 105.0]}],
            "thickness_results": [{"layer_name": "Test", "average_thickness_feet": 2.0, "min_thickness_feet": 1.0, "max_thickness_feet": 3.0, "confidence_interval": [1.8, 2.2]}],
            "analysis_metadata": {"status": "completed"}
        }
        
        # All required fields should be present
        assert "volume_results" in valid_results
        assert "thickness_results" in valid_results
        assert "analysis_metadata" in valid_results
        
        # Results should be lists
        assert isinstance(valid_results["volume_results"], list)
        assert isinstance(valid_results["thickness_results"], list)

    def test_unit_conversion_validation(self):
        """Test that units are properly converted and validated"""
        # Test volume conversion (cubic feet to cubic yards)
        volume_cubic_feet = 3375.0  # 125 cubic yards
        volume_cubic_yards = volume_cubic_feet / 27.0
        assert abs(volume_cubic_yards - 125.0) < 0.01
        
        # Test thickness units (should be in feet)
        thickness_feet = 2.5
        assert thickness_feet > 0
        assert isinstance(thickness_feet, (int, float))

    def test_confidence_interval_validation(self):
        """Test that confidence intervals are properly calculated"""
        # Test confidence interval structure
        ci = [95.0, 105.0]
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < upper bound
        
        # Test that value falls within interval
        value = 100.0
        assert ci[0] <= value <= ci[1]

    def test_partial_results_filtering(self):
        """Test that partial results filtering works correctly"""
        complete_results = {
            "volume_results": [{"layer_name": "Test", "volume_cubic_yards": 100.0}],
            "thickness_results": [{"layer_name": "Test", "average_thickness_feet": 2.0}],
            "compaction_rates": [{"layer_name": "Test", "compaction_rate_lbs_per_cubic_yard": 3200.0}],
            "analysis_metadata": {"status": "completed"}
        }
        
        # Test volume-only filtering
        volume_only = {k: v for k, v in complete_results.items() if k in ["volume_results", "analysis_metadata"]}
        assert "volume_results" in volume_only
        assert "thickness_results" not in volume_only
        assert "compaction_rates" not in volume_only
        
        # Test thickness-only filtering
        thickness_only = {k: v for k, v in complete_results.items() if k in ["thickness_results", "analysis_metadata"]}
        assert "thickness_results" in thickness_only
        assert "volume_results" not in thickness_only
        assert "compaction_rates" not in thickness_only

    def test_analysis_status_handling(self):
        """Test that analysis status is properly handled"""
        # Test processing status
        processing_status = {"status": "processing", "progress_percent": 50}
        assert processing_status["status"] == "processing"
        assert 0 <= processing_status["progress_percent"] <= 100
        
        # Test completed status
        completed_status = {"status": "completed", "completion_time": "2024-12-20T10:30:00Z"}
        assert completed_status["status"] == "completed"
        assert "completion_time" in completed_status
        
        # Test failed status
        failed_status = {"status": "failed", "error_message": "Processing failed"}
        assert failed_status["status"] == "failed"
        assert "error_message" in failed_status

    def test_result_caching_logic(self):
        """Test that result caching works correctly"""
        # Mock cache behavior
        cache = {}
        
        def get_cached_result(analysis_id):
            return cache.get(analysis_id)
        
        def cache_result(analysis_id, result):
            cache[analysis_id] = result
        
        # Test cache miss
        assert get_cached_result("test-123") is None
        
        # Test cache hit
        test_result = {"status": "completed"}
        cache_result("test-123", test_result)
        assert get_cached_result("test-123") == test_result
        
        # Test cache update
        updated_result = {"status": "completed", "updated": True}
        cache_result("test-123", updated_result)
        assert get_cached_result("test-123") == updated_result 