import pytest
from calculator.api import app
from fastapi.testclient import TestClient


@pytest.mark.api
class TestPlanetsAPI:
    def setup_method(self):
        """Setup method that runs before each test."""
        self.client = TestClient(app)

    def test_add_endpoint(self):
        """Test the /add endpoint."""
        response = self.client.get("/add/2/3")
        assert response.status_code == 200
        assert response.json() == {"result": 5}

    def test_divide_endpoint(self):
        """Test the /divide endpoint."""
        response = self.client.get("/divide/6/2")
        assert response.status_code == 200
        assert response.json() == {"result": 3}

    def test_divide_by_zero_endpoint(self):
        """Test division by zero error handling."""
        response = self.client.get("/divide/5/0")
        assert response.status_code == 400
        assert "Cannot divide by zero" in response.json()["detail"]
