import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import the module explicitly to access global variables
import backend.app 
from backend.app import app

@pytest.mark.api
class TestPlanetsAPI:
    def setup_method(self):
        """
        Runs before every test.
        """
        # 1. Initialize Client
        self.client = TestClient(app)
        
        # 2. Reset the in-memory 'database'
        backend.app.starred_words = []

        # 3. Simulate a loaded model state.
        # Since we are not running the full 'lifespan' startup in these unit tests
        # (or we want to bypass file checks), we manually inject the mocks 
        # that the /generate endpoint expects to find.
        backend.app.model_state["model"] = MagicMock()
        backend.app.model_state["tokenizer"] = MagicMock()

    def teardown_method(self):
        """Clean up after tests."""
        backend.app.model_state.clear()
        backend.app.starred_words = []

    # -------------------------------------------------------------------------
    # 1. Test the /health endpoint
    # -------------------------------------------------------------------------
    def test_health_endpoint(self):
        """Test that the health endpoint returns status 200 and 'ok'."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # -------------------------------------------------------------------------
    # 2. Test the /generate endpoint
    # -------------------------------------------------------------------------
    def test_generate_default_parameters(self):
        """Test /generate with no parameters (should use defaults: n=10, temp=1.0)."""
        # We patch 'sample_n' to control the output and avoid running ML logic
        with patch("backend.app.sample_n") as mock_sample:
            # Mock return value (must be list of strings)
            mock_sample.return_value = ["word" for _ in range(10)]
            
            response = self.client.get("/generate")
            
            assert response.status_code == 200
            assert len(response.json()) == 10
            
            # Verify default arguments were passed to the internal function
            mock_sample.assert_called_once()
            args, kwargs = mock_sample.call_args
            assert kwargs["n"] == 10
            assert kwargs["temperature"] == 1.0

    def test_generate_custom_parameters(self):
        """Test /generate with custom number of words and temperature."""
        with patch("backend.app.sample_n") as mock_sample:
            mock_sample.return_value = ["word1", "word2", "word3", "word4", "word5"]
            
            # Request 5 words with temperature 0.5
            response = self.client.get("/generate?num_words=5&temperature=0.5")
            
            assert response.status_code == 200
            assert len(response.json()) == 5
            
            # Verify our custom arguments were passed correctly
            mock_sample.assert_called_once()
            args, kwargs = mock_sample.call_args
            assert kwargs["n"] == 5
            assert kwargs["temperature"] == 0.5

    def test_generate_validation_limits(self):
        """Test edge cases for validation (min words, max temperature)."""
        # Test 0 words (should fail, min is 1)
        response = self.client.get("/generate?num_words=0")
        assert response.status_code == 422
        
        # Test temperature too high (should fail, max is 10)
        response = self.client.get("/generate?temperature=11.0")
        assert response.status_code == 422

    def test_generate_fails_if_model_missing(self):
        """Test that 503 is returned if model state is empty."""
        # Clear the state we set in setup_method
        backend.app.model_state.clear()
        
        response = self.client.get("/generate")
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    # -------------------------------------------------------------------------
    # 3. Test the starred words functionality
    # -------------------------------------------------------------------------
    def test_starred_words_workflow(self):
        """
        Comprehensive test for the Starred Words workflow.
        """
        word_payload = {"word": "nebula"}
        
        # --- A. Adding a word ---
        response = self.client.post("/starred", json=word_payload)
        assert response.status_code == 200
        assert "nebula" in response.json()
        
        # --- B. Adding a duplicate word (should not duplicate) ---
        response = self.client.post("/starred", json=word_payload)
        assert response.status_code == 200
        assert response.json().count("nebula") == 1
        
        # --- C. Getting the starred words list ---
        response = self.client.get("/starred")
        assert response.status_code == 200
        assert response.json() == ["nebula"]
        
        # --- D. Removing a word ---
        response = self.client.post("/unstarred", json=word_payload)
        assert response.status_code == 200
        assert "nebula" not in response.json()