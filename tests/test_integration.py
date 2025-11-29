import pytest
import httpx
import os
import time

# 1. Configuration
# We allow the URL to be configured via environment variable.
# If running locally with Docker mapping port 80 to 80, this default works.
BASE_URL = os.getenv("TEST_URL", "http://localhost:80")

@pytest.mark.integration
class TestDockerIntegration:
    """
    Integration tests for the running Docker container.
    These tests require the service to be actually running at BASE_URL.
    """

    @pytest.fixture(scope="class")
    def client(self):
        """
        Creates an HTTP client that persists for all tests in this class.
        It also performs a 'wait-for-it' check to ensure the Docker service is ready.
        """
        timeout_seconds = 10
        start_time = time.time()
        
        # Simple retry loop to wait for the container to spin up
        while True:
            try:
                response = httpx.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except httpx.RequestError:
                pass
            
            if time.time() - start_time > timeout_seconds:
                pytest.fail(f"Timeout: Service at {BASE_URL} is not reachable.")
            
            time.sleep(1)

        # Return the client configured with the base URL
        with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
            yield client

    def test_full_workflow(self, client):
        """
        Test the complete lifecycle:
        1. Generate words
        2. Star a specific word
        3. Verify persistence
        4. Unstar the word
        5. Verify removal
        """
        # Step 1: Generate words
        # We request 5 words to ensure we have candidates
        gen_response = client.get("/generate", params={"num_words": 5})
        assert gen_response.status_code == 200
        words = gen_response.json()
        assert len(words) == 5
        assert isinstance(words[0], str)

        target_word = words[0]
        
        # Step 2: Star a word
        star_response = client.post("/starred", json={"word": target_word})
        assert star_response.status_code == 200
        # The API returns the updated list
        assert target_word in star_response.json()

        # Step 3: Verify Persistence (Get the list separately)
        get_starred = client.get("/starred")
        assert get_starred.status_code == 200
        assert target_word in get_starred.json()

        # Step 4: Unstar the word
        unstar_response = client.post("/unstarred", json={"word": target_word})
        assert unstar_response.status_code == 200
        
        # Step 5: Verify Removal
        final_check = client.get("/starred")
        assert target_word not in final_check.json()

    def test_error_recovery_invalid_input(self, client):
        """
        Test that the running system handles invalid input gracefully 
        without crashing the container.
        """
        # Case 1: Invalid number of words (0)
        response = client.get("/generate", params={"num_words": 0})
        assert response.status_code == 422
        
        # Case 2: Invalid temperature (too high)
        response = client.get("/generate", params={"temperature": 100.0})
        assert response.status_code == 422

        # Case 3: Verify system is still healthy after errors
        health = client.get("/health")
        assert health.status_code == 200