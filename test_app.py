
#!/usr/bin/env python3
"""
Integration tests for the FastAPI application using mocking
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Import and apply mock settings before importing the app
from test_config import patch_settings_for_testing

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def mock_settings():
    """Set up mock settings for the entire test module."""
    return patch_settings_for_testing()

@pytest.fixture(scope="module")
def client(mock_settings):
    """Create a test client with mocked dependencies."""
    with patch('services.db.Pinecone') as mock_pinecone, \
         patch('services.embedding.httpx.AsyncClient') as mock_httpx, \
         patch('services.llm.Groq') as mock_groq:
        
        # Configure mocks
        mock_pinecone.return_value = Mock()
        mock_httpx.return_value = Mock()
        mock_groq.return_value = Mock()
        
        # Import app after mocking
        from app import app
        from fastapi.testclient import TestClient
        
        yield TestClient(app)

def test_read_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["service"] == "HackRX Query System"
    assert json_response["version"] == "v1"
    assert json_response["status"] == "running"

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "healthy"
    assert "version" in json_response
    assert "services" in json_response

