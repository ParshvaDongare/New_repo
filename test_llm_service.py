#!/usr/bin/env python3
"""
Integration tests for LLM service functionality using mocking
"""

import pytest
import asyncio
import json
import sys
import os
from unittest.mock import patch, Mock, AsyncMock

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Import and apply mock settings before importing services
from test_config import patch_settings_for_testing, MockGroqClient

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def mock_settings():
    """Set up mock settings for the entire test module."""
    return patch_settings_for_testing()

@pytest.fixture
def mock_llm_service(mock_settings):
    """Create a mock LLM service for testing."""
    with patch('groq.Groq') as mock_groq_class:
        # Configure the mock Groq client
        mock_client = MockGroqClient()
        mock_groq_class.return_value = mock_client
        
        # Configure mock responses
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mock LLM response"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 10
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Import the service after mocking
        from services.llm import llm_service
        
        return llm_service

@pytest.mark.asyncio
async def test_safety_check(mock_llm_service):
    """Test the safety check function with mocking."""
    # Mock the safety check to always return safe
    with patch.object(mock_llm_service, '_check_query_safety') as mock_safety:
        mock_safety.return_value = (True, "Query is safe")
        
        safe_query = "What is Article 21?"
        is_safe, reason = await mock_llm_service._check_query_safety(safe_query)
        
        assert is_safe == True
        assert reason == "Query is safe"
        mock_safety.assert_called_once_with(safe_query)

@pytest.mark.asyncio
async def test_basic_llm_call_mock(mock_llm_service):
    """Test basic LLM call with mocking."""
    # Mock the LLM call
    with patch.object(mock_llm_service, '_make_llm_call') as mock_llm_call:
        expected_result = {
            'content': 'Hello, LLM service is working!',
            'tokens': 10
        }
        mock_llm_call.return_value = expected_result
        
        result = await mock_llm_service._make_llm_call(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, LLM service is working!' and nothing else."}
            ],
            model="llama3-8b-8192",
            temperature=0,
            max_tokens=50
        )
        
        assert result['content'] == 'Hello, LLM service is working!'
        assert result['tokens'] == 10
        mock_llm_call.assert_called_once()

@pytest.mark.asyncio
async def test_json_mode_mock(mock_llm_service):
    """Test JSON mode functionality with mocking."""
    # Mock the LLM call for JSON mode
    with patch.object(mock_llm_service, '_make_llm_call') as mock_llm_call:
        expected_json = {"status": "working", "message": "JSON mode test"}
        expected_result = {
            'content': json.dumps(expected_json),
            'tokens': 15
        }
        mock_llm_call.return_value = expected_result
        
        result = await mock_llm_service._make_llm_call(
            messages=[
                {"role": "system", "content": "You are a JSON response generator."},
                {"role": "user", "content": "Return JSON with status: 'working' and message: 'JSON mode test'"}
            ],
            model="llama3-8b-8192",
            temperature=0,
            max_tokens=100,
            json_mode=True
        )
        
        # Verify we can parse the JSON
        json_data = json.loads(result['content'])
        assert json_data['status'] == 'working'
        assert json_data['message'] == 'JSON mode test'
        assert result['tokens'] == 15
        mock_llm_call.assert_called_once()

def test_mock_configuration(mock_settings):
    """Test that mock configuration is working properly."""
    assert mock_settings.app_name == "HackRX Query System"
    assert len(mock_settings.groq_api_keys) > 0
    assert len(mock_settings.jina_api_keys) > 0
    assert mock_settings.groq_api_keys[0] == "mock_groq_key_1"

if __name__ == "__main__":
    # Run the tests using pytest
    pytest.main([__file__, "-v"])
