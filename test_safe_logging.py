#!/usr/bin/env python3
"""
Unit tests for the safe_logging utility functions
"""

import pytest
import os
import sys

pytestmark = pytest.mark.unit

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the function directly without requiring config
from utils.safe_logging import safe_format_error


def test_safe_format_error_normal_string():
    """Test safe_format_error with normal error messages"""
    
    # Create a test exception
    try:
        raise ValueError("This is a normal error message")
    except ValueError as e:
        result = safe_format_error(e)
        assert result == "This is a normal error message"


def test_safe_format_error_with_braces():
    """Test safe_format_error with error messages containing braces"""
    
    # Create a test exception with braces in the message
    try:
        raise ValueError("Error with {braces} in the message")
    except ValueError as e:
        result = safe_format_error(e)
        assert result == "Error with {{braces}} in the message"


def test_safe_format_error_with_multiple_braces():
    """Test safe_format_error with multiple sets of braces"""
    
    try:
        raise RuntimeError("Multiple {errors} with {different} braces {here}")
    except RuntimeError as e:
        result = safe_format_error(e)
        assert result == "Multiple {{errors}} with {{different}} braces {{here}}"


def test_safe_format_error_with_nested_braces():
    """Test safe_format_error with nested braces"""
    
    try:
        raise Exception("Nested {outer {inner} braces} message")
    except Exception as e:
        result = safe_format_error(e)
        assert result == "Nested {{outer {{inner}} braces}} message"


def test_safe_format_error_empty_message():
    """Test safe_format_error with empty error message"""
    
    try:
        raise Exception("")
    except Exception as e:
        result = safe_format_error(e)
        assert result == ""


def test_safe_format_error_only_braces():
    """Test safe_format_error with only braces in the message"""
    
    try:
        raise Exception("{}{}")
    except Exception as e:
        result = safe_format_error(e)
        assert result == "{{}}{{}}"


def test_safe_format_error_with_json_like_content():
    """Test safe_format_error with JSON-like content in error message"""
    
    try:
        raise Exception('{"key": "value", "error": "invalid format"}')
    except Exception as e:
        result = safe_format_error(e)
        assert result == '{{"key": "value", "error": "invalid format"}}'


def test_safe_format_error_with_format_specifiers():
    """Test safe_format_error with Python format specifiers"""
    
    try:
        raise Exception("Error with format specifier: {0} and {name}")
    except Exception as e:
        result = safe_format_error(e)
        assert result == "Error with format specifier: {{0}} and {{name}}"


def test_safe_format_error_with_non_string_exception():
    """Test safe_format_error handles exceptions that might not stringify normally"""
    
    # Test with None (this shouldn't happen in practice but let's be safe)
    result = safe_format_error(None)
    assert result == "None"


def test_safe_format_error_fallback():
    """Test that the fallback works if something unexpected happens"""
    
    # This is harder to test directly, but we can test with a custom object
    # that raises an exception when converted to string
    class BadException:
        def __str__(self):
            raise Exception("Cannot convert to string")
    
    result = safe_format_error(BadException())
    assert result == "<error message could not be formatted>"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
