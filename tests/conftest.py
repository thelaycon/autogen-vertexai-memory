"""
Pytest configuration and fixtures for autogen_vertexai_memory tests.
"""

import sys
from unittest.mock import MagicMock, Mock
import pytest


def pytest_configure(config):
    """Configure pytest - runs before test collection."""
    # Mock vertexai module BEFORE any test modules are imported
    if "vertexai" not in sys.modules:
        mock_vertexai = MagicMock()
        sys.modules["vertexai"] = mock_vertexai

    config.addinivalue_line("markers", "asyncio: mark test as an asyncio test")


@pytest.fixture(scope="function")
def mock_vertexai():
    """Provide access to the mocked vertexai module."""
    # Reset the Client mock for each test
    mock_module = sys.modules["vertexai"]
    mock_module.Client = MagicMock()
    return mock_module
