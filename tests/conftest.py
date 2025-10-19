"""
Pytest configuration and fixtures for autogen_vertexai_memory tests.
"""

import sys
from unittest.mock import MagicMock
import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_vertexai():
    """Mock the vertexai module for all tests."""
    # Create mock vertexai module
    mock_module = MagicMock()
    sys.modules["vertexai"] = mock_module

    # Mock Client class
    mock_client_class = MagicMock()
    mock_module.Client = mock_client_class

    yield mock_module

    # Cleanup
    if "vertexai" in sys.modules:
        del sys.modules["vertexai"]


# Optional: Configure pytest-asyncio
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as an asyncio test")
