"""
Comprehensive test suite for VertexAI Memory implementation.

This module contains unit tests for the VertexaiMemory class and its
configuration. Tests use mocking to avoid actual VertexAI API calls,
ensuring fast, isolated, and repeatable test execution.
"""

import os
import pytest
from unittest.mock import Mock, AsyncMock, patch

from pydantic import ValidationError

from autogen_core import CancellationToken
from autogen_core.memory import (
    MemoryContent,
    MemoryQueryResult,
    MemoryMimeType,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage

from autogen_vertexai_memory.memory import (
    VertexaiMemoryConfig,
    VertexaiMemory,
)


@pytest.fixture
def mock_client():
    """Create a fully mocked VertexAI client for testing."""
    client = Mock()
    client.agent_engines = Mock()
    client.agent_engines.memories = Mock()
    client.delete = Mock()
    return client


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for configuration testing."""
    env_vars = {
        "VERTEX_PROJECT_ID": "test-project-from-env",
        "VERTEX_LOCATION": "us-west1",
        "VERTEX_USER_ID": "env-user-123",
        "VERTEX_API_RESOURCE_NAME": "projects/test/locations/us/memories/env-mem",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def memory_config():
    """Create a valid test memory configuration."""
    return VertexaiMemoryConfig(
        api_resource_name="test-resource",
        project_id="test-project",
        location="us-central1",
        user_id="user123",
    )


@pytest.fixture
def vertexai_memory(mock_client, memory_config):
    """Create a VertexaiMemory instance with mocked dependencies."""
    return VertexaiMemory(config=memory_config, client=mock_client)


class TestVertexaiMemoryConfig:
    """Test suite for VertexaiMemoryConfig validation and initialization."""

    def test_config_with_all_parameters(self):
        """Test configuration creation with all parameters provided directly."""
        config = VertexaiMemoryConfig(
            api_resource_name="projects/p/locations/l/memories/m",
            project_id="my-project",
            location="us-central1",
            user_id="user456",
        )
        assert config.api_resource_name == "projects/p/locations/l/memories/m"
        assert config.project_id == "my-project"
        assert config.location == "us-central1"
        assert config.user_id == "user456"

    def test_config_with_environment_variables(self, mock_env_vars):
        """Test configuration falls back to environment variables."""
        config = VertexaiMemoryConfig()
        assert config.project_id == "test-project-from-env"
        assert config.location == "us-west1"
        assert config.user_id == "env-user-123"
        assert "env-mem" in config.api_resource_name

    def test_config_validation_empty_project_id(self):
        """Test that empty project_id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            VertexaiMemoryConfig(
                api_resource_name="test",
                project_id="",
                location="us-central1",
                user_id="user123",
            )
        assert "project_id" in str(exc_info.value)

    def test_config_validation_missing_location(self):
        """Test that missing location raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            VertexaiMemoryConfig(
                api_resource_name="test",
                project_id="project",
                location="",
                user_id="user123",
            )
        assert "location" in str(exc_info.value)


class TestVertexaiMemoryInit:
    """Test suite for VertexaiMemory initialization."""

    def test_init_with_config_and_client(self, mock_client, memory_config):
        """Test initialization with both config and client provided."""
        memory = VertexaiMemory(config=memory_config, client=mock_client)
        assert memory.api_resource_name == "test-resource"
        assert memory.project_id == "test-project"
        assert memory.location == "us-central1"
        assert memory.user_id == "user123"
        assert memory.client == mock_client

    def test_init_with_config_without_client(self, memory_config):
        """Test initialization with config but no client (lazy init)."""
        memory = VertexaiMemory(config=memory_config, client=None)
        assert memory.api_resource_name == "test-resource"
        assert memory.project_id == "test-project"
        assert memory.location == "us-central1"
        assert memory.user_id == "user123"
        assert memory.client is None

    def test_init_without_config(self):
        """Test initialization without config (env var fallback)."""
        memory = VertexaiMemory(config=None, client=None)
        assert memory.client is None

    def test_component_type_attribute(self):
        """Test that component_type is correctly set."""
        assert VertexaiMemory.component_type == "memory"

    def test_component_config_schema_attribute(self):
        """Test that component_config_schema references correct class."""
        assert VertexaiMemory.component_config_schema == VertexaiMemoryConfig


class TestVertexaiClient:
    """Test suite for VertexAI client initialization."""

    @pytest.mark.asyncio
    async def test_vertexai_client_returns_existing(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test that existing client is returned without recreation."""
        client = await vertexai_memory.vertexai_client()
        assert client == mock_client
        assert vertexai_memory.client == mock_client

    @pytest.mark.asyncio
    async def test_vertexai_client_creates_new(self, memory_config, mock_vertexai):
        """Test that client is created when not provided."""
        memory = VertexaiMemory(config=memory_config, client=None)

        # Create mock client instance
        mock_new_client = Mock()
        mock_vertexai.Client.return_value = mock_new_client

        client = await memory.vertexai_client()

        mock_vertexai.Client.assert_called_once_with(
            project="test-project", location="us-central1"
        )
        assert client == mock_new_client
        assert memory.client == mock_new_client

    @pytest.mark.asyncio
    async def test_vertexai_client_loads_config_from_env(
        self, mock_env_vars, mock_vertexai
    ):
        """Test that client initialization loads config from env if needed."""
        memory = VertexaiMemory(config=None, client=None)

        mock_vertexai.Client.return_value = Mock()
        await memory.vertexai_client()

        assert memory.project_id == "test-project-from-env"
        assert memory.location == "us-west1"


class TestAdd:
    """Test suite for adding memories."""

    @pytest.mark.asyncio
    async def test_add_with_existing_client(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test adding a memory with pre-initialized client."""
        content = MemoryContent(
            content="User prefers Python", mime_type=MemoryMimeType.TEXT
        )
        mock_response = Mock()
        mock_client.agent_engines.memories.generate.return_value = mock_response
        await vertexai_memory.add(content)
        mock_client.agent_engines.memories.generate.assert_called_once_with(
            name="test-resource",
            direct_memories_source={
                "direct_memories": [{"fact": "User prefers Python"}]
            },
            scope={"app_name": "test-resource", "user_id": "user123"},
        )

    @pytest.mark.asyncio
    async def test_add_with_cancellation_token(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test that cancellation token is accepted (even if not used)."""
        content = MemoryContent(content="Test memory", mime_type=MemoryMimeType.TEXT)
        token = CancellationToken()
        mock_client.agent_engines.memories.generate.return_value = Mock()
        await vertexai_memory.add(content, cancellation_token=token)
        mock_client.agent_engines.memories.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_initializes_client_if_needed(self, memory_config, mock_vertexai):
        """Test that add() initializes client on first use."""
        memory = VertexaiMemory(config=memory_config, client=None)
        content = MemoryContent(content="Test memory", mime_type=MemoryMimeType.TEXT)

        # Create a properly structured mock client
        mock_client = Mock()
        mock_client.agent_engines = Mock()
        mock_client.agent_engines.memories = Mock()
        mock_client.agent_engines.memories.generate = Mock(return_value=Mock())

        # Mock the Client constructor to return our mock
        mock_vertexai.Client.return_value = mock_client

        await memory.add(content)

        # Verify client was initialized
        mock_vertexai.Client.assert_called_once_with(
            project="test-project", location="us-central1"
        )
        mock_client.agent_engines.memories.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_converts_content_to_string(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test that content is properly converted to string format."""
        content = MemoryContent(content="String content", mime_type=MemoryMimeType.TEXT)
        mock_client.agent_engines.memories.generate.return_value = Mock()
        await vertexai_memory.add(content)
        call_args = mock_client.agent_engines.memories.generate.call_args[1]
        fact = call_args["direct_memories_source"]["direct_memories"][0]["fact"]
        assert fact == "String content"
        assert isinstance(fact, str)

    @pytest.mark.asyncio
    async def test_add_with_complex_content(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test adding memory with complex content types."""
        content = MemoryContent(
            content={"preference": "dark_mode", "language": "en"},
            mime_type=MemoryMimeType.TEXT,
        )
        mock_client.agent_engines.memories.generate.return_value = Mock()
        await vertexai_memory.add(content)
        call_args = mock_client.agent_engines.memories.generate.call_args[1]
        fact = call_args["direct_memories_source"]["direct_memories"][0]["fact"]
        assert isinstance(fact, str)
        assert "preference" in fact or "dark_mode" in fact


class TestQuery:
    """Test suite for querying memories."""

    @pytest.mark.asyncio
    async def test_query_with_search_string(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test semantic search with string query."""
        mock_retrieved = Mock()
        mock_retrieved.memory.fact = "User likes Python"
        mock_client.agent_engines.memories.retrieve.return_value = [mock_retrieved]
        result = await vertexai_memory.query(query="programming preferences")
        mock_client.agent_engines.memories.retrieve.assert_called_once_with(
            name="test-resource",
            scope={"app_name": "test-resource", "user_id": "user123"},
            similarity_search_params={
                "search_query": "programming preferences",
                "top_k": 3,
            },
        )
        assert isinstance(result, MemoryQueryResult)
        assert len(result.results) == 1
        assert result.results[0].content == "User likes Python"
        assert result.results[0].mime_type == MemoryMimeType.TEXT

    @pytest.mark.asyncio
    async def test_query_with_memory_content(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test query with MemoryContent object instead of string."""
        query_content = MemoryContent(
            content="user preferences", mime_type=MemoryMimeType.TEXT
        )
        mock_retrieved = Mock()
        mock_retrieved.memory.fact = "Retrieved fact"
        mock_client.agent_engines.memories.retrieve.return_value = [mock_retrieved]
        result = await vertexai_memory.query(query=query_content)
        mock_client.agent_engines.memories.retrieve.assert_called_once()
        call_args = mock_client.agent_engines.memories.retrieve.call_args[1]
        assert (
            call_args["similarity_search_params"]["search_query"] == "user preferences"
        )

    @pytest.mark.asyncio
    async def test_query_empty_returns_all(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test that empty query retrieves all memories."""
        mock_retrieved1 = Mock()
        mock_retrieved1.memory.fact = "First memory"
        mock_retrieved2 = Mock()
        mock_retrieved2.memory.fact = "Second memory"
        mock_client.agent_engines.memories.retrieve.return_value = [
            mock_retrieved1,
            mock_retrieved2,
        ]
        result = await vertexai_memory.query(query="")
        mock_client.agent_engines.memories.retrieve.assert_called_once_with(
            name="test-resource",
            scope={"user_id": "user123"},
        )
        assert len(result.results) == 2
        assert result.results[0].content == "First memory"
        assert result.results[1].content == "Second memory"

    @pytest.mark.asyncio
    async def test_query_with_multiple_results(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test query returning multiple results."""
        mock_results = [Mock() for _ in range(3)]
        for i, mock_result in enumerate(mock_results):
            mock_result.memory.fact = f"Memory {i+1}"
        mock_client.agent_engines.memories.retrieve.return_value = mock_results
        result = await vertexai_memory.query(query="test query")
        assert len(result.results) == 3
        for i, memory_item in enumerate(result.results):
            assert memory_item.content == f"Memory {i+1}"
            assert memory_item.mime_type == MemoryMimeType.TEXT

    @pytest.mark.asyncio
    async def test_query_initializes_client_if_needed(
        self, memory_config, mock_vertexai
    ):
        """Test that query() initializes client on first use."""
        memory = VertexaiMemory(config=memory_config, client=None)

        # Create a properly structured mock client
        mock_client = Mock()
        mock_client.agent_engines = Mock()
        mock_client.agent_engines.memories = Mock()
        mock_client.agent_engines.memories.retrieve = Mock(return_value=[])

        # Mock the Client constructor to return our mock
        mock_vertexai.Client.return_value = mock_client

        await memory.query(query="test")

        # Verify client was initialized
        mock_vertexai.Client.assert_called_once_with(
            project="test-project", location="us-central1"
        )
        mock_client.agent_engines.memories.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_no_results(self, vertexai_memory, mock_client, mock_vertexai):
        """Test query returns empty result when no memories found."""
        mock_client.agent_engines.memories.retrieve.return_value = []
        result = await vertexai_memory.query(query="nonexistent topic")
        assert isinstance(result, MemoryQueryResult)
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_query_with_kwargs(self, vertexai_memory, mock_client, mock_vertexai):
        """Test that extra kwargs are accepted without error."""
        mock_client.agent_engines.memories.retrieve.return_value = []
        result = await vertexai_memory.query(
            query="test", extra_param="value", another_param=123
        )
        assert isinstance(result, MemoryQueryResult)


class TestUpdateContext:
    """Test suite for updating chat context with memories."""

    @pytest.mark.asyncio
    async def test_update_context_with_memories(self, vertexai_memory, mock_vertexai):
        """Test context update with multiple memories."""
        mock_context = Mock(spec=ChatCompletionContext)
        mock_context.add_message = AsyncMock()

        mock_memory1 = MemoryContent(
            content="User prefers concise responses", mime_type=MemoryMimeType.TEXT
        )
        mock_memory2 = MemoryContent(
            content="User is learning Python", mime_type=MemoryMimeType.TEXT
        )

        with patch.object(
            vertexai_memory, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = MemoryQueryResult(
                results=[mock_memory1, mock_memory2]
            )
            result = await vertexai_memory.update_context(mock_context)

        mock_query.assert_called_once_with("Information about the user")
        mock_context.add_message.assert_called_once()
        call_args = mock_context.add_message.call_args[0][0]
        assert isinstance(call_args, SystemMessage)
        assert "User prefers concise responses" in call_args.content
        assert "User is learning Python" in call_args.content
        assert "Relevant memory content" in call_args.content
        assert isinstance(result.memories, MemoryQueryResult)
        assert len(result.memories.results) == 2

    @pytest.mark.asyncio
    async def test_update_context_no_memories(self, vertexai_memory, mock_vertexai):
        """Test context update when no memories exist."""
        mock_context = Mock(spec=ChatCompletionContext)
        mock_context.add_message = AsyncMock()

        with patch.object(
            vertexai_memory, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = MemoryQueryResult(results=[])
            result = await vertexai_memory.update_context(mock_context)

        mock_query.assert_called_once_with("Information about the user")
        mock_context.add_message.assert_not_called()
        assert isinstance(result.memories, MemoryQueryResult)
        assert len(result.memories.results) == 0

    @pytest.mark.asyncio
    async def test_update_context_formatting(self, vertexai_memory, mock_vertexai):
        """Test that memories are properly formatted in context."""
        mock_context = Mock(spec=ChatCompletionContext)
        mock_context.add_message = AsyncMock()

        memories = [
            MemoryContent(content="First memory fact", mime_type=MemoryMimeType.TEXT),
            MemoryContent(content="Second memory fact", mime_type=MemoryMimeType.TEXT),
            MemoryContent(content="Third memory fact", mime_type=MemoryMimeType.TEXT),
        ]

        with patch.object(
            vertexai_memory, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = MemoryQueryResult(results=memories)
            await vertexai_memory.update_context(mock_context)

        call_args = mock_context.add_message.call_args[0][0]
        content = call_args.content
        assert "1. First memory fact" in content
        assert "2. Second memory fact" in content
        assert "3. Third memory fact" in content
        assert content.startswith("\nRelevant memory content")
        assert "chronological order" in content


class TestClear:
    """Test suite for clearing/deleting memories."""

    @pytest.mark.asyncio
    async def test_clear_with_existing_client(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test clearing with pre-initialized client."""
        await vertexai_memory.clear()
        mock_client.delete.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_clear_initializes_client_if_needed(
        self, memory_config, mock_vertexai
    ):
        """Test that clear() initializes client if not already done."""
        memory = VertexaiMemory(config=memory_config, client=None)

        # Create a properly structured mock client
        mock_client = Mock()
        mock_client.delete = Mock()

        # Mock the Client constructor to return our mock
        mock_vertexai.Client.return_value = mock_client

        await memory.clear()

        # Verify client was initialized
        mock_vertexai.Client.assert_called_once_with(
            project="test-project", location="us-central1"
        )
        mock_client.delete.assert_called_once_with(force=True)


class TestClose:
    """Test suite for cleanup operations."""

    @pytest.mark.asyncio
    async def test_close_no_error(self, vertexai_memory, mock_vertexai):
        """Test that close() completes without error."""
        await vertexai_memory.close()

    @pytest.mark.asyncio
    async def test_close_multiple_calls(self, vertexai_memory, mock_vertexai):
        """Test that close() can be called multiple times safely."""
        await vertexai_memory.close()
        await vertexai_memory.close()
        await vertexai_memory.close()


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_memory_content_with_complex_types(
        self, vertexai_memory, mock_client, mock_vertexai
    ):
        """Test that various content types are handled correctly."""
        content = MemoryContent(
            content={"key": "value", "number": 42}, mime_type=MemoryMimeType.TEXT
        )
        mock_client.agent_engines.memories.generate.return_value = Mock()
        await vertexai_memory.add(content)
        call_args = mock_client.agent_engines.memories.generate.call_args[1]
        fact = call_args["direct_memories_source"]["direct_memories"][0]["fact"]
        assert "key" in fact or "value" in fact
