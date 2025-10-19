import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any

from autogen_core import CancellationToken
from autogen_core.memory import (
    MemoryContent,
    MemoryQueryResult,
    MemoryMimeType,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage


# Mock vertexai before importing the module under test
@pytest.fixture(autouse=True)
def mock_vertexai_module():
    """Mock the vertexai module globally for all tests."""
    with patch.dict("sys.modules", {"vertexai": MagicMock()}):
        yield


@pytest.fixture
def mock_client():
    """Create a mock VertexAI client."""
    client = Mock()
    client.agent_engines = Mock()
    client.agent_engines.memories = Mock()
    client.delete = Mock()
    return client


@pytest.fixture
def vertexai_memory(mock_client):
    """Create a VertexaiMemory instance with a mock client."""
    # Import here after mocking
    from autogen_vertexai_memory.memory import VertexaiMemory

    return VertexaiMemory(
        api_resource_name="test-resource",
        project_id="test-project",
        location="us-central1",
        client=mock_client,
    )


class TestVertexaiMemoryInit:
    """Test VertexaiMemory initialization."""

    def test_init_with_client(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )
        assert memory.api_resource_name == "test-resource"
        assert memory.project_id == "test-project"
        assert memory.location == "us-central1"
        assert memory.client == mock_client

    def test_init_without_client(self):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=None,
        )
        assert memory.api_resource_name == "test-resource"
        assert memory.project_id == "test-project"
        assert memory.location == "us-central1"
        assert memory.client is None

    def test_component_type(self):
        from autogen_vertexai_memory.memory import VertexaiMemory

        assert VertexaiMemory.component_type == "memory"


class TestVertexaiClient:
    """Test VertexAI client initialization."""

    @pytest.mark.asyncio
    async def test_vertexai_client_already_exists(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )
        client = await memory.vertexai_client()
        assert client == mock_client


class TestAdd:
    """Test adding memories."""

    @pytest.mark.asyncio
    async def test_add_with_client(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        content = MemoryContent(content="Test memory", mime_type=MemoryMimeType.TEXT)
        user_id = "user123"

        mock_response = Mock()
        mock_client.agent_engines.memories.generate.return_value = mock_response

        with patch("builtins.print") as mock_print:
            await memory.add(content, user_id)

        mock_client.agent_engines.memories.generate.assert_called_once_with(
            name="test-resource",
            direct_memories_source={"direct_memories": [{"fact": "Test memory"}]},
            scope={"app_name": "test-resource", "user_id": "user123"},
        )
        mock_print.assert_called_once_with(mock_response)

    @pytest.mark.asyncio
    async def test_add_with_cancellation_token(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        content = MemoryContent(content="Test memory", mime_type=MemoryMimeType.TEXT)
        user_id = "user123"
        token = CancellationToken()

        mock_client.agent_engines.memories.generate.return_value = Mock()

        with patch("builtins.print"):
            await memory.add(content, user_id, cancellation_token=token)

        mock_client.agent_engines.memories.generate.assert_called_once()


class TestQuery:
    """Test querying memories."""

    @pytest.mark.asyncio
    async def test_query_with_search_string(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        mock_retrieved = Mock()
        mock_retrieved.memory.fact = "Retrieved fact"
        mock_client.agent_engines.memories.retrieve.return_value = [mock_retrieved]

        with patch("builtins.print"):
            result = await memory.query(query="search term", user_id="user123")

        mock_client.agent_engines.memories.retrieve.assert_called_once_with(
            name="test-resource",
            scope={"app_name": "test-resource", "user_id": "user123"},
            similarity_search_params={"search_query": "search term", "top_k": 3},
        )

        assert isinstance(result, MemoryQueryResult)
        assert len(result.results) == 1
        assert result.results[0].content == "Retrieved fact"
        assert result.results[0].mime_type == MemoryMimeType.TEXT

    @pytest.mark.asyncio
    async def test_query_with_memory_content(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        query_content = MemoryContent(
            content="search term", mime_type=MemoryMimeType.TEXT
        )
        mock_retrieved = Mock()
        mock_retrieved.memory.fact = "Retrieved fact"
        mock_client.agent_engines.memories.retrieve.return_value = [mock_retrieved]

        with patch("builtins.print"):
            result = await memory.query(query=query_content, user_id="user123")

        mock_client.agent_engines.memories.retrieve.assert_called_once()
        call_args = mock_client.agent_engines.memories.retrieve.call_args[1]
        assert call_args["similarity_search_params"]["search_query"] == "search term"

    @pytest.mark.asyncio
    async def test_query_empty_returns_all(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        mock_retrieved1 = Mock()
        mock_retrieved1.memory.fact = "Fact 1"
        mock_retrieved2 = Mock()
        mock_retrieved2.memory.fact = "Fact 2"
        mock_client.agent_engines.memories.retrieve.return_value = [
            mock_retrieved1,
            mock_retrieved2,
        ]

        result = await memory.query(query="", user_id="user123")

        mock_client.agent_engines.memories.retrieve.assert_called_once_with(
            name="test-resource",
            scope={"user_id": "user123"},
        )

        assert len(result.results) == 2
        assert result.results[0].content == "Fact 1"
        assert result.results[1].content == "Fact 2"

    @pytest.mark.asyncio
    async def test_query_with_multiple_results(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        mock_results = [Mock() for _ in range(3)]
        for i, mock_result in enumerate(mock_results):
            mock_result.memory.fact = f"Fact {i+1}"
        mock_client.agent_engines.memories.retrieve.return_value = mock_results

        with patch("builtins.print"):
            result = await memory.query(query="test", user_id="user123")

        assert len(result.results) == 3
        for i, memory_item in enumerate(result.results):
            assert memory_item.content == f"Fact {i+1}"


class TestUpdateContext:
    """Test updating chat context."""

    @pytest.mark.asyncio
    async def test_update_context_with_memories(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        mock_context = Mock(spec=ChatCompletionContext)
        mock_context.add_message = AsyncMock()

        mock_memory1 = MemoryContent(content="Memory 1", mime_type=MemoryMimeType.TEXT)
        mock_memory2 = MemoryContent(content="Memory 2", mime_type=MemoryMimeType.TEXT)

        with patch.object(memory, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = MemoryQueryResult(
                results=[mock_memory1, mock_memory2]
            )

            result = await memory.update_context(mock_context)

        mock_query.assert_called_once()
        mock_context.add_message.assert_called_once()

        call_args = mock_context.add_message.call_args[0][0]
        assert isinstance(call_args, SystemMessage)
        assert "Memory 1" in call_args.content
        assert "Memory 2" in call_args.content
        assert "Relevant memory content" in call_args.content

        assert isinstance(result.memories, MemoryQueryResult)
        assert len(result.memories.results) == 2

    @pytest.mark.asyncio
    async def test_update_context_no_memories(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        mock_context = Mock(spec=ChatCompletionContext)
        mock_context.add_message = AsyncMock()

        with patch.object(memory, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = MemoryQueryResult(results=[])

            result = await memory.update_context(mock_context)

        mock_query.assert_called_once()
        mock_context.add_message.assert_not_called()

        assert isinstance(result.memories, MemoryQueryResult)
        assert len(result.memories.results) == 0

    @pytest.mark.asyncio
    async def test_update_context_formatting(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        mock_context = Mock(spec=ChatCompletionContext)
        mock_context.add_message = AsyncMock()

        memories = [
            MemoryContent(content="First memory", mime_type=MemoryMimeType.TEXT),
            MemoryContent(content="Second memory", mime_type=MemoryMimeType.TEXT),
        ]

        with patch.object(memory, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = MemoryQueryResult(results=memories)

            await memory.update_context(mock_context)

        call_args = mock_context.add_message.call_args[0][0]
        content = call_args.content

        assert "1. First memory" in content
        assert "2. Second memory" in content
        assert content.startswith("\nRelevant memory content")


class TestClear:
    """Test clearing memories."""

    @pytest.mark.asyncio
    async def test_clear_with_existing_client(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        await memory.clear()
        mock_client.delete.assert_called_once_with(force=True)


class TestClose:
    """Test closing/cleanup."""

    @pytest.mark.asyncio
    async def test_close(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        # Should not raise any exceptions
        await memory.close()


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_query_with_kwargs(self, mock_client):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        mock_client.agent_engines.memories.retrieve.return_value = []

        result = await memory.query(
            query="test", user_id="user123", extra_param="value"
        )

        assert isinstance(result, MemoryQueryResult)

    @pytest.mark.asyncio
    async def test_query_no_results(self, mock_client):
        """Test query returns empty result when no memories found."""
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        mock_client.agent_engines.memories.retrieve.return_value = []

        with patch("builtins.print"):
            result = await memory.query(query="nonexistent", user_id="user123")

        assert isinstance(result, MemoryQueryResult)
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_add_converts_content_to_string(self, mock_client):
        """Test that content is properly converted to string."""
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(
            api_resource_name="test-resource",
            project_id="test-project",
            location="us-central1",
            client=mock_client,
        )

        # Test with various content types that can be stringified
        content = MemoryContent(content="String content", mime_type=MemoryMimeType.TEXT)
        mock_client.agent_engines.memories.generate.return_value = Mock()

        with patch("builtins.print"):
            await memory.add(content, "user123")

        call_args = mock_client.agent_engines.memories.generate.call_args[1]
        assert (
            call_args["direct_memories_source"]["direct_memories"][0]["fact"]
            == "String content"
        )
