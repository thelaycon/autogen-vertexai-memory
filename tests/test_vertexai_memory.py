import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

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
def memory_config():
    """Create a test memory configuration."""
    from autogen_vertexai_memory.memory import VertexaiMemoryConfig

    return VertexaiMemoryConfig(
        api_resource_name="test-resource",
        project_id="test-project",
        location="us-central1",
        user_id="user123",
    )


@pytest.fixture
def vertexai_memory(mock_client, memory_config):
    """Create a VertexaiMemory instance with a mock client."""
    from autogen_vertexai_memory.memory import VertexaiMemory

    return VertexaiMemory(config=memory_config, client=mock_client)


class TestVertexaiMemoryInit:
    """Test VertexaiMemory initialization."""

    def test_init_with_config_and_client(self, mock_client, memory_config):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(config=memory_config, client=mock_client)
        assert memory.api_resource_name == "test-resource"
        assert memory.project_id == "test-project"
        assert memory.location == "us-central1"
        assert memory.user_id == "user123"
        assert memory.client == mock_client

    def test_init_with_config_without_client(self, memory_config):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(config=memory_config, client=None)
        assert memory.api_resource_name == "test-resource"
        assert memory.project_id == "test-project"
        assert memory.location == "us-central1"
        assert memory.user_id == "user123"
        assert memory.client is None

    def test_init_without_config(self):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(config=None, client=None)
        assert memory.client is None

    def test_component_type(self):
        from autogen_vertexai_memory.memory import VertexaiMemory

        assert VertexaiMemory.component_type == "memory"


class TestVertexaiClient:
    """Test VertexAI client initialization."""

    @pytest.mark.asyncio
    async def test_vertexai_client_already_exists(self, vertexai_memory, mock_client):
        client = await vertexai_memory.vertexai_client()
        assert client == mock_client

    @pytest.mark.asyncio
    async def test_vertexai_client_creates_new(self, memory_config):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(config=memory_config, client=None)

        with patch("vertexai.Client") as mock_client_class:
            mock_new_client = Mock()
            mock_client_class.return_value = mock_new_client

            client = await memory.vertexai_client()

            mock_client_class.assert_called_once_with(
                project="test-project", location="us-central1"
            )
            assert client == mock_new_client
            assert memory.client == mock_new_client


class TestAdd:
    """Test adding memories."""

    @pytest.mark.asyncio
    async def test_add_with_client(self, vertexai_memory, mock_client):
        content = MemoryContent(content="Test memory", mime_type=MemoryMimeType.TEXT)

        mock_response = Mock()
        mock_client.agent_engines.memories.generate.return_value = mock_response

        await vertexai_memory.add(content)

        mock_client.agent_engines.memories.generate.assert_called_once_with(
            name="test-resource",
            direct_memories_source={"direct_memories": [{"fact": "Test memory"}]},
            scope={"app_name": "test-resource", "user_id": "user123"},
        )

    @pytest.mark.asyncio
    async def test_add_with_cancellation_token(self, vertexai_memory, mock_client):
        content = MemoryContent(content="Test memory", mime_type=MemoryMimeType.TEXT)
        token = CancellationToken()

        mock_client.agent_engines.memories.generate.return_value = Mock()

        await vertexai_memory.add(content, cancellation_token=token)

        mock_client.agent_engines.memories.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_initializes_client_if_needed(self, memory_config):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(config=memory_config, client=None)
        content = MemoryContent(content="Test memory", mime_type=MemoryMimeType.TEXT)

        with patch.object(
            memory, "vertexai_client", new_callable=AsyncMock
        ) as mock_init:
            mock_client = Mock()
            mock_client.agent_engines.memories.generate.return_value = Mock()
            mock_init.return_value = mock_client

            await memory.add(content)

            mock_init.assert_called_once()


class TestQuery:
    """Test querying memories."""

    @pytest.mark.asyncio
    async def test_query_with_search_string(self, vertexai_memory, mock_client):
        mock_retrieved = Mock()
        mock_retrieved.memory.fact = "Retrieved fact"
        mock_client.agent_engines.memories.retrieve.return_value = [mock_retrieved]

        result = await vertexai_memory.query(query="search term")

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
    async def test_query_with_memory_content(self, vertexai_memory, mock_client):
        query_content = MemoryContent(
            content="search term", mime_type=MemoryMimeType.TEXT
        )
        mock_retrieved = Mock()
        mock_retrieved.memory.fact = "Retrieved fact"
        mock_client.agent_engines.memories.retrieve.return_value = [mock_retrieved]

        result = await vertexai_memory.query(query=query_content)

        mock_client.agent_engines.memories.retrieve.assert_called_once()
        call_args = mock_client.agent_engines.memories.retrieve.call_args[1]
        assert call_args["similarity_search_params"]["search_query"] == "search term"

    @pytest.mark.asyncio
    async def test_query_empty_returns_all(self, vertexai_memory, mock_client):
        mock_retrieved1 = Mock()
        mock_retrieved1.memory.fact = "Fact 1"
        mock_retrieved2 = Mock()
        mock_retrieved2.memory.fact = "Fact 2"
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
        assert result.results[0].content == "Fact 1"
        assert result.results[1].content == "Fact 2"

    @pytest.mark.asyncio
    async def test_query_with_multiple_results(self, vertexai_memory, mock_client):
        mock_results = [Mock() for _ in range(3)]
        for i, mock_result in enumerate(mock_results):
            mock_result.memory.fact = f"Fact {i+1}"
        mock_client.agent_engines.memories.retrieve.return_value = mock_results

        result = await vertexai_memory.query(query="test")

        assert len(result.results) == 3
        for i, memory_item in enumerate(result.results):
            assert memory_item.content == f"Fact {i+1}"

    @pytest.mark.asyncio
    async def test_query_initializes_client_if_needed(self, memory_config):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(config=memory_config, client=None)

        with patch.object(
            memory, "vertexai_client", new_callable=AsyncMock
        ) as mock_init:
            mock_client = Mock()
            mock_client.agent_engines.memories.retrieve.return_value = []
            mock_init.return_value = mock_client

            await memory.query(query="test")

            mock_init.assert_called_once()


class TestUpdateContext:
    """Test updating chat context."""

    @pytest.mark.asyncio
    async def test_update_context_with_memories(self, vertexai_memory):
        mock_context = Mock(spec=ChatCompletionContext)
        mock_context.add_message = AsyncMock()

        mock_memory1 = MemoryContent(content="Memory 1", mime_type=MemoryMimeType.TEXT)
        mock_memory2 = MemoryContent(content="Memory 2", mime_type=MemoryMimeType.TEXT)

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
        assert "Memory 1" in call_args.content
        assert "Memory 2" in call_args.content
        assert "Relevant memory content" in call_args.content

        assert isinstance(result.memories, MemoryQueryResult)
        assert len(result.memories.results) == 2

    @pytest.mark.asyncio
    async def test_update_context_no_memories(self, vertexai_memory):
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
    async def test_update_context_formatting(self, vertexai_memory):
        mock_context = Mock(spec=ChatCompletionContext)
        mock_context.add_message = AsyncMock()

        memories = [
            MemoryContent(content="First memory", mime_type=MemoryMimeType.TEXT),
            MemoryContent(content="Second memory", mime_type=MemoryMimeType.TEXT),
        ]

        with patch.object(
            vertexai_memory, "query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = MemoryQueryResult(results=memories)

            await vertexai_memory.update_context(mock_context)

        call_args = mock_context.add_message.call_args[0][0]
        content = call_args.content

        assert "1. First memory" in content
        assert "2. Second memory" in content
        assert content.startswith("\nRelevant memory content")


class TestClear:
    """Test clearing memories."""

    @pytest.mark.asyncio
    async def test_clear_with_existing_client(self, vertexai_memory, mock_client):
        await vertexai_memory.clear()
        mock_client.delete.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_clear_initializes_client_if_needed(self, memory_config):
        from autogen_vertexai_memory.memory import VertexaiMemory

        memory = VertexaiMemory(config=memory_config, client=None)

        with patch.object(
            memory, "vertexai_client", new_callable=AsyncMock
        ) as mock_init:
            mock_client = Mock()
            mock_client.delete = Mock()
            mock_init.return_value = mock_client

            await memory.clear()

            mock_init.assert_called_once()
            mock_client.delete.assert_called_once_with(force=True)


class TestClose:
    """Test closing/cleanup."""

    @pytest.mark.asyncio
    async def test_close(self, vertexai_memory):
        # Should not raise any exceptions
        await vertexai_memory.close()


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_query_with_kwargs(self, vertexai_memory, mock_client):
        mock_client.agent_engines.memories.retrieve.return_value = []

        result = await vertexai_memory.query(query="test", extra_param="value")

        assert isinstance(result, MemoryQueryResult)

    @pytest.mark.asyncio
    async def test_query_no_results(self, vertexai_memory, mock_client):
        """Test query returns empty result when no memories found."""
        mock_client.agent_engines.memories.retrieve.return_value = []

        result = await vertexai_memory.query(query="nonexistent")

        assert isinstance(result, MemoryQueryResult)
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_add_converts_content_to_string(self, vertexai_memory, mock_client):
        """Test that content is properly converted to string."""
        content = MemoryContent(content="String content", mime_type=MemoryMimeType.TEXT)
        mock_client.agent_engines.memories.generate.return_value = Mock()

        await vertexai_memory.add(content)

        call_args = mock_client.agent_engines.memories.generate.call_args[1]
        assert (
            call_args["direct_memories_source"]["direct_memories"][0]["fact"]
            == "String content"
        )

    @pytest.mark.asyncio
    async def test_memory_content_with_complex_types(
        self, vertexai_memory, mock_client
    ):
        """Test that various content types are handled correctly."""
        # Test with dict-like content
        content = MemoryContent(
            content={"key": "value", "number": 42}, mime_type=MemoryMimeType.TEXT
        )
        mock_client.agent_engines.memories.generate.return_value = Mock()

        await vertexai_memory.add(content)

        call_args = mock_client.agent_engines.memories.generate.call_args[1]
        fact = call_args["direct_memories_source"]["direct_memories"][0]["fact"]
        assert (
            "key" in fact or "value" in fact
        )  # String representation should contain the data
