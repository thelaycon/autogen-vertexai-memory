"""VertexAI Memory implementation for Autogen agents.

This module provides a Memory interface implementation that integrates with
Google Cloud's VertexAI Memory service, enabling Autogen agents to store
and retrieve contextual information across conversations.
"""

from typing import Any

from pydantic import BaseModel

from vertexai import Client
from autogen_core import CancellationToken, Component
from autogen_core.memory import (
    Memory,
    UpdateContextResult,
    MemoryContent,
    MemoryQueryResult,
    MemoryMimeType,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage


class VertexaiMemoryConfig(BaseModel):
    """Configuration for VertexAI Memory.

    Attributes:
        api_resource_name (str): Full resource name of the VertexAI memory.
        project_id (str): Google Cloud project ID.
        location (str): GCP region where the memory resource is located.
        user_id (str): Unique identifier for the user associated with the memory.
    """

    api_resource_name: str
    project_id: str
    location: str
    user_id: str


class VertexaiMemory(Memory, Component[VertexaiMemoryConfig]):
    """Memory implementation using Google Cloud VertexAI Memory service.

    This class implements the Autogen Memory protocol to store and retrieve
    agent memories using VertexAI's memory management capabilities. It supports
    semantic search, user-scoped storage, and automatic context updates.

    Attributes:
        component_type (str): Component type identifier for the memory system.
        api_resource_name (str): Full resource name of the VertexAI memory.
        project_id (str): Google Cloud project ID.
        location (str): GCP region where the memory resource is located.
        client (Client | None): VertexAI client instance for API calls.

    Example:
        >>> memory = VertexaiMemory(
        ...     api_resource_name="projects/my-project/locations/us-central1/memories/agent-memory",
        ...     project_id="my-project",
        ...     location="us-central1"
        ... )
        >>>
        >>> # Add a memory
        >>> await memory.add(
        ...     content=MemoryContent(content="Important fact", mime_type=MemoryMimeType.TEXT),
        ...     user_id="user123"
        ... )
        >>>
        >>> # Query memories
        >>> results = await memory.query(query="user preferences", user_id="user123")
    """

    component_type = "memory"
    component_config_schema = VertexaiMemoryConfig

    def __init__(
        self,
        config: VertexaiMemoryConfig | None = None,
        client: Client | None = None,
    ) -> None:
        """Initialize the VertexAI Memory instance.

        Args:
            api_resource_name: Full resource name of the VertexAI memory in the format:
                "projects/{project}/locations/{location}/memories/{memory_id}"
            project_id: Google Cloud project ID where the memory resource exists.
            location: GCP region (e.g., "us-central1", "europe-west1").
            client: Optional pre-configured VertexAI client. If None, a client
                will be created automatically on first use.

        Example:
            >>> memory = VertexaiMemory(
            ...     api_resource_name="projects/my-project/locations/us-central1/memories/my-memory",
            ...     project_id="my-project",
            ...     location="us-central1"
            ... )
        """
        if config is not None:
            self.api_resource_name = config.api_resource_name
            self.project_id = config.project_id
            self.location = config.location
            self.user_id = config.user_id

        self.client = client

    async def vertexai_client(self) -> Client:
        """Initialize and return the VertexAI client.

        Creates a new VertexAI client if one doesn't already exist. The client
        is cached for subsequent use to avoid repeated initialization.

        Returns:
            Client: Configured VertexAI client instance.

        Note:
            This method uses Application Default Credentials (ADC) for
            authentication. Ensure proper GCP authentication is configured
            before calling this method.
        """
        if self.client is None:
            self.client = Client(project=self.project_id, location=self.location)
        return self.client

    async def update_context(
        self, model_context: ChatCompletionContext
    ) -> UpdateContextResult:
        """Update chat context with relevant memories.

        Retrieves all memories for the current context and adds them as a
        system message, allowing the model to utilize stored information
        when generating responses.

        Args:
            model_context: The chat completion context to update with memories.

        Returns:
            UpdateContextResult: Result containing the retrieved memories.

        Note:
            Memories are formatted as a numbered list in chronological order
            and added as a system message at the current position in the
            conversation context.

        Example:
            >>> from autogen_core.model_context import ChatCompletionContext
            >>> context = ChatCompletionContext()
            >>> result = await memory.update_context(context)
            >>> print(len(result.memories.results))  # Number of memories added
        """
        # Query all memories without filtering
        contents = await self.query("Information about the user")
        results = contents.results

        # Return empty result if no memories found
        if not results:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        # Format memories as numbered list
        memory_strings = [
            f"{i}. {str(memory.content)}" for i, memory in enumerate(results, 1)
        ]

        # Add formatted memories to context as system message
        if memory_strings:
            memory_context = (
                "\nRelevant memory content (in chronological order):\n"
                + "\n".join(memory_strings)
                + "\n"
            )
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=contents)

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Add a new memory to VertexAI storage.

        Stores the provided content as a memory fact associated with the
        specified user. The memory can later be retrieved through semantic
        search or by user ID.

        Args:
            content: Memory content to store. The content will be converted
                to a string representation before storage.
            cancellation_token: Optional token to cancel the operation.
                Currently not actively monitored but accepted for protocol
                compliance.

        Raises:
            google.api_core.exceptions.GoogleAPIError: If the API call fails.

        Example:
            >>> from autogen_core.memory import MemoryContent, MemoryMimeType
            >>> content = MemoryContent(
            ...     content="User prefers dark mode",
            ...     mime_type=MemoryMimeType.TEXT
            ... )
            >>> await memory.add(content)

        Note:
            The memory is scoped to both the application (using api_resource_name)
            and the specific user_id to enable multi-tenant isolation.
        """
        # Initialize client if not already done
        if self.client is None:
            await self.vertexai_client()

        # Generate (store) the memory in VertexAI
        self.client.agent_engines.memories.generate(
            name=self.api_resource_name,
            direct_memories_source={
                "direct_memories": [{"fact": str(content.content)}]
            },
            scope={"app_name": self.api_resource_name, "user_id": self.user_id},
        )

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query memories from VertexAI storage.

        Retrieves memories either through semantic similarity search (if a query
        is provided) or returns all memories for a user (if query is empty).

        Args:
            query: Search query string or MemoryContent object. If empty string,
                returns all memories for the user. For semantic search, provide
                a descriptive query.
            cancellation_token: Optional token to cancel the operation.
                Currently not actively monitored but accepted for protocol
                compliance.
            **kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            MemoryQueryResult: Object containing a list of retrieved memory
                contents, ordered by relevance (for search) or chronologically
                (for all memories).

        Example:
            >>> # Semantic search
            >>> results = await memory.query(
            ...     query="What are the user's preferences?",
            ... )
            >>>
            >>> # Get all memories
            >>> all_memories = await memory.query(user_id="user123")
            >>>
            >>> # Process results
            >>> for mem in results.results:
            ...     print(mem.content)

        Note:
            - Semantic search returns top 3 most relevant results
            - Empty query returns all memories (may be large for active users)
            - Results are printed to stdout for debugging (consider removing in production)
        """
        # Initialize client if not already done
        if self.client is None:
            await self.vertexai_client()

        if query != "":
            # Perform semantic similarity search
            # Convert MemoryContent to string if needed
            query_text = query if isinstance(query, str) else str(query.content)

            retrieved_memories = list(
                self.client.agent_engines.memories.retrieve(
                    name=self.api_resource_name,
                    scope={"app_name": self.api_resource_name, "user_id": self.user_id},
                    similarity_search_params={
                        "search_query": query_text,
                        "top_k": 3,  # Return top 3 most relevant memories
                    },
                )
            )
        else:
            # Retrieve all memories for the user
            retrieved_memories = list(
                self.client.agent_engines.memories.retrieve(
                    name=self.api_resource_name,
                    scope={"user_id": self.user_id},
                )
            )

        # Convert VertexAI memory objects to MemoryContent objects
        results = [
            MemoryContent(
                content=retrieved_memory.memory.fact, mime_type=MemoryMimeType.TEXT
            )
            for retrieved_memory in retrieved_memories
        ]

        return MemoryQueryResult(results=results)

    async def clear(self) -> None:
        """Clear all memory content by deleting the resource.

        Permanently deletes the entire memory resource, including all stored
        memories for all users. This operation cannot be undone.

        Raises:
            google.api_core.exceptions.GoogleAPIError: If the deletion fails.

        Warning:
            This method deletes the entire memory resource. All stored memories
            will be permanently lost. Use with extreme caution.

        Example:
            >>> # Clear all memories (use carefully!)
            >>> await memory.clear()

        Note:
            After calling this method, the memory resource will need to be
            recreated before it can be used again. The force=True parameter
            ensures deletion even if the resource is in use.
        """
        # Initialize client if not already done
        if self.client is None:
            await self.vertexai_client()

        # Force delete the entire memory resource
        self.client.delete(force=True)

    async def close(self) -> None:
        """Cleanup resources and close connections.

        Performs any necessary cleanup operations. Currently a no-op as the
        VertexAI client handles its own resource management, but provided
        for protocol compliance and future extensibility.

        Example:
            >>> # Cleanup when done
            >>> await memory.close()

        Note:
            This method is safe to call multiple times and can be used in
            async context managers or cleanup routines.
        """
        pass
