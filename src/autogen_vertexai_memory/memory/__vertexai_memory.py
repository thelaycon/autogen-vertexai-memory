"""VertexAI Memory implementation with caching for Autogen agents.

This module extends the base VertexAI Memory implementation with intelligent
caching to reduce API calls and improve performance.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from autogen_core import CancellationToken, Component
from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
)
from vertexai import Client


class VertexaiMemoryConfig(BaseModel):
    """Configuration model for VertexAI Memory integration with caching.

    Attributes:
        api_resource_name (str): Fully qualified resource name of the VertexAI memory.
        project_id (str): Google Cloud project ID.
        location (str): GCP region where the memory resource is located.
        user_id (str): Unique identifier for the user.
        cache_ttl_seconds (int): Time-to-live for cached memories in seconds.
            Default is 300 (5 minutes). Set to 0 to disable caching.
        cache_enabled (bool): Whether to enable caching. Default is True.
    """

    api_resource_name: str = Field(
        default_factory=lambda: os.getenv("VERTEX_API_RESOURCE_NAME", ""),
        description="Full resource name of the VertexAI memory",
    )
    project_id: str = Field(
        default_factory=lambda: os.getenv("VERTEX_PROJECT_ID", ""),
        description="Google Cloud project ID",
    )
    location: str = Field(
        default_factory=lambda: os.getenv("VERTEX_LOCATION", ""),
        description="GCP region where the memory resource is located",
    )
    user_id: str = Field(
        default_factory=lambda: os.getenv("VERTEX_USER_ID", ""),
        description="Unique identifier for the user associated with the memory",
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="Cache time-to-live in seconds (0 to disable)",
        ge=0,
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable caching",
    )

    @field_validator("api_resource_name", "project_id", "location", "user_id")
    @classmethod
    def validate_not_empty(cls, v: str, info: ValidationInfo) -> str:
        """Validate that required fields are not empty strings."""
        if not v or not v.strip():
            field_name = info.field_name or "Unknown Field"

            env_var_map = {
                "api_resource_name": "VERTEX_API_RESOURCE_NAME",
                "project_id": "VERTEX_PROJECT_ID",
                "location": "VERTEX_LOCATION",
                "user_id": "VERTEX_USER_ID",
            }
            env_var = env_var_map.get(field_name, field_name.upper())
            raise ValueError(
                f"{field_name} must be provided or {env_var} "
                f"environment variable must be set"
            )
        return v


class VertexaiMemory(Memory, Component[VertexaiMemoryConfig]):
    """Memory implementation using Google Cloud VertexAI with intelligent caching.

    This implementation includes:
    - Memory caching to reduce API calls
    - Automatic cache invalidation on writes
    - Configurable TTL for cache entries
    - Thread-safe cache operations
    """

    component_type = "memory"
    component_config_schema = VertexaiMemoryConfig

    def __init__(
        self,
        config: Optional[VertexaiMemoryConfig] = None,
        client=None,
    ) -> None:
        """Initialize the VertexAI Memory instance with caching support."""
        if config is not None:
            self.api_resource_name = config.api_resource_name
            self.project_id = config.project_id
            self.location = config.location
            self.user_id = config.user_id
            self._cache_enabled = config.cache_enabled
            self._cache_ttl_seconds = config.cache_ttl_seconds
        else:
            self.api_resource_name = ""
            self.project_id = ""
            self.location = ""
            self.user_id = ""
            self._cache_enabled = True
            self._cache_ttl_seconds = 300

        self.client = client

        # Initialize cache storage
        self._cache: Optional[MemoryQueryResult] = None
        self._cache_timestamp: float = 0.0

    def _ensure_client(self) -> None:
        """Initialize the VertexAI client if not already done.

        This should be called inside a worker thread to avoid blocking.
        """
        if self.client is None:
            if not self.api_resource_name:
                # Fallback to defaults/env vars if not set in init
                config = VertexaiMemoryConfig()
                self.api_resource_name = config.api_resource_name
                self.project_id = config.project_id
                self.location = config.location
                self.user_id = config.user_id

            self.client = Client(project=self.project_id, location=self.location)

    def _is_cache_valid(self) -> bool:
        """Check if the current cache is still valid."""
        if not self._cache_enabled:
            return False

        if self._cache is None:
            return False

        if self._cache_ttl_seconds == 0:
            return False

        elapsed = time.time() - self._cache_timestamp
        return elapsed < self._cache_ttl_seconds

    def _invalidate_cache(self) -> None:
        """Invalidate the current cache."""
        self._cache = None
        self._cache_timestamp = 0.0

    def _update_cache(self, result: MemoryQueryResult) -> None:
        """Update the cache with fresh results."""
        if self._cache_enabled:
            self._cache = result
            self._cache_timestamp = time.time()

    async def update_context(
        self, model_context: ChatCompletionContext
    ) -> UpdateContextResult:
        """Inject relevant memories into the chat completion context with caching.

        This method is async and internally calls self.query (which is now non-blocking).
        """
        # Check if we have valid cached data
        if self._is_cache_valid():
            contents = self._cache
        else:
            # Cache miss or expired - fetch from VertexAI
            # self.query is now properly async (non-blocking)
            contents = await self.query("Information about the user")
            # Update cache with fresh results
            self._update_cache(contents)

        results = contents.results

        if not results:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        memory_strings = [
            f"{i}. {str(memory.content)}" for i, memory in enumerate(results, 1)
        ]

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
        cancellation_token: Optional[CancellationToken] = None,
    ) -> None:
        """Store a new memory and invalidate the cache."""

        def _sync_add():
            self._ensure_client()
            self.client.agent_engines.memories.generate(
                name=self.api_resource_name,
                direct_memories_source={
                    "direct_memories": [{"fact": str(content.content)}]
                },
                scope={"app_name": self.api_resource_name, "user_id": self.user_id},
            )

        # Offload blocking call to thread
        try:
            await asyncio.to_thread(_sync_add)
        except Exception as e:
            print(f"Error adding memory to VertexAI: {e}")
            raise e

        # Invalidate cache since we added new data
        self._invalidate_cache()

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Retrieve memories from VertexAI storage using semantic search.

        Wraps the blocking VertexAI client calls in asyncio.to_thread.
        """

        def _sync_query() -> List[Any]:
            self._ensure_client()

            if query != "":
                query_text = query if isinstance(query, str) else str(query.content)
                return list(
                    self.client.agent_engines.memories.retrieve(
                        name=self.api_resource_name,
                        scope={
                            "app_name": self.api_resource_name,
                            "user_id": self.user_id,
                        },
                        similarity_search_params={
                            "search_query": query_text,
                            "top_k": 3,
                        },
                    )
                )
            else:
                return list(
                    self.client.agent_engines.memories.retrieve(
                        name=self.api_resource_name,
                        scope={"user_id": self.user_id},
                    )
                )

        # Offload blocking call to thread
        try:
            retrieved_memories = await asyncio.to_thread(_sync_query)
        except Exception as e:
            print(f"Error querying VertexAI memory: {e}")
            retrieved_memories = []

        results = [
            MemoryContent(
                content=retrieved_memory.memory.fact, mime_type=MemoryMimeType.TEXT
            )
            for retrieved_memory in retrieved_memories
        ]

        return MemoryQueryResult(results=results)

    async def clear(self) -> None:
        """Permanently delete the entire memory resource and invalidate cache."""

        def _sync_clear():
            self._ensure_client()
            self.client.delete(force=True)

        try:
            await asyncio.to_thread(_sync_clear)
        except Exception as e:
            print(f"Error clearing VertexAI memory: {e}")
            pass

        # Invalidate cache since all data is deleted
        self._invalidate_cache()

    async def close(self) -> None:
        """Release resources and clear cache."""
        self._invalidate_cache()
        # No explicit close needed for Vertex Client usually, but good practice to clear ref
        self.client = None

    async def generate_memories_from_events(
        self, user_id: str, events: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """Generate memories from events. Generation is non-blocking."""

        def _sync_generate():
            self._ensure_client()

            # Convert to acceptable format
            formatted_events = [
                {
                    "content": {
                        "role": event["role"],
                        "parts": [{"text": event["content"]}],
                    }
                }
                for event in events
            ]

            self.client.agent_engines.memories.generate(
                name=self.api_resource_name,
                direct_contents_source={"events": formatted_events},
                scope={"user_id": user_id},
                config={
                    "wait_for_completion": False,
                },
            )

        try:
            await asyncio.to_thread(_sync_generate)
            return {
                "status": "ok",
                "message": "Generating memories for user " + user_id,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
