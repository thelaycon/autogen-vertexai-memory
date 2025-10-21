import os
import asyncio
from typing import List, Optional, Any

from pydantic import BaseModel, Field, field_validator, FieldValidationInfo
from autogen_core import CancellationToken
from autogen_core.tools import BaseTool

from vertexai import Client


class SearchQueryInputArgs(BaseModel):
    query: str = Field(
        ..., description="Semantic search query to retrive information about user."
    )
    top_k: int = Field(
        5, description="The maximum number of relevant memories to retrieve."
    )


class SearchQueryReturn(BaseModel):
    results: List[str] = Field(
        ..., description="Semantic search query results from Vertex AI Memory Bank."
    )


class UpdateMemoryInputArgs(BaseModel):
    content: str = Field(
        ..., description="The memory content to store as a fact in the memory bank."
    )


class UpdateMemoryReturn(BaseModel):
    success: bool = Field(
        ..., description="Whether the memory was successfully stored."
    )
    message: str = Field(..., description="Status message about the operation.")


class VertexaiMemoryToolConfig(BaseModel):
    """
    This model defines and validates all required configuration parameters
    for connecting to and using VertexAI for memory operations. It ensures
    type safety and provides clear documentation.

    This config is shared across all VertexAI memory tools (search, update, etc.)
    to avoid redundant initialization.

    Attributes:
        project_id (str): Google Cloud project ID. Falls back to
            `VERTEX_PROJECT_ID` environment variable.
        location (str): Google Cloud region. Falls back to
            `VERTEX_LOCATION` environment variable.
        user_id (str): Unique identifier for the user. Falls back to
            `VERTEX_USER_ID` environment variable.
        api_resource_name (str): Name of the VertexAI memory resource.
            Falls back to `VERTEX_API_RESOURCE_NAME` environment variable.
    """

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
    api_resource_name: str = Field(
        default_factory=lambda: os.getenv("VERTEX_API_RESOURCE_NAME", ""),
        description="Full resource name of the VertexAI memory",
    )

    @field_validator("project_id", "location", "user_id", "api_resource_name")
    @classmethod
    def validate_not_empty(cls, v: str, info: FieldValidationInfo) -> str:
        """
        Validate that required fields are not empty strings.

        Args:
            v: The field value to validate.
            info: Provides context about the field being validated (Pydantic v2).

        Returns:
            The validated value.

        Raises:
            ValueError: If the field is empty.
        """
        if not v or not v.strip():
            env_var_map = {
                "project_id": "VERTEX_PROJECT_ID",
                "location": "VERTEX_LOCATION",
                "user_id": "VERTEX_USER_ID",
                "api_resource_name": "VERTEX_API_RESOURCE_NAME",
            }
            env_var = env_var_map.get(info.field_name, info.field_name.upper())
            raise ValueError(
                f"{info.field_name} must be provided or {env_var} "
                f"environment variable must be set"
            )
        return v


class BaseVertexaiMemoryTool:
    """
    Base class providing shared client initialization and configuration
    for all VertexAI memory tools.

    This prevents config duplication and ensures a single client instance
    is shared across operations.
    """

    def __init__(
        self,
        config: Optional[VertexaiMemoryToolConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize with shared configuration.

        Args:
            config: Pre-validated configuration object. If None, will be
                   loaded from environment variables or kwargs.
            **kwargs: Direct config values for convenience.
        """
        if config is None:
            self._config = VertexaiMemoryToolConfig(**kwargs)
        else:
            self._config = config

        self.project_id = self._config.project_id
        self.location = self._config.location
        self.user_id = self._config.user_id
        self.api_resource_name = self._config.api_resource_name

        self.client: Optional[Client] = None

    def initialize_client(self):
        """
        Establish connection to the VertexAI service.

        This method creates and initializes the VertexAI Client instance
        using the configured project_id and location. The client provides
        access to VertexAI's agent engines and memory capabilities.

        The initialization is performed lazily (only when needed) to:
        - Reduce startup time and resource usage
        - Avoid connection errors during object construction
        - Allow object serialization without active connections

        Side Effects:
            Sets self.client to an initialized Client instance.

        Note:
            This method assumes valid GCP credentials are available in
            the environment (via Application Default Credentials or
            service account key file).
        """
        if self.client is None:
            self.client = Client(project=self.project_id, location=self.location)


class SearchVertexaiMemoryTool(
    BaseVertexaiMemoryTool, BaseTool[SearchQueryInputArgs, SearchQueryReturn]
):
    """
    Semantic memory search interface for VertexAI.

    This class provides a high-level interface to search and retrieve
    memories stored in Google Cloud's VertexAI platform using semantic
    similarity. It maintains a stateful connection to VertexAI and
    supports scoped memory retrieval per user and application.

    Example:
        >>> # Using a config object (recommended for multiple tools)
        >>> config = VertexaiMemoryToolConfig(
        ...     project_id="my-gcp-project",
        ...     location="us-central1",
        ...     user_id="user_12345",
        ...     api_resource_name="chat-memories"
        ... )
        >>> search_tool = SearchVertexaiMemoryTool(config=config)
        >>> update_tool = UpdateVertexaiMemoryTool(config=config)  # Reuses config!

        >>> # Using environment variables
        >>> search_tool = SearchVertexaiMemoryTool()
    """

    def __init__(
        self,
        config: Optional[VertexaiMemoryToolConfig] = None,
        **kwargs: Any,
    ):
        BaseVertexaiMemoryTool.__init__(self, config=config, **kwargs)
        BaseTool.__init__(
            self,
            args_type=SearchQueryInputArgs,
            return_type=SearchQueryReturn,
            name="search_vertexai_memory_tool",
            description="Perform a search with given parameters using vertexai memory bank.",
        )

    async def run(
        self,
        args: SearchQueryInputArgs,
        cancellation_token: CancellationToken | None = None,
    ) -> SearchQueryReturn:
        self.initialize_client()

        def _retrieve_memories_sync():
            return list(
                self.client.agent_engines.memories.retrieve(
                    name=self.api_resource_name,
                    scope={"app_name": self.api_resource_name, "user_id": self.user_id},
                    similarity_search_params={
                        "search_query": args.query,
                        "top_k": args.top_k,
                    },
                )
            )

        retrieved_memories_list = await asyncio.to_thread(_retrieve_memories_sync)
        fact_strings = [memory.memory.fact for memory in retrieved_memories_list]
        return SearchQueryReturn(results=fact_strings)


class UpdateVertexaiMemoryTool(
    BaseVertexaiMemoryTool, BaseTool[UpdateMemoryInputArgs, UpdateMemoryReturn]
):
    """
    Memory update interface for VertexAI.

    This class provides a high-level interface to store new memories
    in Google Cloud's VertexAI platform. Memories are stored as atomic
    facts associated with a specific user and application scope.

    Example:
        >>> # Sharing config between tools
        >>> config = VertexaiMemoryToolConfig(
        ...     project_id="my-gcp-project",
        ...     location="us-central1",
        ...     user_id="user_12345",
        ...     api_resource_name="chat-memories"
        ... )
        >>> update_tool = UpdateVertexaiMemoryTool(config=config)
        >>>
        >>> # Store a memory
        >>> result = await update_tool.run(
        ...     UpdateMemoryInputArgs(content="User prefers dark mode")
        ... )
    """

    def __init__(
        self,
        config: Optional[VertexaiMemoryToolConfig] = None,
        **kwargs: Any,
    ):
        BaseVertexaiMemoryTool.__init__(self, config=config, **kwargs)
        BaseTool.__init__(
            self,
            args_type=UpdateMemoryInputArgs,
            return_type=UpdateMemoryReturn,
            name="update_vertexai_memory_tool",
            description="Store a new memory fact in the VertexAI memory bank for the user.",
        )

    async def run(
        self,
        args: UpdateMemoryInputArgs,
        cancellation_token: CancellationToken | None = None,
    ) -> UpdateMemoryReturn:
        self.initialize_client()

        def _generate_memory_sync():
            self.client.agent_engines.memories.generate(
                name=self.api_resource_name,
                direct_memories_source={
                    "direct_memories": [{"fact": str(args.content)}]
                },
                scope={"app_name": self.api_resource_name, "user_id": self.user_id},
            )

        try:
            await asyncio.to_thread(_generate_memory_sync)
            return UpdateMemoryReturn(
                success=True,
                message=f"Successfully stored memory: {args.content[:50]}...",
            )
        except Exception as e:
            return UpdateMemoryReturn(
                success=False, message=f"Failed to store memory: {str(e)}"
            )
