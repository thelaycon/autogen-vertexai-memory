import asyncio
import os
from typing import Any, List, Optional

from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from vertexai import Client


class SearchQueryInputArgs(BaseModel):
    query: str = Field(
        ..., description="Semantic search query to retrieve information about user."
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
    for connecting to and using VertexAI for memory operations.
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
    def validate_not_empty(cls, v: str, info: ValidationInfo) -> str:
        """
        Validate that required fields are not empty strings.
        """
        if not v or not v.strip():
            # Handle potential None for field_name safely
            field_name = info.field_name or "Unknown Field"

            env_var_map = {
                "project_id": "VERTEX_PROJECT_ID",
                "location": "VERTEX_LOCATION",
                "user_id": "VERTEX_USER_ID",
                "api_resource_name": "VERTEX_API_RESOURCE_NAME",
            }
            env_var = env_var_map.get(field_name, field_name.upper())
            raise ValueError(
                f"{field_name} must be provided or {env_var} "
                f"environment variable must be set"
            )
        return v


class BaseVertexaiMemoryTool:
    """
    Base class providing shared client initialization and configuration
    for all VertexAI memory tools.
    """

    def __init__(
        self,
        config: Optional[VertexaiMemoryToolConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with shared configuration.
        """
        if config is None:
            self._config = VertexaiMemoryToolConfig(**kwargs)
        else:
            self._config = config

        self.project_id = self._config.project_id
        self.location = self._config.location
        self.user_id = self._config.user_id
        self.api_resource_name = self._config.api_resource_name

        # Type hint client as Optional[Any] or specifically Client if type is known
        self.client: Optional[Any] = None

    def initialize_client(self) -> None:
        """
        Establish connection to the VertexAI service.
        """
        if self.client is None:
            self.client = Client(project=self.project_id, location=self.location)


class SearchVertexaiMemoryTool(
    BaseVertexaiMemoryTool, BaseTool[SearchQueryInputArgs, SearchQueryReturn]
):
    """
    Semantic memory search interface for VertexAI.
    """

    def __init__(
        self,
        config: Optional[VertexaiMemoryToolConfig] = None,
        **kwargs: Any,
    ) -> None:
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
        cancellation_token: Optional[CancellationToken] = None,
    ) -> SearchQueryReturn:
        self.initialize_client()

        # Ensure client is initialized for type checkers
        if not self.client:
            raise RuntimeError("Failed to initialize VertexAI Client")

        def _retrieve_memories_sync() -> List[Any]:
            # self.client is Any, so this bypasses strict attribute checking,
            # which is necessary given the dynamic nature of the Vertex AI SDK structure here.
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
    """

    def __init__(
        self,
        config: Optional[VertexaiMemoryToolConfig] = None,
        **kwargs: Any,
    ) -> None:
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
        cancellation_token: Optional[CancellationToken] = None,
    ) -> UpdateMemoryReturn:
        self.initialize_client()

        # Ensure client is initialized for type checkers
        if not self.client:
            raise RuntimeError("Failed to initialize VertexAI Client")

        def _generate_memory_sync() -> None:
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
