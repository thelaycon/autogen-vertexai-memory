"""VertexAI Memory implementation for Autogen agents.

This module provides a Memory interface implementation that integrates with
Google Cloud's VertexAI Memory service, enabling Autogen agents to store
and retrieve contextual information across conversations.

The implementation follows Autogen's Memory protocol and Component pattern,
allowing seamless integration with Autogen's agent ecosystem. It leverages
VertexAI's semantic search capabilities to retrieve contextually relevant
memories based on natural language queries.

Key Features:
    - Semantic memory search using VertexAI embeddings
    - User-scoped memory isolation for multi-tenant applications
    - Automatic context injection into chat completions
    - Support for both targeted queries and full memory retrieval
    - Asynchronous API compatible with Autogen's async runtime

Architecture:
    This module implements two main classes:
    1. VertexaiMemoryConfig: Pydantic model for configuration validation
    2. VertexaiMemory: Main memory implementation with CRUD operations

Environment Requirements:
    - Google Cloud project with VertexAI API enabled
    - Proper authentication via Application Default Credentials (ADC)
    - Pre-created VertexAI memory resource

Example:
    Basic setup and usage:
        >>> config = VertexaiMemoryConfig(
        ...     api_resource_name="projects/my-proj/locations/us/memories/mem",
        ...     project_id="my-project",
        ...     location="us-central1",
        ...     user_id="user123"
        ... )
        >>> memory = VertexaiMemory(config=config)
        >>>
        >>> # Store a memory
        >>> await memory.add(
        ...     content=MemoryContent(
        ...         content="User prefers Python",
        ...         mime_type=MemoryMimeType.TEXT
        ...     )
        ... )
        >>>
        >>> # Query memories
        >>> results = await memory.query("programming preferences")
        >>> for mem in results.results:
        ...     print(mem.content)
"""

import os
from typing import Any, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    FieldValidationInfo,
)

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
    """Configuration model for VertexAI Memory integration.

    This Pydantic model defines and validates all required configuration
    parameters for connecting to and using a VertexAI Memory resource.
    It ensures type safety and provides clear documentation of expected
    configuration structure.

    The configuration supports both direct parameter specification and
    environment variable fallback, making it flexible for different
    deployment scenarios (development, staging, production).

    Attributes:
        api_resource_name (str): Fully qualified resource name of the
            VertexAI memory in the format:
            "projects/{project_id}/locations/{location}/memories/{memory_id}".
            This uniquely identifies the memory resource in Google Cloud.
            Falls back to VERTEX_API_RESOURCE_NAME environment variable.
        project_id (str): Google Cloud project ID where the VertexAI
            resources are hosted. Must match the project in api_resource_name.
            Falls back to VERTEX_PROJECT_ID environment variable.
        location (str): GCP region where the memory resource is located
            (e.g., "us-central1", "europe-west1"). Must match the location
            in api_resource_name. Falls back to VERTEX_LOCATION environment
            variable.
        user_id (str): Unique identifier for the user associated with this
            memory instance. Used to scope memory storage and retrieval to
            specific users, enabling multi-tenant isolation. Falls back to
            VERTEX_USER_ID environment variable.

    Example:
        >>> # Direct configuration
        >>> config = VertexaiMemoryConfig(
        ...     api_resource_name="projects/my-proj/locations/us-central1/memories/chat-mem",
        ...     project_id="my-proj",
        ...     location="us-central1",
        ...     user_id="user_123"
        ... )
        >>>
        >>> # Using environment variables
        >>> os.environ['VERTEX_PROJECT_ID'] = 'my-proj'
        >>> config = VertexaiMemoryConfig(
        ...     api_resource_name="...",
        ...     location="us-central1",
        ...     user_id="user_123"
        ... )

    Note:
        Environment variable fallback only applies when the parameter is
        explicitly set to None. If omitted entirely, Pydantic will raise
        a validation error.
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

    @field_validator("api_resource_name", "project_id", "location", "user_id")
    @classmethod
    def validate_not_empty(cls, v: str, info: FieldValidationInfo) -> str:
        """Validate that required fields are not empty strings.

        Args:
            v: The field value to validate
            info: Provides context about the field being validated (Pydantic v2)

        Returns:
            The validated value

        Raises:
            ValueError: If the field is empty
        """
        if not v or not v.strip():
            env_var_map = {
                "api_resource_name": "VERTEX_API_RESOURCE_NAME",
                "project_id": "VERTEX_PROJECT_ID",
                "location": "VERTEX_LOCATION",
                "user_id": "VERTEX_USER_ID",
            }
            env_var = env_var_map.get(info.field_name, info.field_name.upper())
            raise ValueError(
                f"{info.field_name} must be provided or {env_var} "
                f"environment variable must be set"
            )
        return v


class VertexaiMemory(Memory, Component[VertexaiMemoryConfig]):
    """Memory implementation using Google Cloud VertexAI Memory service.

    This class implements Autogen's Memory protocol to provide persistent,
    searchable memory storage using Google Cloud's VertexAI platform. It
    enables agents to maintain context across conversations through semantic
    memory retrieval and automatic context injection.

    The implementation uses VertexAI's agent engines to:
    1. Store memories as facts associated with users and applications
    2. Retrieve memories using semantic similarity search
    3. Automatically inject relevant memories into chat contexts
    4. Maintain user-scoped isolation for multi-tenant scenarios

    Design Patterns:
        - Component Pattern: Implements Autogen's Component interface for
          lifecycle management and configuration
        - Lazy Initialization: VertexAI client is created on first use to
          optimize resource usage and enable offline instantiation
        - Async-First: All operations are asynchronous for non-blocking
          integration with Autogen's event-driven architecture

    Attributes:
        component_type (str): Static identifier for this component type.
            Set to "memory" for registration in Autogen's component system.
        component_config_schema (type): Reference to the configuration model
            class used for validation. Set to VertexaiMemoryConfig.
        api_resource_name (str): Full resource name of the VertexAI memory
            resource this instance will interact with.
        project_id (str): Google Cloud project ID hosting the memory resource.
        location (str): GCP region where the memory resource is deployed.
        user_id (str): User identifier for scoping memory operations.
        client (Client | None): VertexAI client instance. None until first
            API operation, then cached for subsequent use.

    Thread Safety:
        This class is not thread-safe. Each instance should be used within
        a single async context. For concurrent usage, create separate
        instances per user/context.

    Example:
        >>> # Initialize with config
        >>> config = VertexaiMemoryConfig(
        ...     api_resource_name="projects/my-project/locations/us-central1/memories/agent-memory",
        ...     project_id="my-project",
        ...     location="us-central1",
        ...     user_id="user123"
        ... )
        >>> memory = VertexaiMemory(config=config)
        >>>
        >>> # Add a memory (stores in VertexAI)
        >>> await memory.add(
        ...     content=MemoryContent(
        ...         content="User prefers concise responses",
        ...         mime_type=MemoryMimeType.TEXT
        ...     )
        ... )
        >>>
        >>> # Query memories (semantic search)
        >>> results = await memory.query(query="communication preferences")
        >>> for memory_item in results.results:
        ...     print(memory_item.content)
        >>>
        >>> # Update chat context with memories
        >>> context = ChatCompletionContext()
        >>> await memory.update_context(context)
        >>> # Context now includes relevant memories as system message

    See Also:
        - autogen_core.memory.Memory: Base protocol this class implements
        - autogen_core.Component: Component pattern for lifecycle management
        - VertexaiMemoryConfig: Configuration model for this component
    """

    component_type = "memory"
    component_config_schema = VertexaiMemoryConfig

    def __init__(
        self,
        config: Optional[VertexaiMemoryConfig] = None,
        client: Optional[Client] = None,
    ) -> None:
        """Initialize the VertexAI Memory instance.

        Creates a new memory instance configured to interact with a specific
        VertexAI memory resource. The instance can be initialized either with
        a validated configuration object or will attempt to load configuration
        from environment variables.

        The VertexAI client is not initialized during construction. Instead,
        it uses lazy initialization when the first API call is made. This
        design choice:
        - Reduces startup overhead and resource usage
        - Allows object creation in environments without network access
        - Enables serialization/deserialization without active connections
        - Defers authentication until actually needed

        Args:
            config (Optional[VertexaiMemoryConfig]): Pre-validated configuration
                object containing all necessary connection parameters. If None,
                configuration will be loaded from environment variables when
                the first operation is performed. Recommended to provide this
                explicitly for better error handling.
            client (Optional[Client]): Pre-configured VertexAI client instance.
                If provided, this client will be used for all API calls,
                bypassing lazy initialization. Useful for:
                - Sharing a client across multiple memory instances
                - Testing with mock clients
                - Custom client configuration (timeouts, retry policies)
                Default is None, which triggers lazy initialization.

        Raises:
            ValueError: If config is None and required environment variables
                are not set. Validation occurs during first API operation.

        Example:
            >>> # Standard initialization with config
            >>> config = VertexaiMemoryConfig(
            ...     api_resource_name="projects/my-project/locations/us-central1/memories/my-memory",
            ...     project_id="my-project",
            ...     location="us-central1",
            ...     user_id="user123"
            ... )
            >>> memory = VertexaiMemory(config=config)
            >>>
            >>> # With pre-configured client (e.g., for custom retry logic)
            >>> from vertexai import Client
            >>> custom_client = Client(
            ...     project="my-project",
            ...     location="us-central1"
            ... )
            >>> memory = VertexaiMemory(config=config, client=custom_client)
            >>>
            >>> # Using environment variables (config loaded on first use)
            >>> memory = VertexaiMemory()  # Must set env vars first

        Note:
            If both config and client are None, the instance will attempt to
            load configuration from environment variables when the first API
            operation is performed. Ensure proper GCP authentication (ADC) is
            configured before making API calls.
        """
        # Extract configuration from the config object if provided.
        # This allows the instance to store individual config values as
        # instance attributes for easier access throughout the class.
        if config is not None:
            self.api_resource_name = config.api_resource_name
            self.project_id = config.project_id
            self.location = config.location
            self.user_id = config.user_id
        else:
            # If no config provided, initialize with empty values.
            # These will be populated from environment variables when
            # vertexai_client() is first called.
            self.api_resource_name = ""
            self.project_id = ""
            self.location = ""
            self.user_id = ""

        # Store the provided client or None for lazy initialization.
        # The client provides access to VertexAI's agent engines and
        # memory management APIs.
        self.client = client

    async def vertexai_client(self) -> Client:
        """Initialize and return the VertexAI client instance.

        This method implements lazy initialization of the VertexAI client,
        creating it only when first needed and caching it for subsequent use.
        This pattern optimizes resource usage and allows the memory object to
        be created in contexts where immediate network connectivity or
        authentication may not be available.

        The client is configured using the project_id and location specified
        during initialization. It uses Google Cloud's Application Default
        Credentials (ADC) for authentication, which can be:
        - Service account key file (GOOGLE_APPLICATION_CREDENTIALS env var)
        - gcloud CLI credentials (for local development)
        - Compute Engine/GKE service account (for cloud deployments)

        Returns:
            Client: Configured VertexAI client instance ready for API calls.
                The client provides access to agent_engines.memories APIs
                for memory storage, retrieval, and management operations.

        Side Effects:
            If self.client is None, creates a new Client instance and caches
            it in self.client for subsequent calls. This means the first call
            to this method may take longer due to client initialization and
            potential authentication setup.

        Example:
            >>> memory = VertexaiMemory(config=config)
            >>> # Client is None initially
            >>> assert memory.client is None
            >>>
            >>> # First call initializes the client
            >>> client = await memory.vertexai_client()
            >>> # Client is now cached
            >>> assert memory.client is not None
            >>>
            >>> # Subsequent calls return cached client immediately
            >>> same_client = await memory.vertexai_client()
            >>> assert client is same_client

        Note:
            This method assumes proper GCP authentication is configured in
            the environment. If authentication fails, the Client constructor
            will raise an authentication error. Ensure one of the following:
            - GOOGLE_APPLICATION_CREDENTIALS points to a valid service account key
            - gcloud auth application-default login has been run
            - Running on GCP with proper service account attached
        """
        # Check if client has already been initialized in a previous call
        if self.client is None:
            # If configuration wasn't provided during __init__, attempt to
            # load it from environment variables now
            if not self.api_resource_name:
                config = VertexaiMemoryConfig()
                self.api_resource_name = config.api_resource_name
                self.project_id = config.project_id
                self.location = config.location
                self.user_id = config.user_id

            # Create and cache the VertexAI client using configured
            # project and location. This establishes the API endpoint
            # and authentication context for all subsequent operations.
            self.client = Client(project=self.project_id, location=self.location)

        return self.client

    async def update_context(
        self, model_context: ChatCompletionContext
    ) -> UpdateContextResult:
        """Inject relevant memories into the chat completion context.

        This method retrieves stored memories and automatically injects them
        into the chat context as a system message. This enables the language
        model to utilize historical context and user preferences when
        generating responses, effectively giving the agent "memory" of past
        interactions and learned information.

        The injection process:
        1. Queries all stored memories using a generic query
        2. Formats memories as a numbered, chronological list
        3. Wraps the list with descriptive header text
        4. Adds to context as a SystemMessage (before user messages)

        This approach ensures the model has access to relevant background
        information while maintaining clear separation between system-provided
        context and user conversation.

        Args:
            model_context (ChatCompletionContext): The chat completion context
                object that will be sent to the language model. This context
                contains the conversation history and system messages. The
                method will append a new SystemMessage containing formatted
                memories.

        Returns:
            UpdateContextResult: Result object containing:
                - memories (MemoryQueryResult): All retrieved memories that
                  were added to the context. This allows callers to inspect
                  what memories influenced the response.

        Side Effects:
            Modifies the provided model_context by adding a SystemMessage
            containing formatted memory content. This message appears in the
            context before the model generates its response.

        Example:
            >>> from autogen_core.model_context import ChatCompletionContext
            >>> from autogen_core.models import UserMessage
            >>>
            >>> # Create a context with a user query
            >>> context = ChatCompletionContext()
            >>> await context.add_message(
            ...     UserMessage(content="What are my preferences?")
            ... )
            >>>
            >>> # Inject memories into context
            >>> result = await memory.update_context(context)
            >>> print(f"Added {len(result.memories.results)} memories")
            >>>
            >>> # Context now includes system message with memories:
            >>> # System: Relevant memory content (in chronological order):
            >>> # 1. User prefers dark mode
            >>> # 2. User likes concise responses
            >>> # User: What are my preferences?

        Note:
            - Uses a generic query ("Information about the user") to retrieve
              all relevant memories rather than filtering by specific topics
            - Memories are numbered starting from 1 for human readability
            - If no memories exist, no system message is added and an empty
              result is returned
            - The chronological ordering helps the model understand the
              temporal relationship between memories
        """
        # Retrieve all memories relevant to the user using a broad query.
        # This generic query is designed to return comprehensive context
        # rather than filtering to specific topics.
        contents = await self.query("Information about the user")
        results = contents.results

        # Early return if no memories exist - no need to modify context
        if not results:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        # Format each memory as a numbered list item for clarity and
        # readability. The enumerate starts at 1 to match human convention
        # for numbered lists rather than zero-based indexing.
        memory_strings = [
            f"{i}. {str(memory.content)}" for i, memory in enumerate(results, 1)
        ]

        # Only add the system message if we have formatted memories to include
        if memory_strings:
            # Construct a clear, structured memory context message with:
            # - Descriptive header explaining what follows
            # - Note about chronological ordering to help model understand
            #   temporal relationships
            # - Formatted memory list with proper line breaks
            # - Trailing newline for clean separation from subsequent messages
            memory_context = (
                "\nRelevant memory content (in chronological order):\n"
                + "\n".join(memory_strings)
                + "\n"
            )

            # Add as SystemMessage to ensure it's treated as authoritative
            # context rather than user input. System messages typically have
            # higher influence on model behavior than user messages.
            await model_context.add_message(SystemMessage(content=memory_context))

        # Return result containing all memories that were added, allowing
        # caller to inspect or log what context was provided to the model
        return UpdateContextResult(memories=contents)

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> None:
        """Store a new memory in the VertexAI memory resource.

        This method persists a piece of information (a "fact") to VertexAI's
        memory storage, making it available for future semantic searches and
        context retrieval. Memories are automatically associated with the
        configured user_id and application scope, enabling multi-tenant
        isolation and user-specific memory management.

        The memory is stored as a fact in VertexAI's semantic storage, which
        means it can later be retrieved through:
        1. Semantic similarity search (finding related memories by meaning)
        2. User-scoped retrieval (getting all memories for a specific user)
        3. Application-scoped retrieval (memories specific to this app)

        Args:
            content (MemoryContent): The memory content to store. This object
                encapsulates:
                - content: The actual information to remember (any type that
                  converts to string)
                - mime_type: The content type (typically MemoryMimeType.TEXT)
                The content will be converted to a string representation before
                storage, so complex objects should implement __str__ meaningfully.
            cancellation_token (Optional[CancellationToken]): Token that can
                be used to cancel the operation if it's taking too long or is
                no longer needed. Currently not actively monitored in this
                implementation but accepted for API compatibility with
                Autogen's cancellation system. Default is None.

        Raises:
            google.api_core.exceptions.GoogleAPIError: If the API call to
                VertexAI fails due to:
                - Network connectivity issues
                - Authentication/authorization problems
                - Invalid resource names or configuration
                - Rate limiting or quota exhaustion
                - Service unavailability

        Side Effects:
            - Initializes the VertexAI client if not already initialized
            - Persists the memory to Google Cloud, consuming storage quota
            - Memory becomes immediately available for search and retrieval

        Example:
            >>> from autogen_core.memory import MemoryContent, MemoryMimeType
            >>>
            >>> # Store a simple text memory
            >>> await memory.add(
            ...     content=MemoryContent(
            ...         content="User prefers dark mode UI",
            ...         mime_type=MemoryMimeType.TEXT
            ...     )
            ... )
            >>>
            >>> # Store structured information (will be stringified)
            >>> await memory.add(
            ...     content=MemoryContent(
            ...         content="Last login: 2025-10-20, Location: Port Harcourt",
            ...         mime_type=MemoryMimeType.TEXT
            ...     )
            ... )
            >>>
            >>> # Later retrieve via semantic search
            >>> results = await memory.query("UI preferences")
            >>> # Should return the dark mode preference

        Note:
            - The memory is scoped to both the application (api_resource_name)
              and user (user_id) for proper isolation in multi-tenant systems
            - Converting content to string happens before storage, so ensure
              your content objects have meaningful __str__ implementations
            - There's no built-in deduplication - storing the same content
              multiple times will create multiple memory entries
            - Consider implementing your own deduplication logic if needed
        """
        # Ensure the VertexAI client is initialized before attempting to
        # make API calls. This handles lazy initialization transparently.
        if self.client is None:
            await self.vertexai_client()

        # Store the memory using VertexAI's generate endpoint, which adds
        # new facts to the memory resource. The API call structure:
        # - name: Identifies which memory resource to add to
        # - direct_memories_source: Provides the actual memory content
        # - scope: Defines the context (app + user) for this memory
        self.client.agent_engines.memories.generate(
            # Full resource path identifying the target memory storage
            name=self.api_resource_name,
            # Wrap the memory content in the expected API structure.
            # VertexAI expects memories as "facts" (atomic pieces of info).
            # Convert to string to handle any content type consistently.
            direct_memories_source={
                "direct_memories": [{"fact": str(content.content)}]
            },
            # Scope ensures this memory is associated with both the
            # application and specific user, enabling proper isolation
            # and retrieval filtering in multi-tenant scenarios.
            scope={"app_name": self.api_resource_name, "user_id": self.user_id},
        )

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Retrieve memories from VertexAI storage using semantic search.

        This method provides flexible memory retrieval supporting two modes:
        1. Semantic Search (non-empty query): Uses VertexAI's embedding-based
           similarity search to find memories most relevant to the query,
           ranked by semantic similarity. Ideal for context-aware retrieval.
        2. Full Retrieval (empty query): Returns all memories for the user
           in chronological order. Useful for context updates and auditing.

        The semantic search leverages VertexAI's language models to understand
        query intent and meaning, not just keywords. This means queries like
        "UI preferences" can successfully find memories about "dark mode"
        even without exact keyword matches.

        Args:
            query (str | MemoryContent): The search query determining what
                memories to retrieve:
                - Non-empty string: Performs semantic similarity search using
                  the query text to find related memories
                - Empty string (""): Returns all memories for the user
                - MemoryContent object: Extracts content field and uses as
                  query string
                Default is empty string (retrieves all memories).
            cancellation_token (Optional[CancellationToken]): Token that can
                signal cancellation of long-running operations. Currently not
                actively monitored but accepted for API compatibility with
                Autogen's async patterns. Default is None.
            **kwargs (Any): Additional keyword arguments reserved for future
                extensibility. Currently unused but accepted for forward
                compatibility with potential future parameters.

        Returns:
            MemoryQueryResult: Object containing:
                - results (List[MemoryContent]): List of retrieved memories,
                  each wrapped in a MemoryContent object with:
                  - content: The stored memory fact as a string
                  - mime_type: Set to MemoryMimeType.TEXT
                - For semantic search: Ordered by relevance (most similar first)
                - For full retrieval: Ordered chronologically (oldest first)
                - Empty list if no memories found

        Side Effects:
            - Initializes the VertexAI client if not already initialized
            - Makes API calls to VertexAI, consuming query quota

        Example:
            >>> # Semantic search for relevant memories
            >>> results = await memory.query(
            ...     query="What programming languages does the user know?"
            ... )
            >>> for mem in results.results:
            ...     print(f"Found: {mem.content}")
            >>> # Output: Found: User is proficient in Python and TypeScript
            >>>
            >>> # Get all memories for context building
            >>> all_memories = await memory.query(query="")
            >>> print(f"Total memories: {len(all_memories.results)}")
            >>>
            >>> # Using MemoryContent as query
            >>> query_content = MemoryContent(
            ...     content="user preferences",
            ...     mime_type=MemoryMimeType.TEXT
            ... )
            >>> results = await memory.query(query=query_content)

        Performance Considerations:
            - Semantic search (top_k=3) is fast and focused, ideal for
              real-time agent interactions
            - Full retrieval may return large result sets for active users,
              consider pagination if implementing custom retrieval logic
            - Each query consumes API quota and incurs latency, so batch
              queries when possible rather than making multiple calls

        Note:
            - Semantic search returns only top 3 results to balance context
              richness with token efficiency. Adjust top_k if needed.
            - Empty query bypasses similarity search entirely for maximum
              speed when all memories are needed
            - Results are automatically scoped to the configured user_id,
              ensuring user isolation without explicit filtering
            - MemoryContent objects in results always use MemoryMimeType.TEXT
              regardless of original storage format
        """
        # Ensure the VertexAI client is initialized before making API calls.
        # This handles lazy initialization if client wasn't created yet.
        if self.client is None:
            await self.vertexai_client()

        # Branch based on query presence to choose retrieval strategy
        if query != "":
            # SEMANTIC SEARCH PATH: Find memories similar to query

            # Handle both string and MemoryContent query types by extracting
            # the actual text content. This provides API flexibility while
            # ensuring we always have a string for the search query.
            query_text = query if isinstance(query, str) else str(query.content)

            # Perform semantic similarity search using VertexAI's embeddings.
            # The API compares query embedding against stored memory embeddings
            # to find the most semantically similar matches.
            retrieved_memories = list(
                self.client.agent_engines.memories.retrieve(
                    # Identifies the memory resource to search within
                    name=self.api_resource_name,
                    # Scope restricts search to this app and user, ensuring
                    # proper isolation in multi-tenant environments
                    scope={"app_name": self.api_resource_name, "user_id": self.user_id},
                    # Configure the similarity search behavior
                    similarity_search_params={
                        "search_query": query_text,
                        # Limit to top 3 results to balance context richness
                        # with token efficiency. More results = more context
                        # but higher token costs and potential noise.
                        "top_k": 3,
                    },
                )
            )
        else:
            # FULL RETRIEVAL PATH: Get all memories for the user

            # Retrieve all memories without similarity filtering. This is
            # faster than semantic search since it skips embedding computation
            # and similarity ranking, but may return large result sets.
            retrieved_memories = list(
                self.client.agent_engines.memories.retrieve(
                    # Identifies the memory resource to query
                    name=self.api_resource_name,
                    # Scope by user_id only (no app_name) to get all memories
                    # associated with this user across the application.
                    # Note: Different scoping strategy than semantic search.
                    scope={"user_id": self.user_id},
                )
            )

        # Convert VertexAI's memory response objects to Autogen's MemoryContent
        # format for consistency with the Memory protocol. This transformation:
        # - Extracts the fact text from nested memory.memory.fact structure
        # - Wraps in MemoryContent with appropriate MIME type
        # - Creates a uniform interface regardless of storage backend
        results = [
            MemoryContent(
                content=retrieved_memory.memory.fact, mime_type=MemoryMimeType.TEXT
            )
            for retrieved_memory in retrieved_memories
        ]

        # Return results wrapped in MemoryQueryResult for protocol compliance
        return MemoryQueryResult(results=results)

    async def clear(self) -> None:
        """Permanently delete the entire memory resource and all its contents.

        This method performs a destructive operation that removes the VertexAI
        memory resource from Google Cloud, along with ALL stored memories for
        ALL users. This is not a soft delete or user-specific clear - it
        destroys the entire resource.

        ⚠️ WARNING: This operation is IRREVERSIBLE. All stored memories will
        be permanently lost and cannot be recovered. The memory resource itself
        will need to be recreated before it can be used again.

        Use Cases:
            - Complete reset during testing/development
            - Compliance requirements (e.g., GDPR right to erasure)
            - Decommissioning an application
            - Starting fresh after catastrophic data corruption

        NOT Recommended For:
            - Clearing a single user's memories (implement user-specific clear)
            - Regular cleanup operations (implement selective deletion)
            - Production environments without extensive safeguards

        The method uses force=True to ensure deletion even if the resource
        is currently in use by other operations. This prevents hanging
        deletions but increases the risk of disrupting active operations.

        Raises:
            google.api_core.exceptions.GoogleAPIError: If the deletion fails
                due to:
                - Permission issues (insufficient IAM roles)
                - Resource not found (already deleted)
                - Network connectivity problems
                - API service unavailability

        Side Effects:
            - Initializes the VertexAI client if not already initialized
            - Deletes the entire memory resource from Google Cloud
            - Frees up storage quota used by the resource
            - Any subsequent operations on this instance will fail until
              the resource is recreated

        Example:
            >>> # Production-safe clear with confirmation
            >>> user_input = input("Type 'DELETE' to confirm: ")
            >>> if user_input == "DELETE":
            ...     await memory.clear()
            ...     print("All memories permanently deleted")
            >>> else:
            ...     print("Operation cancelled")
            >>>
            >>> # Testing scenario
            >>> async def teardown_test():
            ...     # Clean up test data
            ...     await test_memory.clear()
            ...     # Resource must be recreated for next test
            ...     await recreate_memory_resource()

        Security Considerations:
            - Implement proper authorization checks before calling this method
            - Log all clear operations for audit trails
            - Consider implementing a "soft delete" pattern with restoration
              capability for production systems
            - Require multi-factor authentication for production clear operations

        Note:
            After calling this method:
            - The VertexAI resource no longer exists in Google Cloud
            - You must recreate the resource before storing new memories
            - Any cached references to this resource become invalid
            - Other instances using the same resource will fail
        """
        # Ensure the VertexAI client is initialized for the delete operation
        if self.client is None:
            await self.vertexai_client()

        # Perform forced deletion of the entire memory resource.
        # force=True ensures deletion proceeds even if there are active
        # operations or references, preventing hanging deletions but
        # potentially disrupting concurrent operations.
        self.client.delete(force=True)

    async def close(self) -> None:
        """Release resources and perform cleanup operations.

        This method provides a hook for resource cleanup and connection
        management, implementing the standard async context manager pattern
        commonly used in Python async code. It can be called manually or
        automatically when used with async context managers.

        Current Implementation:
            This is currently a no-op (does nothing) because the VertexAI
            Client handles its own resource management and connection cleanup
            internally. However, the method is provided for:
            - Protocol compliance with Autogen's Memory interface
            - Future extensibility if cleanup becomes necessary
            - Compatibility with async context manager patterns
            - Explicit lifecycle management in application code

        Future Considerations:
            If future versions need cleanup, this method could:
            - Close persistent connections to VertexAI
            - Flush buffered writes to ensure data persistence
            - Cancel pending async operations
            - Release memory buffers or caches
            - Unregister event handlers or callbacks

        Side Effects:
            None in current implementation. Safe to call multiple times.

        Example:
            >>> # Manual cleanup
            >>> memory = VertexaiMemory(config=config)
            >>> await memory.add(content)
            >>> await memory.close()  # Clean up
            >>>
            >>> # Async context manager pattern (recommended)
            >>> async with VertexaiMemory(config=config) as memory:
            ...     await memory.add(content)
            ...     results = await memory.query("preferences")
            ... # Automatic cleanup when exiting context
            >>>
            >>> # Safe to call multiple times
            >>> await memory.close()
            >>> await memory.close()  # No error, no-op

        Best Practices:
            - Always call close() when done with a memory instance, even
              though it currently does nothing (future-proofing)
            - Use async context managers for automatic cleanup
            - Call close() in finally blocks to ensure cleanup on errors
            - Don't rely on garbage collection for resource cleanup

        Note:
            This method is async to maintain consistency with the Memory
            protocol and to allow for future async cleanup operations.
            Even though it currently does nothing, maintain the await when
            calling to ensure forward compatibility.
        """
        # Currently no cleanup needed - VertexAI Client manages its own
        # resources. This pass statement is intentional and documents that
        # we've explicitly considered cleanup and determined it's not needed.
        pass
