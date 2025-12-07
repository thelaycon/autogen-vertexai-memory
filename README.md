# autogen-vertexai-memory

VertexAI Memory integration for Autogen agents. Store and retrieve agent memories using Google Cloud's VertexAI Memory service with semantic search capabilities and intelligent caching.

## Features

- **Persistent Memory Storage** - Store agent memories in Google Cloud VertexAI
- **Semantic Search** - Find relevant memories using natural language queries
- **Automatic Memory Generation** - Extract and store facts from conversation events
- **Intelligent Caching** - Reduce API calls with configurable cache TTL (default 5 minutes)
- **Automatic Cache Invalidation** - Cache updates automatically on write operations
- **Automatic Context Updates** - Seamlessly inject memories into chat contexts
- **Async/Await Support** - Full async API compatible with Autogen's runtime
- **User-Scoped Isolation** - Multi-tenant memory management
- **Tool Integration** - Ready-to-use tools for agent workflows

## Installation

```bash
pip install autogen-vertexai-memory
```

## Prerequisites

1. **Google Cloud Project** with VertexAI API enabled
2. **Authentication** configured (Application Default Credentials)
3. **VertexAI Memory Resource** created in your project

```bash
# Set up authentication
gcloud auth application-default login

# Enable VertexAI API
gcloud services enable aiplatform.googleapis.com
```

## Quick Start

### Basic Memory Usage

```python
from autogen_vertexai_memory import VertexaiMemory, VertexaiMemoryConfig
from autogen_core.memory import MemoryContent, MemoryMimeType

# Configure memory with caching enabled (default)
config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project/locations/us-central1/......./",
    project_id="my-project",
    location="us-central1",
    user_id="user123",
    cache_enabled=True,        # Enable caching (default)
    cache_ttl_seconds=300      # Cache for 5 minutes (default)
)

memory = VertexaiMemory(config=config)

# Store a memory directly (invalidates cache)
await memory.add(
    content=MemoryContent(
        content="User prefers concise responses and uses Python",
        mime_type=MemoryMimeType.TEXT
    )
)

# Generate, updates, and deletes memories from conversation events (non-blocking)
events = [
    {"role": "user", "content": "I love hiking in the mountains"},
    {"role": "model", "content": "That's wonderful! Do you have a favorite trail?"},
    {"role": "user", "content": "Yes, the Pacific Crest Trail is amazing"}
]
result = await memory.generate_memories_from_events(
    user_id="user123",
    events=events
)
# VertexAI will automatically extract facts like "User loves hiking in mountains". If there are redundant memories, VertexAI will update or delete them.

# Semantic search for relevant memories
results = await memory.query(query="programming preferences")
for mem in results.results:
    print(mem.content)
# Output: User prefers concise responses and uses Python

# Retrieve all memories
all_memories = await memory.query(query="")
```

### Using with Autogen Agents

```python
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import UserMessage

# Create chat context
context = ChatCompletionContext()

# Add user message
await context.add_message(
    UserMessage(content="What programming language should I use?")
)

# Inject relevant memories into context (uses caching)
# First call: Fetches from VertexAI and caches
# Subsequent calls: Returns cached results if still valid
result = await memory.update_context(context)
print(f"Added {len(result.memories.results)} memories to context")

# Now the agent has access to stored preferences
```

### Environment Variables

You can also configure using environment variables:

```bash
export VERTEX_PROJECT_ID="my-project"
export VERTEX_LOCATION="us-central1"
export VERTEX_USER_ID="user123"
export VERTEX_API_RESOURCE_NAME="projects/my-project/locations/us-central1/memories/agent-memory"
```

```python
# Auto-loads from environment
config = VertexaiMemoryConfig()
memory = VertexaiMemory(config=config)
```

## Memory Tools for Agents

Integrate memory capabilities directly into your Autogen agents:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_vertexai_memory.tools import (
    SearchVertexaiMemoryTool,
    UpdateVertexaiMemoryTool,
    VertexaiMemoryToolConfig
)

# Configure memory tools
memory_config = VertexaiMemoryToolConfig(
    project_id="my-project",
    location="us-central1",
    user_id="user123",
    api_resource_name="projects/my-project/locations/us-central1/memories/agent-memory"
)

# Create memory tools
search_tool = SearchVertexaiMemoryTool(config=memory_config)
update_tool = UpdateVertexaiMemoryTool(config=memory_config)

# Create agent with memory tools
agent = AssistantAgent(
    name="memory_assistant",
    model_client=OpenAIChatCompletionClient(model="gpt-4"),
    tools=[search_tool, update_tool],
    system_message="""You are a helpful assistant with memory capabilities.
    
    Use search_vertexai_memory_tool to retrieve relevant information about the user.
    Use update_vertexai_memory_tool to store important facts you learn during conversations.
    """
)

# Now the agent can search and store memories automatically!
# Example conversation:
# User: "I prefer Python for data analysis"
# Agent uses update_vertexai_memory_tool to store this preference
# 
# Later...
# User: "What language should I use for my data project?"
# Agent uses search_vertexai_memory_tool, retrieves the preference, and responds accordingly
```

## API Reference

### VertexaiMemoryConfig

Configuration model for VertexAI Memory with caching support.

```python
VertexaiMemoryConfig(
    api_resource_name: str,      # Full resource name: "projects/{project}/locations/{location}/"
    project_id: str,             # Google Cloud project ID
    location: str,               # GCP region (e.g., "us-central1", "europe-west1")
    user_id: str,                # Unique user identifier for memory isolation
    cache_ttl_seconds: int = 300,  # Cache time-to-live in seconds (0 to disable)
    cache_enabled: bool = True   # Whether to enable caching
)
```

**Caching Behavior:**
- Cache is used by `update_context()` method to reduce repeated API calls
- Cache is automatically invalidated on `add()` and `clear()` operations
- Set `cache_ttl_seconds=0` or `cache_enabled=False` to disable caching
- `query()` method does NOT use caching as queries may vary

**Environment Variables:**
- `VERTEX_API_RESOURCE_NAME`
- `VERTEX_PROJECT_ID`
- `VERTEX_LOCATION`
- `VERTEX_USER_ID`

### VertexaiMemory

Main memory interface implementing Autogen's Memory protocol with intelligent caching.

```python
VertexaiMemory(
    config: Optional[VertexaiMemoryConfig] = None,
    client: Optional[Client] = None
)
```

**Methods:**

#### `add(content, cancellation_token=None)`
Store a new memory and invalidate the cache.

```python
await memory.add(
    content=MemoryContent(
        content="Important fact to remember",
        mime_type=MemoryMimeType.TEXT
    )
)
```

#### `query(query="", cancellation_token=None, **kwargs)`
Search memories or retrieve all. Does NOT use caching.

```python
# Semantic search (top 3 results)
results = await memory.query(query="user preferences")

# Get all memories
all_results = await memory.query(query="")
```

**Returns:** `MemoryQueryResult` with list of `MemoryContent` objects

#### `update_context(model_context)`
Inject memories into chat context as system message. Uses caching to reduce API calls.

```python
context = ChatCompletionContext()
result = await memory.update_context(context)
# Context now includes relevant memories
```

**Caching Details:**
- First call: Fetches from VertexAI and caches results
- Subsequent calls: Returns cached results if still valid
- After cache expiry: Fetches fresh data and updates cache

**Returns:** `UpdateContextResult` with retrieved memories

#### `generate_memories_from_events(user_id, events)`
Generate memories from a list of conversation events. This is a non-blocking operation that triggers VertexAI to automatically extract and store relevant facts from the conversation history.

```python
# Define conversation events
events = [
    {"role": "user", "content": "I'm working on a machine learning project"},
    {"role": "model", "content": "That's great! What framework are you using?"},
    {"role": "user", "content": "I prefer TensorFlow for deep learning"},
    {"role": "model", "content": "TensorFlow is an excellent choice for neural networks"}
]

# Generate memories from events (non-blocking)
result = await memory.generate_memories_from_events(
    user_id="user123",
    events=events
)
print(result)
# Output: {'status': 'ok', 'message': 'Generating memories for user user123'}

# VertexAI will analyze the conversation and automatically extract facts like:
# - "User is working on a machine learning project"
# - "User prefers TensorFlow for deep learning"
```

**Parameters:**
- `user_id` (str): The ID of the user for whom memories are being generated
- `events` (List[Dict[str, str]]): List of conversation events with "role" ("user" or "model") and "content" keys

**Returns:** Dict with status and message confirming memory generation has started

**Note:** This operation is non-blocking (`wait_for_completion=False`). VertexAI processes events asynchronously and extracts relevant facts automatically. Memories will be available for querying once processing completes.

#### `clear()`
Permanently delete all memories and invalidate cache (irreversible).

```python
await memory.clear()  # Use with caution!
```

#### `close()`
Release resources and clear cache.

```python
await memory.close()
```

### Memory Tools

#### VertexaiMemoryToolConfig

Shared configuration for memory tools.

```python
VertexaiMemoryToolConfig(
    project_id: str,
    location: str,
    user_id: str,
    api_resource_name: str
)
```

**Environment Variables:**
- `VERTEX_PROJECT_ID`
- `VERTEX_LOCATION`
- `VERTEX_USER_ID`
- `VERTEX_API_RESOURCE_NAME`

#### SearchVertexaiMemoryTool

Tool for semantic memory search. Automatically used by agents to retrieve relevant memories.

```python
SearchVertexaiMemoryTool(config: Optional[VertexaiMemoryToolConfig] = None, **kwargs)
```

**Tool Name:** `search_vertexai_memory_tool`  
**Description:** Perform a search with given parameters using vertexai memory bank  
**Parameters:**
- `query` (str): Semantic search query to retrieve information about user
- `top_k` (int, default=5): Maximum number of relevant memories to retrieve

**Returns:** `SearchQueryReturn` with list of matching memory strings

#### UpdateVertexaiMemoryTool

Tool for storing new memories. Automatically used by agents to save important information.

```python
UpdateVertexaiMemoryTool(config: Optional[VertexaiMemoryToolConfig] = None, **kwargs)
```

**Tool Name:** `update_vertexai_memory_tool`  
**Description:** Store a new memory fact in the VertexAI memory bank for the user  
**Parameters:**
- `content` (str): The memory content to store as a fact in the memory bank

**Returns:** `UpdateMemoryReturn` with success status and message

## Advanced Examples

### Configuring Cache Behavior

```python
# Disable caching completely
config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project/locations/us-central1/...",
    project_id="my-project",
    location="us-central1",
    user_id="user123",
    cache_enabled=False
)

# Short cache TTL (30 seconds)
config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project/locations/us-central1/...",
    project_id="my-project",
    location="us-central1",
    user_id="user123",
    cache_ttl_seconds=30
)

# Long cache TTL (1 hour)
config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project/locations/us-central1/...",
    project_id="my-project",
    location="us-central1",
    user_id="user123",
    cache_ttl_seconds=3600
)
```

### Generating Memories from Conversations

```python
# Automatically extract and store memories from conversation history
conversation_events = [
    {"role": "user", "content": "I'm a software engineer at Google"},
    {"role": "model", "content": "That's impressive! What technologies do you work with?"},
    {"role": "user", "content": "Mainly Kubernetes and Go for cloud infrastructure"},
    {"role": "model", "content": "Great combination for scalable systems!"},
    {"role": "user", "content": "I also enjoy contributing to open source on weekends"}
]

# Let VertexAI automatically analyze and extract relevant facts
result = await memory.generate_memories_from_events(
    user_id="user123",
    events=conversation_events
)

print(result)
# Output: {'status': 'ok', 'message': 'Generating memories for user user123'}

# VertexAI will process the conversation and extract facts like:
# - "User is a software engineer at Google"
# - "User works with Kubernetes and Go"
# - "User works on cloud infrastructure"
# - "User contributes to open source projects on weekends"

# Wait a moment for processing, then query the extracted memories
import asyncio
await asyncio.sleep(2)  # Give VertexAI time to process

results = await memory.query(query="user's work")
for mem in results.results:
    print(mem.content)
```

### Custom Client Configuration

```python
from vertexai import Client

# Create custom client with specific settings
client = Client(
    project="my-project",
    location="us-central1"
)

memory = VertexaiMemory(config=config, client=client)
```

### Multi-User Isolation

```python
# User 1's memories
user1_config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project/locations/us-central1/...",
    project_id="my-project",
    location="us-central1",
    user_id="user1"
)
user1_memory = VertexaiMemory(config=user1_config)

# User 2's memories (isolated from User 1)
user2_config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project/locations/us-central1/...",
    project_id="my-project",
    location="us-central1",
    user_id="user2"
)
user2_memory = VertexaiMemory(config=user2_config)
```

### Sharing Config Across Tools

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_vertexai_memory.tools import (
    SearchVertexaiMemoryTool,
    UpdateVertexaiMemoryTool,
    VertexaiMemoryToolConfig
)

# Create config once
config = VertexaiMemoryToolConfig(
    project_id="my-project",
    location="us-central1",
    user_id="user123",
    api_resource_name="projects/my-project/locations/us-central1/..."
)

# Share across multiple tools
search_tool = SearchVertexaiMemoryTool(config=config)
update_tool = UpdateVertexaiMemoryTool(config=config)

# Use in multiple agents
agent1 = AssistantAgent(
    name="agent1",
    model_client=OpenAIChatCompletionClient(model="gpt-4"),
    tools=[search_tool, update_tool]
)

agent2 = AssistantAgent(
    name="agent2",
    model_client=OpenAIChatCompletionClient(model="gpt-4"),
    tools=[search_tool]  # This agent can only search, not update
)

# Both agents use the same VertexAI client and configuration
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/thelaycon/autogen-vertexai-memory.git
cd autogen-vertexai-memory

# Install dependencies with Poetry
poetry install

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=autogen_vertexai_memory --cov-report=html

# Type checking
poetry run mypy src/autogen_vertexai_memory

# Linting
poetry run ruff check src/
```

### Project Structure

```
autogen-vertexai-memory/
├── src/
│   └── autogen_vertexai_memory/
│       ├── __init__.py
│       ├── memory/
│       │   ├── __init__.py
│       │   └── _vertexai_memory.py    # Main memory implementation with caching
│       └── tools/
│           ├── __init__.py
│           └── _vertexai_memory_tools.py  # Tool implementations
├── tests/
│   ├── conftest.py
│   └── test_vertexai_memory.py
├── pyproject.toml
└── README.md
```

### Running Tests

The test suite uses mocking to avoid real VertexAI API calls:

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific test class
poetry run pytest tests/test_vertexai_memory.py::TestVertexaiMemoryConfig

# Run with coverage report
poetry run pytest --cov=autogen_vertexai_memory --cov-report=term-missing
```

## Troubleshooting

### Authentication Issues

```python
# Verify authentication
gcloud auth application-default print-access-token

# Set explicit credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### Empty Query Results

```python
# Check if memories exist
all_memories = await memory.query(query="")
print(f"Total memories: {len(all_memories.results)}")

# Verify user_id matches
print(f"Using user_id: {memory.user_id}")
```

### Cache Not Working

```python
# Check cache configuration
print(f"Cache enabled: {memory._cache_enabled}")
print(f"Cache TTL: {memory._cache_ttl_seconds}")

# Manually invalidate cache if needed
memory._invalidate_cache()
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run tests (`poetry run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Write tests for new features
- Follow existing code style
- Update documentation for API changes
- Ensure all tests pass before submitting PR

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- [GitHub Issues](https://github.com/thelaycon/autogen-vertexai-memory/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/thelaycon/autogen-vertexai-memory/discussions) - Questions and community support
- [VertexAI Documentation](https://cloud.google.com/vertex-ai/docs) - Official VertexAI docs
- [Autogen Documentation](https://microsoft.github.io/autogen/) - Autogen framework docs
