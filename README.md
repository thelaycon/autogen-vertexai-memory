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
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789",
    app_name="1234567890123456789",
    project_id="my-project-id",
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

# Generate memories from conversation events (non-blocking)
events = [
    {"role": "user", "content": "I love hiking in the mountains"},
    {"role": "model", "content": "That's wonderful! Do you have a favorite trail?"},
    {"role": "user", "content": "Yes, the Pacific Crest Trail is amazing"}
]
result = await memory.generate_memories_from_events(
    user_id="user123",
    events=events
)
# VertexAI will automatically extract facts like "User loves hiking in mountains"

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
export VERTEX_PROJECT_ID="my-project-id"
export VERTEX_LOCATION="us-central1"
export VERTEX_USER_ID="user123"
export VERTEX_APP_NAME="1234567890123456789"
export VERTEX_API_RESOURCE_NAME="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789"
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
    project_id="my-project-id",
    location="us-central1",
    user_id="user123",
    app_name="1234567890123456789",
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789"
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
```

## API Reference

### VertexaiMemoryConfig

Configuration model for VertexAI Memory with caching support.

```python
VertexaiMemoryConfig(
    api_resource_name: str,      # Full resource name: "projects/{project}/locations/{location}/reasoningEngines/{engine_id}"
    app_name: str,               # Application name (typically the reasoning engine ID)
    project_id: str,             # Google Cloud project ID
    location: str,               # GCP region (e.g., "us-central1", "europe-west1")
    user_id: str,                # Unique user identifier for memory isolation
    cache_ttl_seconds: int = 300,  # Cache time-to-live in seconds (0 to disable)
    cache_enabled: bool = True   # Whether to enable caching
)
```

**Caching Behavior:**
- Cache is used by `update_context()` to reduce repeated API calls
- Cache is automatically invalidated on `add()` and `clear()` operations
- Set `cache_ttl_seconds=0` or `cache_enabled=False` to disable caching
- `query()` method does NOT use caching as queries may vary

**Environment Variables:**
- `VERTEX_API_RESOURCE_NAME`
- `VERTEX_APP_NAME`
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
# Semantic search (top 3 results by default)
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
# Context now includes relevant memories in chronological order
```

**Caching Details:**
- First call: Fetches from VertexAI and caches results
- Subsequent calls: Returns cached results if still valid
- After cache expiry: Fetches fresh data and updates cache

**Returns:** `UpdateContextResult` with retrieved memories

#### `generate_memories_from_events(user_id, events)`
Generate memories from conversation events. Non-blocking operation that triggers VertexAI to automatically extract and store relevant facts.

```python
events = [
    {"role": "user", "content": "I'm working on a machine learning project"},
    {"role": "model", "content": "That's great! What framework are you using?"},
    {"role": "user", "content": "I prefer TensorFlow for deep learning"}
]

result = await memory.generate_memories_from_events(
    user_id="user123",
    events=events
)
print(result)
# Output: {'status': 'ok', 'message': 'Generating memories for user user123'}
```

**Parameters:**
- `user_id` (str): User ID for memory generation
- `events` (List[Dict[str, str]]): Events with "role" (user/model) and "content" keys

**Returns:** Dict with status and message

**Note:** Non-blocking operation (`wait_for_completion=False`). VertexAI processes asynchronously.

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

Configuration for memory tools.

```python
VertexaiMemoryToolConfig(
    project_id: str,
    location: str,
    user_id: str,
    api_resource_name: str,
    app_name: str
)
```

**Environment Variables:**
- `VERTEX_PROJECT_ID`
- `VERTEX_LOCATION`
- `VERTEX_USER_ID`
- `VERTEX_API_RESOURCE_NAME`
- `VERTEX_APP_NAME`

#### SearchVertexaiMemoryTool

Tool for semantic memory search.

```python
SearchVertexaiMemoryTool(config: Optional[VertexaiMemoryToolConfig] = None, **kwargs)
```

**Tool Name:** `search_vertexai_memory_tool`  
**Description:** Perform a search with given parameters using vertexai memory bank  
**Parameters:**
- `query` (str): Semantic search query
- `top_k` (int, default=5): Maximum number of memories to retrieve

**Returns:** `SearchQueryReturn` with list of matching memory strings

#### UpdateVertexaiMemoryTool

Tool for storing new memories.

```python
UpdateVertexaiMemoryTool(config: Optional[VertexaiMemoryToolConfig] = None, **kwargs)
```

**Tool Name:** `update_vertexai_memory_tool`  
**Description:** Store a new memory fact in the VertexAI memory bank for the user  
**Parameters:**
- `content` (str): Memory content to store

**Returns:** `UpdateMemoryReturn` with success status and message

## Real-World Example: Personal Shopping Assistant

Here's a complete example of building a shopping assistant that remembers user preferences:

```python
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_vertexai_memory.memory import VertexaiMemory, VertexaiMemoryConfig
from autogen_vertexai_memory.tools import (
    SearchVertexaiMemoryTool,
    UpdateVertexaiMemoryTool,
    VertexaiMemoryToolConfig,
)

# Configure memory for a specific user
memory_config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789",
    app_name="1234567890123456789",
    project_id="my-project-id",
    location="us-central1",
    user_id="customer123",
    cache_enabled=True,
    cache_ttl_seconds=300
)

# Initialize memory
memory = VertexaiMemory(config=memory_config)

# Configure tools with same settings
tool_config = VertexaiMemoryToolConfig(
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789",
    app_name="1234567890123456789",
    project_id="my-project-id",
    location="us-central1",
    user_id="customer123"
)

# Create memory tools
search_tool = SearchVertexaiMemoryTool(config=tool_config)
update_tool = UpdateVertexaiMemoryTool(config=tool_config)

# Create shopping assistant agent
shopping_agent = AssistantAgent(
    name="shopping_assistant",
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"]
    ),
    tools=[search_tool, update_tool],
    memory=[memory],
    system_message="""You are a helpful shopping assistant that remembers customer preferences.
    
    Use search_vertexai_memory_tool to recall what you know about the customer.
    Use update_vertexai_memory_tool to store new preferences you learn.
    
    Always personalize recommendations based on stored memories."""
)

# Example conversation
async def main():
    # First interaction - agent learns preferences
    print("User: I'm looking for running shoes, I prefer Nike brand")
    async for chunk in shopping_agent.run_stream(
        task="I'm looking for running shoes, I prefer Nike brand"
    ):
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
    
    print("\n\n---\n")
    
    # Later conversation - agent recalls preferences
    print("User: Show me some athletic shoes")
    async for chunk in shopping_agent.run_stream(
        task="Show me some athletic shoes"
    ):
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
    # Agent automatically searches memory and finds "prefers Nike brand"

# Batch import customer history
async def import_customer_history():
    """Import past purchase history as memories."""
    past_conversations = [
        {"role": "user", "content": "I bought those blue Nike Air Max last month"},
        {"role": "model", "content": "Great choice! How are they working out?"},
        {"role": "user", "content": "Love them! I wear size 10"},
        {"role": "model", "content": "Perfect, I'll remember your size."},
        {"role": "user", "content": "I usually shop during sales"}
    ]
    
    # VertexAI extracts facts like:
    # - "Customer bought Nike Air Max in blue"
    # - "Customer wears size 10"
    # - "Customer prefers shopping during sales"
    result = await memory.generate_memories_from_events(
        user_id="customer123",
        events=past_conversations
    )
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**What Happens Behind the Scenes:**

1. **First Message**: Agent uses `search_tool` to check existing memories (cache miss, fetches from VertexAI)
2. **Learning**: Agent uses `update_tool` to store "Customer prefers Nike brand"
3. **Subsequent Requests**: Agent uses `search_tool` again (cache hit, returns instantly)
4. **Personalization**: Recommendations automatically include Nike products

**Benefits:**
- Users don't repeat preferences
- Context persists across sessions
- Reduced API calls with intelligent caching
- Multi-user isolation via `user_id`

```python
# Disable caching completely
config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789",
    app_name="1234567890123456789",
    project_id="my-project-id",
    location="us-central1",
    user_id="user123",
    cache_enabled=False
)

# Short cache TTL (30 seconds)
config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789",
    app_name="1234567890123456789",
    project_id="my-project-id",
    location="us-central1",
    user_id="user123",
    cache_ttl_seconds=30
)

# Long cache TTL (1 hour)
config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789",
    app_name="1234567890123456789",
    project_id="my-project-id",
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

# VertexAI processes the conversation and extracts facts like:
# - "User is a software engineer at Google"
# - "User works with Kubernetes and Go"
# - "User works on cloud infrastructure"
# - "User contributes to open source projects on weekends"

# Wait for processing, then query the extracted memories
import asyncio
await asyncio.sleep(2)

results = await memory.query(query="user's work")
for mem in results.results:
    print(mem.content)
```

### Custom Client Configuration

```python
from vertexai import Client

# Create custom client with specific settings
client = Client(
    project="my-project-id",
    location="us-central1"
)

memory = VertexaiMemory(config=config, client=client)
```

### Multi-User Isolation

```python
# User 1's memories
user1_config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789",
    app_name="1234567890123456789",
    project_id="my-project-id",
    location="us-central1",
    user_id="user1"
)
user1_memory = VertexaiMemory(config=user1_config)

# User 2's memories (isolated from User 1)
user2_config = VertexaiMemoryConfig(
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789",
    app_name="1234567890123456789",
    project_id="my-project-id",
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
    project_id="my-project-id",
    location="us-central1",
    user_id="user123",
    app_name="1234567890123456789",
    api_resource_name="projects/my-project-id/locations/us-central1/reasoningEngines/1234567890123456789"
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

# Verify user_id and app_name match
print(f"Using user_id: {memory.user_id}")
print(f"Using app_name: {memory.app_name}")
```

### Cache Not Working

```python
# Check cache configuration
print(f"Cache enabled: {memory._cache_enabled}")
print(f"Cache TTL: {memory._cache_ttl_seconds}")

# Manually invalidate cache if needed
memory._invalidate_cache()
```
