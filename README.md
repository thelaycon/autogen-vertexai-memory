# autogen-vertexai-memory

VertexAI Memory integration for Autogen agents. Store and retrieve agent memories using Google Cloud's VertexAI Memory service.

## Features

- 🧠 Persistent memory storage in VertexAI
- 🔍 Semantic search for memory retrieval
- 🔄 Automatic chat context updates
- ⚡ Async/await support
- 🎯 User-scoped memory isolation

## Installation

```bash
pip install autogen-vertexai-memory
```

## Prerequisites

- Google Cloud Project with VertexAI API enabled
- Authentication configured (Application Default Credentials recommended)
- VertexAI Memory resource created

```bash
gcloud auth application-default login
```

## Quick Start

```python
from autogen_vertexai_memory import VertexaiMemory, VertexaiMemoryConfig
from autogen_core.memory import MemoryContent, MemoryMimeType

# Configure
config = VertexaiMemoryConfig(
    api_resource_name="projects/xxxxxxxxxx/locations/us-central1/reasoningEngines/xxxxxxxxxxxxxxxx",
    project_id="YOUR_PROJECT_ID",
    location="us-central1",
    user_id="user123"
)

memory = VertexaiMemory(config=config)

# Add memory
await memory.add(
    content=MemoryContent(
        content="User prefers Python",
        mime_type=MemoryMimeType.TEXT
    )
)

# Query with semantic search
results = await memory.query(query="programming language preference")

# Get all memories
all_memories = await memory.query(query="")
```

## API Reference

### VertexaiMemoryConfig

```python
VertexaiMemoryConfig(
    api_resource_name: str,  # Full VertexAI memory resource name
    project_id: str,         # GCP project ID
    location: str,           # GCP region (e.g., "us-central1")
    user_id: str            # User identifier for memory isolation
)
```

### VertexaiMemory

```python
VertexaiMemory(
    config: VertexaiMemoryConfig | None = None,
    client: Client | None = None
)
```

**Methods:**

- `add(content, cancellation_token=None)` - Add a memory
- `query(query="", cancellation_token=None, **kwargs)` - Query memories (empty query returns all)
- `update_context(model_context)` - Update chat context with memories
- `clear()` - Delete all memories (irreversible)
- `close()` - Cleanup resources

## Examples

### Update Chat Context

```python
from autogen_core.model_context import ChatCompletionContext

context = ChatCompletionContext()
result = await memory.update_context(context)
```

### Custom Client

```python
from vertexai import Client

client = Client(project="my-project", location="us-central1")
memory = VertexaiMemory(config=config, client=client)
```

## Development

```bash
git clone https://github.com/thelaycon/autogen-vertexai-memory.git
cd autogen-vertexai-memory
poetry install
poetry run pytest
```

## Contributing

Contributions welcome! Please submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📫 [GitHub Issues](https://github.com/thelaycon/autogen-vertexai-memory/issues)
- 💬 [GitHub Discussions](https://github.com/thelaycon/autogen-vertexai-memory/discussions)

---

Made with ❤️ for the Autogen community