# autogen-vertexai-memory

VertexAI Memory integration for Autogen agents. This package provides a Memory implementation that enables Autogen agents to store and retrieve memories using Google Cloud's VertexAI Memory service.

## Features

- 🧠 **Persistent Memory**: Store agent memories in VertexAI for long-term retention
- 🔍 **Semantic Search**: Retrieve relevant memories using similarity search
- 🔄 **Context Management**: Automatically update chat context with relevant memories
- ⚡ **Async Support**: Built with async/await for efficient operations
- 🎯 **User Scoping**: Isolate memories per user for multi-tenant applications

## Installation

```bash
pip install autogen-vertexai-memory
```

Or with poetry:

```bash
poetry add autogen-vertexai-memory
```

## Prerequisites

1. **Google Cloud Project**: You need a GCP project with VertexAI API enabled
2. **Authentication**: Set up authentication using one of these methods:
   - Application Default Credentials (recommended for local development)
   - Service Account Key
   - Workload Identity (for GKE)

```bash
# Set up Application Default Credentials
gcloud auth application-default login
```

3. **VertexAI Memory Resource**: Create a memory resource in your GCP project

## Quick Start

```python
from autogen_vertexai_memory import VertexaiMemory
from autogen_core.memory import MemoryContent, MemoryMimeType

# Initialize the memory
memory = VertexaiMemory(
    api_resource_name="projects/YOUR_PROJECT/locations/us-central1/memories/YOUR_MEMORY",
    project_id="YOUR_PROJECT_ID",
    location="us-central1"
)

# Add a memory
await memory.add(
    content=MemoryContent(
        content="The user prefers Python over JavaScript",
        mime_type=MemoryMimeType.TEXT
    ),
    user_id="user123"
)

# Query memories
results = await memory.query(
    query="What programming language does the user prefer?",
    user_id="user123"
)

for result in results.results:
    print(result.content)
```

## API Reference

### `VertexaiMemory`

#### Constructor

```python
VertexaiMemory(
    api_resource_name: str,
    project_id: str,
    location: str,
    client: Client | None = None
)
```

**Parameters:**
- `api_resource_name`: Full resource name of the VertexAI memory
- `project_id`: Google Cloud project ID
- `location`: GCP region (e.g., "us-central1")
- `client`: Optional pre-configured VertexAI client

#### Methods

##### `add(content, user_id, cancellation_token=None)`

Add a new memory.

```python
await memory.add(
    content=MemoryContent(content="Important fact", mime_type=MemoryMimeType.TEXT),
    user_id="user123"
)
```

##### `query(query="", user_id="", cancellation_token=None, **kwargs)`

Query memories. If query is empty, returns all memories for the user.

```python
# Search with query
results = await memory.query(query="user preferences", user_id="user123")

# Get all memories
all_memories = await memory.query(user_id="user123")
```

##### `update_context(model_context)`

Update chat context with relevant memories.

```python
from autogen_core.model_context import ChatCompletionContext

context = ChatCompletionContext()
await memory.update_context(context)
```

##### `clear()`

Clear all memories (deletes the resource).

```python
await memory.clear()
```

##### `close()`

Cleanup resources.

```python
await memory.close()
```

## Advanced Usage

### Custom Client Configuration

```python
from vertexai import Client

# Create custom client with specific settings
client = Client(
    project="my-project",
    location="us-central1",
    # Additional client configurations...
)

# Use with VertexaiMemory
memory = VertexaiMemory(
    api_resource_name="...",
    project_id="my-project",
    location="us-central1",
    client=client
)
```

### Batch Operations

```python
# Add multiple memories
memories = [
    ("User likes cats", "user123"),
    ("User prefers dark mode", "user123"),
    ("User is from New York", "user123")
]

for content, user_id in memories:
    await memory.add(
        content=MemoryContent(content=content, mime_type=MemoryMimeType.TEXT),
        user_id=user_id
    )
```

### Error Handling

```python
from google.api_core.exceptions import GoogleAPIError

try:
    await memory.add(content=..., user_id="user123")
except GoogleAPIError as e:
    print(f"Error adding memory: {e}")
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/autogen-vertexai-memory.git
cd autogen-vertexai-memory

# Install dependencies
poetry install

# Run tests
poetry run pytest
```

### Running Tests

```bash
# Run all tests
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=autogen_vertexai_memory --cov-report=html

# Run specific test
poetry run pytest tests/test_vertexai_memory.py::TestAdd -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for use with [Autogen](https://github.com/microsoft/autogen)
- Powered by [Google Cloud VertexAI](https://cloud.google.com/vertex-ai)

## Support

- 📫 Report issues on [GitHub Issues](https://github.com/thelaycon/autogen-vertexai-memory/issues)
- 💬 Discuss on [GitHub Discussions](https://github.com/thelaycon/autogen-vertexai-memory/discussions)
- 📖 Read the [documentation](https://github.com/laycon/autogen-vertexai-memory/wiki)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes.

---

Made with ❤️ for the Autogen community