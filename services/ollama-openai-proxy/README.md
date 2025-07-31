# Ollama OpenAI API Proxy

A FastAPI-based proxy service that translates OpenAI API calls to work with Ollama servers. This allows you to use
OpenAI-compatible libraries and tools with your local Ollama models.

## Features

- **OpenAI Compatibility**: Supports `/v1/chat/completions` endpoint with OpenAI-compatible request/response format
- **Native Ollama Support**: Direct proxy to Ollama's `/api/chat` endpoint for native streaming
- **Health Monitoring**: Built-in health check endpoint
- **Docker Support**: Containerized deployment with Docker Compose
- **Configurable**: Environment-based configuration for Ollama server URL and timeouts
- **Logging**: Comprehensive logging for debugging and monitoring

## Quick Start

### Using Docker Compose (Recommended)

1. Clone or download the project files
2. Run the service:
   ```bash
   docker-compose up -d
   ```
3. The proxy will be available at `http://localhost:5000`

### Manual Setup

1. Install dependencies:
   ```bash
   pip install -r requirements-ollama-openai-proxy.txt
   ```

2. Set environment variables (optional):
   ```bash
   export OLLAMA_BASE_URL=http://your-ollama-server:11434
   export REQUEST_TIMEOUT=300.0
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## API Endpoints

### OpenAI-Compatible Endpoints

- **POST** `/v1/chat/completions` - OpenAI-compatible chat completions
    - Accepts standard OpenAI request format
    - Returns OpenAI-compatible responses
    - Use with any OpenAI-compatible library

### Ollama Native Endpoints

- **POST** `/api/chat` - Direct proxy to Ollama's native chat API
    - Supports Ollama's native streaming format
    - Direct passthrough to Ollama server

### Utility Endpoints

- **GET** `/health` - Health check endpoint
- **GET** `/` - Service information and available endpoints
- **GET** `/docs` - Interactive API documentation (Swagger UI)

## Usage Examples

### With OpenAI Library

```python
from openai import OpenAI

client = OpenAI(
    api_key="fake-key",  # Not used but required
    base_url="http://localhost:5000/v1"
)

response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

### With curl (OpenAI format)

```bash
curl -X POST "http://localhost:5000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### With curl (Ollama native format)

```bash
curl -X POST "http://localhost:5000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

## Configuration

Configure the service using environment variables:

| Variable          | Default                           | Description                |
|-------------------|-----------------------------------|----------------------------|
| `OLLAMA_BASE_URL` | `http://your-ollama-server:11434` | URL of your Ollama server  |
| `REQUEST_TIMEOUT` | `300.0`                           | Request timeout in seconds |

### Docker Compose Configuration

Edit the `docker-compose.yml` file to change environment variables:

```yaml
environment:
  - OLLAMA_BASE_URL=http://your-ollama-server:11434
  - REQUEST_TIMEOUT=600.0
```

## Development

### Running in Development Mode

```bash
# Install dependencies
pip install -r requirements-ollama-openai-proxy.txt

# Run with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```

### Building Docker Image

```bash
docker build -t ollama-openai-proxy .
```

## Health Monitoring

The service includes health checks accessible at `/health`:

```bash
curl http://localhost:5000/health
```

Response:

```json
{
  "status": "healthy",
  "message": "Ollama OpenAI API Proxy is running",
  "ollama_url": "http://your-ollama-server:11434"
}
```

## License

This project is open source and available under the MIT License.