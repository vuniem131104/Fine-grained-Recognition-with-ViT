# Chatbot Service - HTTP API

This document describes the REST API for the chatbot service.

## Overview

The Chatbot API provides a `/chat` endpoint that processes user queries through a LangGraph workflow. The workflow includes:
1. **Query Enricher**: Enhances the query and classifies it into tool types
2. **Tool Executor**: Executes appropriate tools (classification, retrieval, direct answer)
3. **Answer Generator**: Generates human-friendly responses

## Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "chatbot-api"
}
```

### Chat

```
POST /chat
Content-Type: application/json
```

**Request:**
```json
{
  "query": "What species is this bird?",
  "img_b64": "optional-base64-encoded-image"
}
```

**Response:**
```json
{
  "response": "This appears to be a Black-capped Chickadee based on...",
  "query": "What species is this bird?"
}
```

### API Information

```
GET /
```

**Response:**
```json
{
  "name": "Chatbot API",
  "description": "API for the bird classification chatbot",
  "endpoints": {
    "health": "/health",
    "chat": "/chat (POST)",
    "docs": "/docs",
    "openapi": "/openapi.json"
  }
}
```

## Starting the Server

### Using Python

```bash
# Install dependencies
uv sync

# Run the server
python -m chatbot.api
```

The server will start on `http://0.0.0.0:8000`

### Using Docker

See the [Dockerfile](./Dockerfile) for containerized deployment.

## Example Usage

### Using cURL

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about birds with long beaks"
  }'
```

### Using Python

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/chat",
        json={"query": "What is this bird?"}
    )
    print(response.json())
```

### Using JavaScript/Fetch

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'Tell me about common songbirds'
  })
});

const data = await response.json();
console.log(data.response);
```

## Configuration

### Environment Variables

The API uses the following environment variables (currently hardcoded for development):

- `LITELLM_API_KEY`: API token for LiteLLM service
- `LITELLM_URL`: URL of the LiteLLM service (default: `http://localhost:8000/`)
- `LITELLM_MODEL`: LLM model to use (default: `gpt-4`)
- `LITELLM_EMBEDDING_MODEL`: Embedding model (default: `text-embedding-3-small`)

**TODO**: Migrate these to environment variables using `pydantic-settings`.

### Settings

Customize the workflow settings by modifying the initialization in [api.py](./src/chatbot/api.py):

```python
litellm_setting = LiteLLMSetting(
    url="http://localhost:8000/",
    token="your-api-token",
    model="gpt-4",
    embedding_model="text-embedding-3-small",
    ...
)
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Request/Response Models

### ChatRequest

```python
class ChatRequest(BaseModel):
    query: str  # Required: 1-5000 characters
    img_b64: Optional[str] = None  # Optional: Base64-encoded image
```

### ChatResponse

```python
class ChatResponse(BaseModel):
    response: str  # The chatbot's response
    query: str  # The original query
```

## Error Handling

### 503 Service Unavailable

```json
{
  "detail": "Chatbot service is not ready. Please try again later."
}
```

This occurs when the workflow hasn't been initialized yet.

### 500 Internal Server Error

```json
{
  "detail": "Error processing query: [error details]"
}
```

This occurs when there's an error during query processing.

## Logging

The API uses structured logging via `structlog`. All requests and responses are logged with:
- Request query and image presence
- Response length
- Processing errors with stack traces

## Frontend Integration

### Example: React Component

```typescript
import { useState } from 'react';

export function ChatBox() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const result = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      
      const data = await result.json();
      setResponse(data.response);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input 
        value={query} 
        onChange={(e) => setQuery(e.target.value)} 
        placeholder="Ask about birds..."
      />
      <button type="submit" disabled={loading}>Send</button>
      {response && <p>{response}</p>}
    </form>
  );
}
```

## Workflow

See [WORKFLOW.md](./WORKFLOW.md) for detailed information about the internal workflow architecture.

## Clean Architecture

See [CLEAN_ARCHITECTURE.md](./CLEAN_ARCHITECTURE.md) for details about the service layers and design patterns.
