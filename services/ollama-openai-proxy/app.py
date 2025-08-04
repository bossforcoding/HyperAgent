import os
import json
import logging
import requests

from openai import OpenAI
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, Response, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ollama OpenAI API Proxy",
    description="Proxy service that translates OpenAI API calls to Ollama server",
    version="1.0.0"
)

# Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300.0"))

# Initialize OpenAI client for Ollama compatibility
ollama_client = OpenAI(
    api_key="ollama",  # Ollama doesn't require a real key but OpenAI lib needs one
    base_url=f"{OLLAMA_BASE_URL}/v1"
)


@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    logger.info(f"Starting Ollama OpenAI API Proxy")
    logger.info(f"Forwarding requests to: {OLLAMA_BASE_URL}")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT}s")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler."""
    logger.info("Ollama OpenAI API Proxy shut down")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify both the proxy and Ollama server are running.

    This endpoint contacts the Ollama server to ensure it's accessible and responding.
    Returns healthy status only if Ollama responds correctly.

    Returns:
        dict: Status information about the proxy service and Ollama connectivity
    """
    logger.info("Health check endpoint called")

    try:
        # Contact Ollama server root endpoint to check if it's running
        response = requests.get(OLLAMA_BASE_URL, timeout=10)

        print("RESPONSE:", response.text)

        # Check if Ollama responds with expected message
        if response.status_code == 200 or "Ollama is running" in response.text:
            return {
                "status": "healthy",
                "message": "Ollama OpenAI API Proxy is running",
                "ollama_url": OLLAMA_BASE_URL,
                "ollama_status": "connected"
            }
        else:
            logger.warning(f"Ollama server responded with unexpected content: {response.text}")
            return {
                "status": "unhealthy",
                "message": "Ollama OpenAI API Proxy is running but Ollama server is not responding correctly",
                "ollama_url": OLLAMA_BASE_URL,
                "ollama_status": "error",
                "error": f"Unexpected response from Ollama: {response.text}"
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        return {
            "status": "unhealthy",
            "message": "Ollama OpenAI API Proxy is running but cannot reach Ollama server",
            "ollama_url": OLLAMA_BASE_URL,
            "ollama_status": "unreachable",
            "error": str(e)
        }


@app.post("/api/chat")
async def proxy_native_chat(request: Request):
    """
    Proxy endpoint for Ollama's native /api/chat endpoint.

    This endpoint forwards requests directly to Ollama's native API without
    any format conversion. Useful for clients that want to use Ollama's
    native streaming format.

    Args:
        request: The incoming HTTP request

    Returns:
        Response: Raw response from Ollama server
    """
    body = await request.body()
    headers = dict(request.headers)

    # Remove problematic headers that might cause conflicts
    headers_to_remove = ["host", "content-length", "transfer-encoding", "connection", "content-encoding"]
    for header in headers_to_remove:
        headers.pop(header, None)

    logger.info("Proxying request to Ollama native /api/chat endpoint")
    logger.debug(f"Request headers: {headers}")
    logger.debug(f"Request body: {body.decode('utf-8')}")

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=body,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json")
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error forwarding request to Ollama: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to Ollama server: {str(e)}"
        )


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    This endpoint accepts OpenAI-formatted requests and translates them
    to work with Ollama's API. It maintains compatibility with OpenAI's
    response format.

    Args:
        request: The incoming HTTP request with OpenAI-formatted payload

    Returns:
        JSONResponse: OpenAI-compatible response from Ollama
    """
    try:
        # Parse the incoming OpenAI-formatted request
        payload = await request.json()

        logger.info("Processing OpenAI chat completions request")
        logger.debug(f"Received payload: {json.dumps(payload, indent=2)}")

        # Validate required fields
        if "model" not in payload:
            raise HTTPException(status_code=400, detail="Model field is required")
        if "messages" not in payload:
            raise HTTPException(status_code=400, detail="Messages field is required")

        # Use OpenAI library to make the request to Ollama
        # This handles the format translation automatically
        model = payload.pop("model")
        messages = payload.pop("messages")

        response = ollama_client.chat.completions.create(
            model=model,
            messages=messages,
            **payload  # Pass through any additional parameters
        )

        # Convert response to JSON format
        response_dict = json.loads(response.model_dump_json())
        logger.debug(f"Ollama response: {json.dumps(response_dict, indent=2)}")

        return JSONResponse(content=response_dict)

    except json.JSONDecodeError:
        logger.error("Invalid JSON payload received")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint providing basic information about the proxy.

    Returns:
        dict: Basic information about the service and available endpoints
    """
    return {
        "service": "Ollama OpenAI API Proxy",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "openai_chat": "/v1/chat/completions",
            "ollama_chat": "/api/chat"
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
