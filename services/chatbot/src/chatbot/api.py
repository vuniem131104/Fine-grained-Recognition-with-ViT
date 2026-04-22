import asyncio
import uuid
import httpx
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from structlog import get_logger

from chatbot.service import ChatbotSettings, ChatbotService

logger = get_logger(__name__)


class ChatRequest(BaseModel):
    query: str = Field("", description="The user's query", max_length=5000)
    img_b64: Optional[str] = Field(None, description="Optional base64-encoded image for classification")
    user_id: Optional[int] = Field(None, description="Authenticated user ID for conversation history")
    conversation_id: Optional[uuid.UUID] = Field(None, description="Existing conversation UUID to continue", examples=[None])

    @model_validator(mode="after")
    def require_query_or_image(self) -> "ChatRequest":
        if not self.query.strip() and not self.img_b64:
            raise ValueError("Either query or img_b64 must be provided")
        return self


class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's response")


_chatbot_service: Optional[ChatbotService] = None
_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chatbot_service, _http_client

    try:
        settings = ChatbotSettings()

        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(150.0, connect=15.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=40),
        )

        _chatbot_service = ChatbotService(
            settings=settings,
            http_client=_http_client,
        )

        logger.info("Chatbot service initialized successfully")

        yield

    except Exception as e:
        logger.exception("Failed to initialize chatbot service", extra={"error": str(e)})
        raise

    finally:
        logger.info("Shutting down chatbot API")
        if _http_client:
            await _http_client.aclose()


app = FastAPI(
    title="Chatbot API",
    description="API for the bird classification chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service_ready": _chatbot_service is not None}


@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    if not _chatbot_service:
        logger.error("Chatbot service not initialized")
        raise HTTPException(status_code=503, detail="Chatbot service is not ready.")

    logger.info(
        "Received chat request",
        extra={
            "query": request.query,
            "has_image": bool(request.img_b64),
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
        },
    )

    collected_chunks: list[str] = []

    async def event_stream():
        try:
            async for chunk in _chatbot_service.stream(
                request.query,
                img_b64=request.img_b64,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
            ):
                collected_chunks.append(chunk)
                yield f"data: {chunk.replace(chr(10), chr(92) + 'n')}\n\n"

            yield "data: [DONE]\n\n"

            if request.user_id:
                await asyncio.to_thread(
                    _chatbot_service.memory_management._save_message,
                    request.user_id,
                    request.conversation_id,
                    request.query,
                    "".join(collected_chunks),
                )
        except Exception as e:
            logger.exception("Error streaming chat response", extra={"error": str(e)})
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
