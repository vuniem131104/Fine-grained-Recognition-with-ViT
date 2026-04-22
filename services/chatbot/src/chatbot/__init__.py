from chatbot.api import app
import os
import uvicorn

def main() -> None:
    """Run the FastAPI application."""
    
    uvicorn.run(
        "chatbot:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=int(os.getenv("WORKERS", 1)),
        reload=True,
    )