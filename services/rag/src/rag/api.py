from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import structlog
from contextlib import asynccontextmanager

from rag.service import ContextManagement

logger = structlog.get_logger(__name__)

context_manager = ContextManagement(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    qdrant_url=os.getenv("QDRANT_URL"),
    vector_size=int(os.getenv("QDRANT_VECTOR_SIZE")),
    top_k=int(os.getenv("RAG_TOP_K")),
    score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD"))
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await context_manager.initialize_collection()
        logger.info("Context manager initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize context manager: {e}")
    yield

app = FastAPI(title="RAG Service API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    # filter: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]

class IndexResponse(BaseModel):
    message: str
    chunks_indexed: int

class DeleteResponse(BaseModel):
    message: str

@app.post("/index", response_model=IndexResponse)
async def index_documents():
    try:
        logger.info("Starting indexing process...")
        chunks = context_manager.get_all_chunks()
        await context_manager.process_add_context(chunks)
        logger.info(f"Successfully indexed {len(chunks)} chunks.")
        return IndexResponse(
            message="Indexing completed successfully",
            chunks_indexed=len(chunks)
        )
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_context(request: QueryRequest):
    try:
        results = await context_manager.process_query(
            query=request.query,
            # filter=request.filter
        )
        return QueryResponse(results=results)
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collection", response_model=DeleteResponse)
async def delete_collection():
    try:
        logger.info("Starting collection deletion...")
        await context_manager.delete_collection()
        logger.info("Collection deleted successfully.")
        return DeleteResponse(message="Collection deleted successfully")
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}