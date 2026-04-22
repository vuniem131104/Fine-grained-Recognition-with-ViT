from annotated_types import doc
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    Modifier, 
    VectorParams, 
    SparseVectorParams,
    PointStruct,
    Prefetch,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchValue
)
from openai import AsyncAzureOpenAI
from structlog import get_logger
from uuid import uuid4
import psycopg2
import os 
from rag.chunk import ChunkerInput, ChunkerService
from tqdm import tqdm


logger = get_logger(__name__)

class ContextManagement:
    def __init__(self, collection_name: str, qdrant_url: str, vector_size: int = 1536, top_k: int = 5, score_threshold: float = 0.5):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.client = AsyncQdrantClient(url=qdrant_url, prefer_grpc=True, timeout=10)
        self.chunker_service = ChunkerService()
        self.embedding_client = AsyncAzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            azure_endpoint=os.getenv("LLM_API_BASE")
        )
        
    async def embed_content(self, content: str) -> list[float]:
        try:
            response = await self.embedding_client.embeddings.create(
                input=[content],
                model=os.getenv("EMBEDDING_MODEL"),
                dimensions=self.vector_size
            )
            embedding_vector = response.data[0].embedding
            return embedding_vector
        except Exception as e:
            logger.exception(
                f"Error during embedding creation: {e}",
                extra={"content": content}
            )
            return [0.0] * self.vector_size
        
    def get_all_records(self):
        try:
            connection = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                database=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                port=os.getenv("POSTGRES_PORT")
            )
            cursor = connection.cursor()
            cursor.execute("SELECT url, species, content FROM wiki_documents;")
            records = cursor.fetchall()
            return records
        except Exception as e:
            logger.error(f"Error fetching records from PostgreSQL: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
    def get_all_chunks(self):
        records = self.get_all_records()
        chunks = []
        for url, species, content in records:
            chunker_input = ChunkerInput(contents=content)
            chunker_output = self.chunker_service.process(chunker_input)
            chunks.extend(
                [
                    {
                        "text": chunk,
                        "species": species.lower(),
                        "url": url
                    }
                    for chunk in chunker_output.chunks
                ]
            )
        return chunks


    async def initialize_collection(self):
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            distance=Distance.COSINE,
                            size=self.vector_size,
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            modifier=Modifier.IDF,
                        )
                    }
                )
                logger.info(f"Collection '{self.collection_name}' created.")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
        
    async def delete_collection(self):
        try:
            await self.client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted.")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    async def process_add_context(self, chunks: list[dict]):
        try:
            points = []
            for chunk in tqdm(chunks, desc="Processing chunks"):
                dense_vector = await self.embed_content(chunk['text'])
                points.append(
                    PointStruct(
                        id=uuid4().hex,
                        vector={
                            "dense": dense_vector,
                            "sparse": Document(
                                text=chunk['text'],
                                model="Qdrant/bm25",
                            ),
                        },
                        payload=chunk,
                    )
                )
            logger.info(f"Adding {len(points)} chunks to collection '{self.collection_name}'...")
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Finished adding chunks to collection '{self.collection_name}'.")

        except Exception as e:
            logger.error(f"Error adding context: {e}")
            raise
        
    async def process_query(self, query: str) -> list[dict]:
        try:
            # query_filter = Filter(
            #     must=[
            #         FieldCondition(
            #             key="species",
            #             match=MatchValue(value=filter.get("species").lower())
            #         )
            #     ]
            # ) if filter else None
            query_vector = await self.embed_content(query)
            response = await self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=Document(
                            text=query,
                            model="Qdrant/bm25",
                        ),
                        using="sparse",
                        limit=self.top_k,
                    ),
                    Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=self.top_k,
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=self.top_k,
                score_threshold=self.score_threshold,
                # query_filter=query_filter,
                with_payload=True
            )
            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "url": point.payload.get("url", ""),
                    "species": point.payload.get("species", "Unknown")
                }
                for point in response.points
            ]
        except Exception as e:
            logger.error(f"Error querying context: {e}")
            raise
