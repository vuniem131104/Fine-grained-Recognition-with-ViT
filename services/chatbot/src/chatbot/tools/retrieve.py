import httpx 
from structlog import get_logger
import os 

logger = get_logger(__name__)

async def context_retrieve(client: httpx.AsyncClient, query: str) -> list[dict]:
    try:
        logger.info('Sending retrieval request to RAG service.', extra={'query': query})
        response = await client.post(
            os.getenv('RAG_URL') + '/query',
            json={"query": query}
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        logger.info(f'Retrieval successfully returned {len(results)} results.')
        return results
    except Exception as e:
        logger.exception("Error during retrieval.", extra={"error": str(e)})
        raise