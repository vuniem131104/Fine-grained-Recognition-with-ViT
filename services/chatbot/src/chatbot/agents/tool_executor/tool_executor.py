from chatbot.tools import classify, context_retrieve
from pydantic import BaseModel
from structlog import get_logger
import httpx
from fastapi.encoders import jsonable_encoder
import asyncio
from typing import Optional, Union
from chatbot.agents.state import ChatbotState
from enum import Enum

logger = get_logger(__name__)

class ToolType(str, Enum):
    """Tool type classification for enriched queries"""
    CLASSIFICATION = "classification"
    RETRIEVAL = "retrieval"
    DIRECT_ANSWER = "direct_answer"

class ToolExecutorInput(BaseModel):
    query: str
    tool_type: ToolType
    img_b64: Optional[str] = None
    
class ToolExecutorOutput(BaseModel):
    tool_type: ToolType
    result: Union[dict, list] = {}
    
class ToolExecutorService:
    def __init__( self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        
    async def process(self, inputs: ToolExecutorInput) -> ToolExecutorOutput:
        if inputs.tool_type == ToolType.CLASSIFICATION:
            if not inputs.img_b64:
                raise ValueError("Image data is required for classification.")
            result = await classify(self.http_client, inputs.img_b64)
        elif inputs.tool_type == ToolType.RETRIEVAL:
            result = await context_retrieve(self.http_client, inputs.query)
        elif inputs.tool_type == ToolType.DIRECT_ANSWER:
            result = {}
        else:
            raise ValueError(f"Unsupported tool type: {inputs.tool_type}")
        
        return ToolExecutorOutput(tool_type=inputs.tool_type, result=result)

    async def gprocess(self, state: ChatbotState) -> dict:
        """Graph process: Execute tools in parallel for all enriched queries.

        Processes all enriched queries from the state by executing their
        corresponding tools in parallel using asyncio.gather.
        This method is designed to be used directly as a LangGraph node.

        Args:
            state: ChatbotState containing enriched_queries

        Returns:
            Dict with tool_results key to be merged back into state
        """
        try:
            enriched_queries = state.get("enriched_queries", [])

            if not enriched_queries:
                logger.warning("No enriched queries to execute")
                return {"tool_results": []}

            logger.info(
                "Starting parallel tool execution",
                extra={"query_count": len(enriched_queries)},
            )

            executor_tasks = [
                self._execute_single_tool(enriched_query=eq, img_b64=state.get("img_b64"))
                for eq in enriched_queries
            ]

            tool_results = await asyncio.gather(
                *executor_tasks, return_exceptions=False
            )

            tool_results = [r for r in tool_results if r is not None]

            logger.info(
                "Tool execution completed",
                extra={
                    "total_queries": len(enriched_queries),
                    "successful_results": len(tool_results),
                },
            )

            return {"tool_results": tool_results}

        except Exception as e:
            logger.exception(
                "Error in tool executor gprocess",
                extra={"error": str(e)},
            )
            raise

    async def _execute_single_tool(
        self,
        enriched_query: dict,
        img_b64: Optional[str] = None,
    ) -> Optional[dict]:
        """Execute a single tool for enriched query.

        Args:
            enriched_query: Dict with 'query' and 'tool_type' keys
            original_query: The original query (for reference)

        Returns:
            Tool result dict or None if execution fails
        """
        try:
            tool_type = enriched_query.get("tool_type")
            query = enriched_query.get("query")

            logger.debug(
                "Executing tool",
                extra={"tool_type": tool_type, "query": query},
            )
            
            executor_input = ToolExecutorInput(
                query=query,
                tool_type=tool_type,
                img_b64=img_b64,
            )
                
            executor_output = await self.process(executor_input)

            result = {
                "query": query,
                "tool_type": tool_type,
                "result": executor_output.result,
            }

            logger.debug(
                "Tool execution successful",
                extra={"tool_type": tool_type, "query": query},
            )

            return result

        except Exception as e:
            logger.exception(
                "Error executing tool",
                extra={
                    "tool_type": enriched_query.get("tool_type"),
                    "query": enriched_query.get("query"),
                    "error": str(e),
                },
            )
            return None