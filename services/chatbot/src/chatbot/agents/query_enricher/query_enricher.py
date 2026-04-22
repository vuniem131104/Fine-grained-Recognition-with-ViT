import json
from enum import Enum
from pydantic import BaseModel, Field
from structlog import get_logger
from chatbot.llm import LiteLLMService, LiteLLMInput, CompletionMessage, Role
from .settings import QueryEnricherSetting
from .prompts import QUERY_ENRICHER_SYSTEM_PROMPT, QUERY_ENRICHER_USER_PROMPT
from chatbot.agents.state import ChatbotState
from fastapi.encoders import jsonable_encoder
from chatbot.agents.tool_executor.tool_executor import ToolType

logger = get_logger(__name__)


class EnrichedQuery(BaseModel):
    """Enriched query with associated tool type"""
    query: str
    tool_type: ToolType


class QueryEnricherInput(BaseModel):
    original_query: str

class QueryEnricherOutput(BaseModel):
    rephrased_original_query: str = Field(
        description="Original query rephrased for clarity, taking conversation history into account"
    )
    enriched_queries: list[EnrichedQuery] = Field(
        description="List of enriched queries with their tool type classifications"
    )
    

class QueryEnricherService:
    def __init__(self, litellm_service: LiteLLMService, settings: QueryEnricherSetting):
        self.litellm_service = litellm_service
        self.settings = settings

    async def process(self, inputs: QueryEnricherInput, history: list[dict] | None = None) -> QueryEnricherOutput:
        try:
            history_text = ""
            if history:
                lines = []
                for msg in history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    lines.append(f"{role}: {msg['content']}")
                history_text = "\n\nConversation History:\n" + "\n".join(lines)
                
            user_prompt = QUERY_ENRICHER_USER_PROMPT.format(
                original_query=inputs.original_query,
                history_text=history_text,
            )

            output = await self.litellm_service.process_async(
                inputs=LiteLLMInput(
                    messages=[
                        CompletionMessage(
                            role=Role.SYSTEM,
                            content=QUERY_ENRICHER_SYSTEM_PROMPT
                        ),
                        CompletionMessage(
                            role=Role.USER,
                            content=user_prompt
                        )
                    ],
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    n=self.settings.n,
                    frequency_penalty=self.settings.frequency_penalty,
                    max_completion_tokens=self.settings.max_completion_tokens,
                    reasoning_effort=self.settings.reasoning_effort,
                )
            )
            
            response_text = output.response
            try:
                parsed_response = json.loads(response_text)
                enriched_queries_data = parsed_response.get("enriched_queries", [])
                rephrased = parsed_response.get("rephrased_original_query") or inputs.original_query
                logger.info(
                    "Query enriched successfully",
                    extra={
                        "original_query": inputs.original_query,
                        "rephrased_original_query": rephrased,
                        "enriched_queries_data": enriched_queries_data,
                    }
                )
                enriched_queries = []
                for item in enriched_queries_data:
                    if isinstance(item, dict) and "query" in item and "tool_type" in item:
                        try:
                            tool_type = ToolType(item["tool_type"])
                            enriched_queries.append(
                                EnrichedQuery(query=item["query"], tool_type=tool_type)
                            )
                        except ValueError:
                            logger.warning(
                                "Invalid tool_type value, defaulting to direct_answer",
                                extra={
                                    "original_query": inputs.original_query,
                                    "provided_tool_type": item.get("tool_type"),
                                    "query": item.get("query"),
                                }
                            )
                            enriched_queries.append(
                                EnrichedQuery(
                                    query=item["query"],
                                    tool_type=ToolType.DIRECT_ANSWER
                                )
                            )
                    elif isinstance(item, str):
                        enriched_queries.append(
                            EnrichedQuery(query=item, tool_type=ToolType.DIRECT_ANSWER)
                        )

                if not enriched_queries:
                    logger.warning(
                        "No enriched queries extracted, using rephrased query",
                        extra={"original_query": inputs.original_query}
                    )
                    enriched_queries = [
                        EnrichedQuery(query=rephrased, tool_type=ToolType.DIRECT_ANSWER)
                    ]

            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse LLM response as JSON, falling back to original query",
                    extra={"original_query": inputs.original_query, "response": response_text}
                )
                rephrased = inputs.original_query
                enriched_queries = [
                    EnrichedQuery(query=inputs.original_query, tool_type=ToolType.DIRECT_ANSWER)
                ]

            return QueryEnricherOutput(
                rephrased_original_query=rephrased,
                enriched_queries=enriched_queries,
            )
        except Exception as e:
            logger.exception(
                "Error when processing query enrichment with litellm",
                extra={
                    "original_query": inputs.original_query,
                    "error": str(e),
                } 
            )
            raise e
        
    async def gprocess(self, state: ChatbotState) -> dict:
        original_query = state.get("original_query", "")
        has_image = bool(state.get("img_b64"))
        history: list[dict] = state.get("conversation_history") or []

        if has_image and not original_query.strip():
            return {
                "enriched_queries": [
                    {"query": "Identify and classify the bird species in this image", "tool_type": ToolType.CLASSIFICATION}
                ]
            }

        input_data = QueryEnricherInput(original_query=original_query)
        output = await self.process(input_data, history=history)
        enriched = list(output.enriched_queries)

        if has_image and not any(eq.tool_type == ToolType.CLASSIFICATION for eq in enriched):
            enriched.insert(0, EnrichedQuery(
                query="Identify and classify the bird species in this image",
                tool_type=ToolType.CLASSIFICATION,
            ))

        return {
            "enriched_queries": jsonable_encoder(enriched),
            "rephrased_original_query": output.rephrased_original_query,
        }
