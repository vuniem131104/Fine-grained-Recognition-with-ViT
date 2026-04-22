import uuid
from langgraph.graph import StateGraph, START, END
from structlog import get_logger

from chatbot.llm import LiteLLMService
from chatbot.agents.state import ChatbotState
from chatbot.agents.query_enricher import QueryEnricherService
from chatbot.agents.tool_executor import ToolExecutorService
from chatbot.agents.answer_generator import AnswerGeneratorService
from chatbot.agents.memory_management import MemoryManagementService
import httpx
from pathlib import Path
from typing import Optional
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from chatbot.agents.query_enricher.settings import QueryEnricherSetting
from chatbot.agents.answer_generator.settings import AnswerGeneratorSetting
from chatbot.llm.settings import LiteLLMSetting

logger = get_logger(__name__)

class ChatbotSettings(BaseSettings):
    """Automatically load settings from YAML and environment variables.
    
    Priority (highest to lowest):
    1. Environment variables
    2. YAML file (settings.yaml)
    3. .env file
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    query_enricher: QueryEnricherSetting
    answer_generator: AnswerGeneratorSetting
    litellm: LiteLLMSetting

    def __init__(self, **data):
        """Initialize settings, auto-loading from YAML if no data provided."""
        if not data:
            yaml_path = self._find_settings_yaml()
            if yaml_path:
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f) or {}
        
        super().__init__(**data)
    
    @staticmethod
    def _find_settings_yaml() -> Optional[Path]:
        """Find settings.yaml in common locations.
        
        Searches in order:
        1. Current working directory
        2. Same directory as this file
        
        Returns:
            Path to settings.yaml if found, None otherwise
        """
        search_paths = [
            Path.cwd() / "settings.yaml",
            Path(__file__).parent / "settings.yaml",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None



class ChatbotService:
    def __init__(
        self,
        settings: ChatbotSettings,
        http_client: httpx.AsyncClient,
    ):
        """Initialize the chatbot service with required services.

        Args:
            settings: Chatbot settings loaded from YAML/env
            http_client: Async HTTP client for tool execution
        """
        self.litellm_service = LiteLLMService(
            litellm_setting=settings.litellm
        )
        self.query_enricher = QueryEnricherService(
            self.litellm_service, settings.query_enricher
        )
        self.tool_executor = ToolExecutorService(http_client)
        self.answer_generator = AnswerGeneratorService(
            self.litellm_service, settings.answer_generator
        )
        self.memory_management = MemoryManagementService()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph flow.

        The  is a simple linear pipeline:
        START → query_enricher → tool_executor → answer_generator → END

        Returns:
            Compiled StateGraph
        """
        graph = StateGraph(ChatbotState)

        graph.add_node("memory_management", self.memory_management.gprocess)
        graph.add_node("query_enricher", self.query_enricher.gprocess)
        graph.add_node("tool_executor", self.tool_executor.gprocess)
        graph.add_node("answer_generator", self.answer_generator.gprocess)

        graph.add_edge(START, "memory_management")
        graph.add_edge("memory_management", "query_enricher")
        graph.add_edge("query_enricher", "tool_executor")
        graph.add_edge("tool_executor", "answer_generator")
        graph.add_edge("answer_generator", END)

        return graph.compile()

    async def stream(
        self,
        query: str,
        img_b64: Optional[str] = None,
        user_id: Optional[int] = None,
        conversation_id: Optional[uuid.UUID] = None,
    ):
        """Run graph (memory + query enricher + tool executor), then stream answer tokens.

        Yields text chunks, then finally yields a sentinel dict:
          {"__meta__": True, "conversation_id": <int|None>}
        """
        initial_state: ChatbotState = {
            "original_query": query,
            "img_b64": img_b64,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "conversation_history": None,
            "enriched_queries": None,
            "tool_results": None,
            "final_response": None,
            "rephrased_original_query": None,
        }

        state = await self.graph.ainvoke(initial_state)
        tool_results = state.get("tool_results", [])
        original_query = state.get("original_query", query)
        rephrased_original_query = state.get("rephrased_original_query") or original_query
        history = state.get("conversation_history") or []

        async for chunk in self.answer_generator.stream_all_results(
            tool_results, original_query, rephrased_original_query=rephrased_original_query, history=history
        ):
            yield chunk

    async def process(self, query: str, img_b64: Optional[str] = None) -> str:
        """Run the graph with a user query.

        Args:
            query: The user's input query

        Returns:
            The final response from the chatbot
        """
        try:
            logger.info("Starting invocation", extra={"query": query})

            initial_state: ChatbotState = {
                "original_query": query,
                "img_b64": img_b64,
            }

            result = await self.graph.ainvoke(initial_state)

            final_response = result.get("final_response", "No response generated")

            logger.info(
                "Invocation completed",
                extra={"query": query, "has_response": bool(final_response)},
            )

            return final_response

        except Exception as e:
            logger.exception("Invocation error", extra={"error": str(e)})
            raise
