from enum import Enum
from typing import Union
from pydantic import BaseModel, Field
from structlog import get_logger
from chatbot.llm import LiteLLMService, LiteLLMInput, CompletionMessage, Role
from .settings import AnswerGeneratorSetting
from .prompts import (
    ANSWER_GENERATOR_SYSTEM_PROMPT,
    ANSWER_GENERATOR_CLASSIFICATION_PROMPT,
    ANSWER_GENERATOR_RETRIEVAL_PROMPT,
    ANSWER_GENERATOR_DIRECT_ANSWER_PROMPT,
    ANSWER_GENERATOR_COMBINED_PROMPT,
)
import asyncio
from fastapi.encoders import jsonable_encoder
from chatbot.agents.state import ChatbotState
from chatbot.agents.tool_executor.tool_executor import ToolType

logger = get_logger(__name__)


class AnswerGeneratorInput(BaseModel):
    """Input for answer generation from tool executor"""
    original_query: str = Field(description="The original user query")
    rephrased_original_query: str = Field(description="The rephrased standalone version of the original query")
    tool_type: ToolType = Field(description="The type of tool that was executed")
    tool_result: Union[dict, list] = Field(description="The result from the tool execution")


class AnswerGeneratorOutput(BaseModel):
    """Output from answer generator"""
    answer: str = Field(description="The generated answer to the user's query")


class AnswerGeneratorService:
    """Service for generating answers based on tool execution results"""

    def __init__(self, litellm_service: LiteLLMService, settings: AnswerGeneratorSetting):
        self.litellm_service = litellm_service
        self.settings = settings

    async def process(self, inputs: AnswerGeneratorInput) -> AnswerGeneratorOutput:
        """
        Process tool results and generate a comprehensive answer

        Args:
            inputs: AnswerGeneratorInput containing query, tool type, and tool result

        Returns:
            AnswerGeneratorOutput with the generated answer
        """
        try:
            output = await self.litellm_service.process_async(
                inputs=self._build_llm_input(inputs)
            )
            return AnswerGeneratorOutput(answer=output.response.strip())

        except Exception as e:
            logger.exception(
                "Error when processing answer generation with litellm",
                extra={
                    "original_query": inputs.original_query,
                    "tool_type": inputs.tool_type,
                    "error": str(e),
                },
            )
            raise e

    @staticmethod
    def _format_result(result: Union[dict, list]) -> str:
        """
        Format tool result for inclusion in prompt

        Args:
            result: The tool result dictionary

        Returns:
            Formatted string representation of the result
        """
        import json

        return json.dumps(result, indent=2)

    async def gprocess(self, state: ChatbotState) -> dict:
        """Graph process: Generate answers for all tool results.

        Processes all tool results from the state and generates answers for each.
        This method is designed to be used directly as a LangGraph node.

        Args:
            state: ChatbotState containing original_query and tool_results

        Returns:
            Dict with final_response key to be merged back into state
        """
        try:
            tool_results = state.get("tool_results", [])
            original_query = state.get("original_query", "")
            rephrased_original_query = state.get("rephrased_original_query") or original_query

            if not tool_results:
                logger.warning("No tool results to process for answer generation")
                return {"final_response": "Could not process your query. Please try again."}

            logger.info(
                "Generating answers for tool results",
                extra={"tool_results_count": len(tool_results)},
            )

            # Generate answers for each tool result in parallel
            answer_tasks = [
                self._generate_single_answer(tr, original_query, rephrased_original_query)
                for tr in tool_results
            ]

            answers = await asyncio.gather(*answer_tasks, return_exceptions=False)

            # Filter out None answers
            answers = [a for a in answers if a is not None]

            # Combine all answers into final response
            if answers:
                final_response = "\n\n".join(answers)
            else:
                final_response = "Could not generate an answer. Please try again."

            logger.info(
                "Answers generated successfully",
                extra={
                    "total_results": len(tool_results),
                    "successful_answers": len(answers),
                },
            )

            return {"final_response": final_response}

        except Exception as e:
            logger.exception(
                "Error in answer generator gprocess",
                extra={"error": str(e)},
            )
            raise

    def _build_llm_input(self, inputs: AnswerGeneratorInput, history_text: str = "") -> LiteLLMInput:
        if inputs.tool_type == ToolType.CLASSIFICATION:
            user_prompt = ANSWER_GENERATOR_CLASSIFICATION_PROMPT.format(
                original_query=inputs.original_query,
                rephrased_original_query=inputs.rephrased_original_query,
                classification_result=self._format_result(inputs.tool_result),
                history_text=history_text,
            )
        elif inputs.tool_type == ToolType.RETRIEVAL:
            user_prompt = ANSWER_GENERATOR_RETRIEVAL_PROMPT.format(
                original_query=inputs.original_query,
                rephrased_original_query=inputs.rephrased_original_query,
                retrieval_result=self._format_result(inputs.tool_result),
                history_text=history_text,
            )
        else:
            user_prompt = ANSWER_GENERATOR_DIRECT_ANSWER_PROMPT.format(
                original_query=inputs.original_query,
                rephrased_original_query=inputs.rephrased_original_query,
                direct_answer_result=self._format_result(inputs.tool_result),
                history_text=history_text,
            )
        return LiteLLMInput(
            messages=[
                CompletionMessage(role=Role.SYSTEM, content=ANSWER_GENERATOR_SYSTEM_PROMPT),
                CompletionMessage(role=Role.USER, content=user_prompt),
            ],
            model=self.settings.model,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            n=self.settings.n,
            frequency_penalty=self.settings.frequency_penalty,
            max_completion_tokens=self.settings.max_completion_tokens,
            reasoning_effort=self.settings.reasoning_effort,
        )

    @staticmethod
    def _extract_all_sources(tool_results: list[dict]) -> str:
        """Collect deduped URLs from all retrieval tool results."""
        seen: set[str] = set()
        urls: list[str] = []
        for tr in tool_results:
            if tr.get("tool_type") != ToolType.RETRIEVAL:
                continue
            result = tr.get("result", [])
            if not isinstance(result, list):
                continue
            for item in result:
                url = item.get("url", "").strip()
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)
        if not urls:
            return ""
        lines = "\n".join(f"- {url}" for url in urls)
        return f"\n\n---\n**Sources:**\n{lines}"

    def _build_combined_prompt(
        self,
        original_query: str,
        rephrased_original_query: str,
        tool_results: list[dict],
        history_text: str = "",
    ) -> str:
        """Build a single user prompt combining all tool results."""
        import json
        sections: list[str] = []
        for tr in tool_results:
            tool_type_str = tr.get("tool_type", "")
            result = tr.get("result", {})
            formatted = json.dumps(result, indent=2)
            if tool_type_str == ToolType.CLASSIFICATION:
                sections.append(f"[Classification Result]\n{formatted}")
            elif tool_type_str == ToolType.RETRIEVAL:
                sections.append(f"[Retrieval Result for: {tr.get('query', '')}]\n{formatted}")
            else:
                sections.append(f"[Direct Answer]\n{formatted}")

        combined = "\n\n".join(sections)
        return ANSWER_GENERATOR_COMBINED_PROMPT.format(
            original_query=original_query,
            rephrased_original_query=rephrased_original_query,
            combined_results=combined,
            history_text=history_text,
        )

    async def stream_all_results(
        self,
        tool_results: list[dict],
        original_query: str,
        rephrased_original_query: str = "",
        history: list[dict] | None = None,
    ):
        """Stream one unified answer from all tool results, then append sources."""
        if not tool_results:
            yield "Could not process your query. Please try again."
            return

        history_text = ""
        if history:
            lines = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            history_text = "\n\nConversation History:\n" + "\n".join(lines)

        user_prompt = self._build_combined_prompt(
            original_query, rephrased_original_query or original_query, tool_results, history_text
        )
        llm_input = LiteLLMInput(
            messages=[
                CompletionMessage(role=Role.SYSTEM, content=ANSWER_GENERATOR_SYSTEM_PROMPT),
                CompletionMessage(role=Role.USER, content=user_prompt),
            ],
            model=self.settings.model,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            n=self.settings.n,
            frequency_penalty=self.settings.frequency_penalty,
            max_completion_tokens=self.settings.max_completion_tokens,
            reasoning_effort=self.settings.reasoning_effort,
        )
        async for chunk in self.litellm_service.stream_async(llm_input):
            yield chunk

        sources = self._extract_all_sources(tool_results)
        if sources:
            yield sources

    async def _generate_single_answer(
        self, tool_result: dict, original_query: str, rephrased_original_query: str = ""
    ) -> str | None:
        """Generate answer for a single tool result.

        Args:
            tool_result: Result from tool execution
            original_query: The original query
            rephrased_original_query: The rephrased standalone version of the query

        Returns:
            Generated answer or None if generation fails
        """
        try:
            tool_type_str = tool_result.get("tool_type")
            query = tool_result.get("query", "")

            logger.debug(
                "Generating answer for tool result",
                extra={"tool_type": tool_type_str, "query": query},
            )

            # Map string to enum
            try:
                tool_type = ToolType(tool_type_str)
            except ValueError:
                logger.warning(
                    "Invalid tool type for answer generator",
                    extra={"tool_type": tool_type_str},
                )
                return None

            # Create input and process
            generator_input = AnswerGeneratorInput(
                original_query=original_query,
                rephrased_original_query=rephrased_original_query or original_query,
                tool_type=tool_type,
                tool_result=tool_result.get("result", {}),
            )

            generator_output = await self.process(generator_input)

            logger.debug("Answer generated successfully", extra={"tool_type": tool_type_str})

            return generator_output.answer

        except Exception as e:
            logger.exception(
                "Error generating answer for tool result",
                extra={
                    "tool_type": tool_result.get("tool_type"),
                    "error": str(e),
                },
            )
            return None
