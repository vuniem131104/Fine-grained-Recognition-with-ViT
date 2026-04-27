import asyncio
import json
import os
import uuid
from enum import Enum
from typing import Optional, Union

import httpx
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import BaseModel
from structlog import get_logger

from chatbot.agents.state import ChatbotState
from chatbot.tools import context_retrieve

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
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client

        self.bootstrap = os.getenv("REDPANDA_BOOTSTRAP_SERVERS", "redpanda:29092")
        self.requests_topic = os.getenv("REDPANDA_REQUESTS_TOPIC", "inference-requests")
        self.results_topic = os.getenv("REDPANDA_RESULTS_TOPIC", "inference-results")
        self.request_timeout = float(os.getenv("INFERENCE_TIMEOUT", "30"))

        self._producer: Optional[AIOKafkaProducer] = None
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._pending: dict[str, asyncio.Future] = {}

    async def start(self) -> None:
        """Start Kafka producer + reply consumer. Call once at app startup."""
        self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap)
        await self._producer.start()

        self._consumer = AIOKafkaConsumer(
            self.results_topic,
            bootstrap_servers=self.bootstrap,
            group_id=f"chatbot-{uuid.uuid4()}",
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        await self._consumer.start()
        self._consumer_task = asyncio.create_task(self._consume_results())
        logger.info(
            "ToolExecutor Kafka started",
            extra={
                "bootstrap": self.bootstrap,
                "requests_topic": self.requests_topic,
                "results_topic": self.results_topic,
            },
        )

    async def stop(self) -> None:
        """Stop Kafka clients gracefully."""
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            await self._producer.stop()

        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    async def _consume_results(self) -> None:
        """Background task: route reply messages to the matching pending future."""
        try:
            async for msg in self._consumer:
                try:
                    payload = json.loads(msg.value)
                    request_id = payload.get("request_id")
                    if not request_id:
                        continue
                    fut = self._pending.pop(request_id, None)
                    if fut and not fut.done():
                        if "error" in payload:
                            fut.set_exception(RuntimeError(payload["error"]))
                        else:
                            fut.set_result(payload.get("result", {}))
                except Exception:
                    logger.exception("Failed to handle inference result message")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Result consumer loop crashed")

    async def _classify_via_stream(self, img_b64: str) -> dict:
        """Produce an inference request and await the matching reply."""
        if not self._producer:
            raise RuntimeError("ToolExecutorService not started")

        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future

        message = json.dumps({"request_id": request_id, "img_b64": img_b64}).encode()
        await self._producer.send_and_wait(
            self.requests_topic, value=message, key=request_id.encode()
        )
        logger.info("Inference request produced", extra={"request_id": request_id})

        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise RuntimeError(
                f"Inference timeout after {self.request_timeout}s "
                f"(request_id={request_id})"
            )

    async def process(self, inputs: ToolExecutorInput) -> ToolExecutorOutput:
        if inputs.tool_type == ToolType.CLASSIFICATION:
            if not inputs.img_b64:
                raise ValueError("Image data is required for classification.")
            result = await self._classify_via_stream(inputs.img_b64)
        elif inputs.tool_type == ToolType.RETRIEVAL:
            result = await context_retrieve(self.http_client, inputs.query)
        elif inputs.tool_type == ToolType.DIRECT_ANSWER:
            result = {}
        else:
            raise ValueError(f"Unsupported tool type: {inputs.tool_type}")

        return ToolExecutorOutput(tool_type=inputs.tool_type, result=result)

    async def gprocess(self, state: ChatbotState) -> dict:
        """Graph process: Execute tools in parallel for all enriched queries."""
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
        """Execute a single tool for an enriched query."""
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

            return {
                "query": query,
                "tool_type": tool_type,
                "result": executor_output.result,
            }

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
