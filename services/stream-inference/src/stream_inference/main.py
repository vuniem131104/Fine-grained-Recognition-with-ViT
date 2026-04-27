import asyncio
import json
import os
import time

import httpx
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from structlog import get_logger

from stream_inference.inference import call_kserve

logger = get_logger(__name__)


async def handle_message(
    msg,
    http_client: httpx.AsyncClient,
    producer: AIOKafkaProducer,
    kserve_url: str,
    results_topic: str,
    logs_topic: str,
    model_version: str,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        try:
            payload = json.loads(msg.value)
        except Exception:
            logger.exception("Bad request payload, skipping")
            return

        request_id = payload.get("request_id")
        img_b64 = payload.get("img_b64")
        if not request_id or not img_b64:
            logger.warning(
                "Missing fields in request",
                extra={"has_id": bool(request_id), "has_img": bool(img_b64)},
            )
            return

        started = time.time()
        try:
            result = await call_kserve(http_client, kserve_url, img_b64)
            latency_ms = (time.time() - started) * 1000.0

            await producer.send_and_wait(
                results_topic,
                value=json.dumps({"request_id": request_id, "result": result}).encode(),
                key=request_id.encode(),
            )

            await producer.send_and_wait(
                logs_topic,
                value=json.dumps(
                    {
                        "request_id": request_id,
                        "ts": time.time(),
                        "predicted_class": result["predicted_class"],
                        "probability": result["probability"],
                        "top_3_alternatives": result["top_3_alternatives"],
                        "latency_ms": latency_ms,
                        "model_version": model_version,
                    }
                ).encode(),
                key=request_id.encode(),
            )

            logger.info(
                "Inference handled",
                extra={
                    "request_id": request_id,
                    "predicted_class": result["predicted_class"],
                    "latency_ms": round(latency_ms, 2),
                },
            )
        except Exception as e:
            logger.exception("Inference failed", extra={"request_id": request_id})
            await producer.send_and_wait(
                results_topic,
                value=json.dumps(
                    {"request_id": request_id, "error": str(e)}
                ).encode(),
                key=request_id.encode(),
            )


async def run() -> None:
    bootstrap = os.getenv("REDPANDA_BOOTSTRAP_SERVERS")
    requests_topic = os.getenv("REDPANDA_REQUESTS_TOPIC")
    results_topic = os.getenv("REDPANDA_RESULTS_TOPIC")
    logs_topic = os.getenv("REDPANDA_LOGS_TOPIC")
    group_id = os.getenv("REDPANDA_GROUP_ID")
    kserve_url = os.getenv("KSERVE_MODEL_URL")
    model_version = os.getenv("MODEL_VERSION", "v1")
    max_inflight = int(os.getenv("MAX_INFLIGHT", "8"))

    if not kserve_url:
        raise RuntimeError("KSERVE_MODEL_URL is required")

    consumer = AIOKafkaConsumer(
        requests_topic,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)

    await consumer.start()
    await producer.start()
    logger.info(
        "stream-inference started",
        extra={
            "bootstrap": bootstrap,
            "requests_topic": requests_topic,
            "results_topic": results_topic,
            "logs_topic": logs_topic,
            "kserve_url": kserve_url,
        },
    )

    semaphore = asyncio.Semaphore(max_inflight)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=max_inflight * 2),
    ) as http_client:
        try:
            async for msg in consumer:
                await handle_message(
                    msg,
                    http_client,
                    producer,
                    kserve_url,
                    results_topic,
                    logs_topic,
                    model_version,
                    semaphore,
                )
                await consumer.commit()
        finally:
            await consumer.stop()
            await producer.stop()


def main() -> None:
    asyncio.run(run())
