FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_NO_CACHE=1

COPY pyproject.toml uv.lock ./

COPY common/ ./common/

COPY services/stream-inference/pyproject.toml ./services/stream-inference/
COPY services/stream-inference/README.md ./services/stream-inference/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY services/stream-inference/src/ ./services/stream-inference/src/

WORKDIR /app/services/stream-inference

CMD ["uv", "run", "stream-inference"]
