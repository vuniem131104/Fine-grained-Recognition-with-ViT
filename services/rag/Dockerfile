FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_NO_CACHE=1

COPY pyproject.toml uv.lock ./

COPY services/rag/pyproject.toml ./services/rag/
COPY services/rag/README.md ./services/rag/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY services/rag/src/ ./services/rag/src/

WORKDIR /app/services/rag

EXPOSE 3010

CMD ["uv", "run", "rag"]
