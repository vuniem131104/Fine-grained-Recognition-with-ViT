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

COPY common/ ./common/

COPY services/chatbot/pyproject.toml ./services/chatbot/
COPY services/chatbot/README.md ./services/chatbot/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY services/chatbot/src/ ./services/chatbot/src/

WORKDIR /app/services/chatbot

EXPOSE 8000

CMD ["uv", "run", "chatbot"]
