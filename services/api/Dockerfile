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

COPY services/api/pyproject.toml ./services/api/
COPY services/api/README.md ./services/api/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY services/api/src/ ./services/api/src/

WORKDIR /app/services/api

EXPOSE 8000

CMD ["uv", "run", "api"]
