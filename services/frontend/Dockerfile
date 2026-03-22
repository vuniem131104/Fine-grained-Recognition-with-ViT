# Use official Python runtime as a parent image
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_NO_CACHE=1

COPY pyproject.toml uv.lock ./

COPY services/frontend/pyproject.toml ./services/frontend/
COPY services/frontend/README.md ./services/frontend/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY services/frontend/src ./services/frontend/src
COPY services/frontend/.streamlit ./services/frontend/.streamlit

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_LOGGER_LEVEL=info

WORKDIR /app/services/frontend

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/frontend/app.py"]
