FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_NO_CACHE=1

COPY pyproject.toml uv.lock ./

COPY common/ ./common/

COPY services/authentication/pyproject.toml ./services/authentication/
COPY services/authentication/README.md ./services/authentication/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY services/authentication/src/ ./services/authentication/src/


WORKDIR /app/services/authentication

EXPOSE 8001

CMD ["uv", "run", "authentication"]