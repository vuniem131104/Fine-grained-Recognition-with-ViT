FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock ./

COPY services/serving/pyproject.toml ./services/serving/
COPY services/serving/README.md ./services/serving/
COPY services/serving/ ./services/serving/

RUN uv sync --no-install-project --no-dev
RUN uv sync --no-dev --package serving

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "serving"]
