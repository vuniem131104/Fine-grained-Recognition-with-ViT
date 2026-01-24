FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./

RUN uv sync

COPY ./src/serving/ .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "serving"]
