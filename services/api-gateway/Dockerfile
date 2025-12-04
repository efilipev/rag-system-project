FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy shared library first
COPY shared /app/shared
RUN cd /app/shared && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root --only main

# Copy api-gateway service
COPY services/api-gateway/pyproject.toml services/api-gateway/poetry.lock* ./

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --no-root --only main

COPY services/api-gateway/ .
RUN poetry install --no-interaction --no-ansi --only-root || true

# Create logs directory
RUN mkdir -p logs

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
