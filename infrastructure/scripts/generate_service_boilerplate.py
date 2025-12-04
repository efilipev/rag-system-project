#!/usr/bin/env python3
"""
Script to generate boilerplate for microservices
"""
import os
from pathlib import Path

SERVICES_CONFIG = {
    "document-ranking": {
        "description": "Document Ranking Service using cross-encoder models",
        "dependencies": ["sentence-transformers", "transformers", "torch"],
    },
    "latex-parser": {
        "description": "LaTeX Parser Service for mathematical formulas",
        "dependencies": ["sympy", "latex2mathml", "pylatexenc"],
    },
    "llm-generation": {
        "description": "LLM Generation Service using OpenAI or local models",
        "dependencies": ["langchain-openai", "openai", "tiktoken"],
    },
    "response-formatter": {
        "description": "Response Formatter Service with Jinja2 templating",
        "dependencies": ["jinja2", "markdown"],
    },
}

PYPROJECT_TEMPLATE = """[tool.poetry]
name = "{service_name}-service"
version = "0.1.0"
description = "{description}"
authors = ["RAG Team"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.1"
uvicorn = {{extras = ["standard"], version = "^0.24.0"}}
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
langchain = "^0.1.0"
langchain-community = "^0.0.10"
aio-pika = "^9.3.1"
redis = "^5.0.1"
prometheus-client = "^0.19.0"
python-json-logger = "^2.0.7"
{extra_deps}

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-asyncio = "^0.21.1"
black = "^23.11.0"
flake8 = "^6.1.0"
mypy = "^1.7.1"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""

DOCKERFILE_TEMPLATE = """FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - \\
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --no-root --only main

COPY . .
RUN poetry install --no-interaction --no-ansi --only-root

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

MAIN_PY_TEMPLATE = '''"""
{description}
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"Starting {{settings.SERVICE_NAME}} service")
    yield
    logger.info("Shutting down service")


app = FastAPI(
    title="{service_title}",
    description="{description}",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {{"status": "healthy", "service": settings.SERVICE_NAME, "version": "0.1.0"}}


@app.get("/")
async def root():
    return {{"service": settings.SERVICE_NAME, "version": "0.1.0", "status": "running"}}
'''

CONFIG_PY_TEMPLATE = '''"""
Configuration settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    SERVICE_NAME: str = "{service_name}"
    LOG_LEVEL: str = "INFO"
    RABBITMQ_URL: str = "amqp://raguser:ragpassword@localhost:5672/"
    REDIS_URL: str = "redis://localhost:6379"


settings = Settings()
'''

ROUTES_PY_TEMPLATE = '''"""
API routes
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {{"status": "healthy"}}
'''


def create_service(service_name: str, config: dict):
    """Create service directory structure and files"""
    base_path = Path(f"services/{service_name}")
    app_path = base_path / "app"

    # Create directories
    (app_path / "api").mkdir(parents=True, exist_ok=True)
    (app_path / "core").mkdir(parents=True, exist_ok=True)
    (app_path / "services").mkdir(parents=True, exist_ok=True)
    (app_path / "models").mkdir(parents=True, exist_ok=True)

    # Create __init__.py files
    for dir_path in [app_path, app_path / "api", app_path / "core", app_path / "services", app_path / "models"]:
        (dir_path / "__init__.py").write_text("")

    # Create pyproject.toml
    extra_deps = "\n".join([f'{dep} = "*"' for dep in config.get("dependencies", [])])
    pyproject_content = PYPROJECT_TEMPLATE.format(
        service_name=service_name,
        description=config["description"],
        extra_deps=extra_deps
    )
    (base_path / "pyproject.toml").write_text(pyproject_content)

    # Create Dockerfile
    (base_path / "Dockerfile").write_text(DOCKERFILE_TEMPLATE)

    # Create main.py
    service_title = " ".join(word.capitalize() for word in service_name.split("-"))
    main_content = MAIN_PY_TEMPLATE.format(
        description=config["description"],
        service_title=service_title
    )
    (app_path / "main.py").write_text(main_content)

    # Create config.py
    config_content = CONFIG_PY_TEMPLATE.format(service_name=service_name)
    (app_path / "core" / "config.py").write_text(config_content)

    # Copy logging.py from query-analysis
    logging_source = Path("services/query-analysis/app/core/logging.py")
    if logging_source.exists():
        (app_path / "core" / "logging.py").write_text(logging_source.read_text())

    # Create routes.py
    (app_path / "api" / "routes.py").write_text(ROUTES_PY_TEMPLATE)

    print(f"✓ Created {service_name} service")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent.parent)

    for service_name, config in SERVICES_CONFIG.items():
        create_service(service_name, config)

    print("\nAll services created successfully!")