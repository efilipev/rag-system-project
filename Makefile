.PHONY: help build up down restart logs clean test lint format install install-dev install-benchmarks verify-deps
.PHONY: dev-start dev-stop dev-query-analysis dev-retrieval dev-ranking dev-latex dev-llm dev-formatter dev-gateway
.PHONY: check-infra

# Default target
help:
	@echo "RAG System - Makefile Commands"
	@echo "=============================="
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "make build          - Build all Docker images"
	@echo "make up             - Start all services (Docker)"
	@echo "make down           - Stop all services (Docker)"
	@echo "make restart        - Restart all services"
	@echo "make logs           - View logs from all services"
	@echo "make logs-<service> - View logs from specific service"
	@echo "make clean          - Remove all containers, volumes, and networks"
	@echo ""
	@echo "💻 Local Development (No Docker):"
	@echo "make check-infra    - Check if infrastructure services are running"
	@echo "make dev-start      - Instructions for starting services locally"
	@echo "make dev-query-analysis  - Run Query Analysis service locally (port 8101)"
	@echo "make dev-retrieval       - Run Document Retrieval service locally (port 8102)"
	@echo "make dev-ranking         - Run Document Ranking service locally (port 8103)"
	@echo "make dev-latex           - Run LaTeX Parser service locally (port 8104)"
	@echo "make dev-llm             - Run LLM Generation service locally (port 8105)"
	@echo "make dev-formatter       - Run Response Formatter service locally (port 8106)"
	@echo "make dev-gateway         - Run API Gateway service locally (port 8100)"
	@echo ""
	@echo "📦 Dependency Commands:"
	@echo "make install        - Install all dependencies (Poetry + pip)"
	@echo "make install-dev    - Install only development dependencies"
	@echo "make install-benchmarks - Install benchmarking dependencies"
	@echo "make verify-deps    - Verify all dependencies are installed"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "make test           - Run tests for all services"
	@echo "make test-benchmarks- Run benchmark tests"
	@echo "make lint           - Run linting for all services"
	@echo "make format         - Format code for all services"
	@echo ""
	@echo "🏥 Health & Status:"
	@echo "make health         - Check health of all services"
	@echo "make status         - View service status"
	@echo ""
	@echo "🔧 Database & Shell:"
	@echo "make init-db        - Initialize database"
	@echo "make shell-<service>- Open shell in service container"

# Build all services
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d
	@echo "Services are starting..."
	@echo "Run 'make logs' to view logs"
	@echo "Run 'make health' to check service health"

# Stop all services
down:
	docker-compose down

# Restart all services
restart:
	docker-compose restart

# View logs from all services
logs:
	docker-compose logs -f

# View logs from specific service
logs-query-analysis:
	docker-compose logs -f query-analysis

logs-document-retrieval:
	docker-compose logs -f document-retrieval

logs-document-ranking:
	docker-compose logs -f document-ranking

logs-latex-parser:
	docker-compose logs -f latex-parser

logs-llm-generation:
	docker-compose logs -f llm-generation

logs-response-formatter:
	docker-compose logs -f response-formatter

# Clean everything
clean:
	docker-compose down -v
	docker system prune -f

# Initialize database
init-db:
	@echo "🗄️  Initializing databases..."
	@echo "Creating ragdb database..."
	@docker-compose exec -T postgres psql -U raguser -d postgres -c "CREATE DATABASE ragdb;" || echo "  ℹ️  ragdb already exists"
	@echo "Creating kong user..."
	@docker-compose exec -T postgres psql -U raguser -d postgres -c "CREATE USER kong WITH PASSWORD 'kong';" || echo "  ℹ️  kong user already exists"
	@echo "Creating kong database..."
	@docker-compose exec -T postgres psql -U raguser -d postgres -c "CREATE DATABASE kong OWNER kong;" || echo "  ℹ️  kong database already exists"
	@echo "Granting kong user permissions..."
	@docker-compose exec -T postgres psql -U raguser -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE kong TO kong;" || echo "  ℹ️  permissions already granted"
	@echo "Running Kong migrations..."
	@docker-compose run --rm api-gateway kong migrations bootstrap || echo "  ℹ️  Kong migrations already applied"
	@echo "✅ Database initialization complete!"

# Health check for all services
health:
	@echo "Checking service health..."
	@echo "\nQuery Analysis Service:"
	@curl -s http://localhost:8101/health || echo "❌ Not responding"
	@echo "\nDocument Retrieval Service:"
	@curl -s http://localhost:8102/health || echo "❌ Not responding"
	@echo "\nDocument Ranking Service:"
	@curl -s http://localhost:8103/health || echo "❌ Not responding"
	@echo "\nLaTeX Parser Service:"
	@curl -s http://localhost:8104/health || echo "❌ Not responding"
	@echo "\nLLM Generation Service:"
	@curl -s http://localhost:8105/health || echo "❌ Not responding"
	@echo "\nResponse Formatter Service:"
	@curl -s http://localhost:8106/health || echo "❌ Not responding"
	@echo "\n"

# Run tests for all services
test:
	@for service in query-analysis document-retrieval document-ranking latex-parser llm-generation response-formatter; do \
		echo "Testing $$service..."; \
		cd services/$$service && poetry run pytest || true; \
		cd ../..; \
	done

# Run linting for all services
lint:
	@for service in query-analysis document-retrieval document-ranking latex-parser llm-generation response-formatter; do \
		echo "Linting $$service..."; \
		cd services/$$service && poetry run flake8 . || true; \
		cd ../..; \
	done

# Format code for all services
format:
	@for service in query-analysis document-retrieval document-ranking latex-parser llm-generation response-formatter; do \
		echo "Formatting $$service..."; \
		cd services/$$service && poetry run black . && poetry run isort . || true; \
		cd ../..; \
	done

# Open shell in service container
shell-query-analysis:
	docker-compose exec query-analysis /bin/sh

shell-document-retrieval:
	docker-compose exec document-retrieval /bin/sh

shell-postgres:
	docker-compose exec postgres psql -U raguser -d ragdb

shell-redis:
	docker-compose exec redis redis-cli

# View service status
status:
	docker-compose ps

# Pull latest images
pull:
	docker-compose pull

# Rebuild and restart specific service
rebuild-%:
	docker-compose up -d --build $*

# Scale specific service
scale-%:
	docker-compose up -d --scale $*

# Export environment variables
env:
	@echo "Exporting environment variables from .env"
	@export $(cat .env | xargs)

# ============================================================================
# 📦 Dependency Installation Targets
# ============================================================================

# Install all dependencies (Poetry for services + pip for benchmarks)
install:
	@echo "🔧 Installing all dependencies..."
	@echo ""
	@echo "[1/4] Installing Poetry dependencies for each service..."
	@for service in query-analysis document-retrieval document-ranking latex-parser llm-generation response-formatter api-gateway; do \
		if [ -d "services/$$service" ]; then \
			echo "  → Installing $$service..."; \
			cd services/$$service && poetry install && cd ../..; \
		fi \
	done
	@if [ -d "shared" ]; then \
		echo "  → Installing shared..."; \
		cd shared && poetry install && cd ..; \
	fi
	@echo ""
	@echo "[2/4] Installing advanced RAG dependencies..."
	pip install -r requirements-advanced-rag.txt
	@echo ""
	@echo "[3/4] Downloading spaCy language model..."
	python -m spacy download en_core_web_sm
	@echo ""
	@echo "[4/4] Verifying installation..."
	@$(MAKE) verify-deps
	@echo ""
	@echo "✅ All dependencies installed!"

# Install only development dependencies
install-dev:
	@echo "🔧 Installing development dependencies..."
	@for service in query-analysis document-retrieval document-ranking latex-parser llm-generation response-formatter api-gateway; do \
		if [ -d "services/$$service" ]; then \
			echo "  → Installing $$service (dev)..."; \
			cd services/$$service && poetry install --with dev && cd ../..; \
		fi \
	done
	pip install pytest pytest-asyncio pytest-cov pytest-benchmark black flake8 mypy isort
	@echo "✅ Development dependencies installed!"

# Install only benchmarking dependencies
install-benchmarks:
	@echo "🔧 Installing benchmarking dependencies..."
	pip install -r requirements-advanced-rag.txt
	python -m spacy download en_core_web_sm
	@echo "✅ Benchmarking dependencies installed!"

# Verify all dependencies are installed correctly
verify-deps:
	@echo "🔍 Verifying dependencies..."
	@echo ""
	@echo "Core dependencies (Poetry virtualenv):"
	@cd services/query-analysis && poetry run python -c "import fastapi, pydantic; print('  ✅ Core (FastAPI, Pydantic)')" 2>/dev/null || echo "  ❌ Core dependencies missing in query-analysis"
	@echo ""
	@echo "LaTeX dependencies (global):"
	@python -c "import sympy, latex2mathml, pylatexenc; print('  ✅ LaTeX (SymPy, latex2mathml, pylatexenc)')" 2>/dev/null || echo "  ❌ LaTeX dependencies missing"
	@echo ""
	@echo "HTTP client (global):"
	@python -c "import httpx; print('  ✅ HTTP (httpx)')" 2>/dev/null || echo "  ❌ httpx missing"
	@echo ""
	@echo "ML & Data Science (global):"
	@python -c "import numpy, sklearn, pandas; print('  ✅ ML (NumPy, sklearn, pandas)')" 2>/dev/null || echo "  ❌ ML dependencies missing"
	@echo ""
	@echo "Benchmarking (global):"
	@python -c "import beir, matplotlib, seaborn; print('  ✅ Benchmarks (BEIR, matplotlib, seaborn)')" 2>/dev/null || echo "  ❌ Benchmark dependencies missing"
	@echo ""
	@echo "NLP (global):"
	@python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('  ✅ NLP (spaCy + en_core_web_sm)')" 2>/dev/null || echo "  ❌ spaCy or model missing"
	@echo ""
	@echo "Query Construction (Poetry virtualenv):"
	@cd services/query-analysis && PYTHONPATH=../.. poetry run python -c "from app.services.latex_query_analyzer import LatexQueryAnalyzer; from app.services.query_constructor import QueryConstructor; print('  ✅ Query Construction')" 2>/dev/null || echo "  ❌ Query construction modules have issues"
	@echo ""
	@echo "Reranking (global):"
	@python -c "import sentence_transformers, rank_bm25; print('  ✅ Reranking (sentence-transformers, rank-bm25)')" 2>/dev/null || echo "  ❌ Reranking dependencies missing"
	@echo ""
	@echo "Service Dependencies:"
	@cd services/query-analysis && poetry run python -c "import fastapi, langchain, spacy, transformers; print('  ✅ query-analysis')" 2>/dev/null || echo "  ❌ query-analysis"
	@cd services/document-retrieval && poetry run python -c "import fastapi; print('  ✅ document-retrieval')" 2>/dev/null || echo "  ❌ document-retrieval"
	@cd services/document-ranking && poetry run python -c "import fastapi, sentence_transformers; print('  ✅ document-ranking')" 2>/dev/null || echo "  ❌ document-ranking"
	@cd services/latex-parser && poetry run python -c "import fastapi, sympy, latex2mathml; print('  ✅ latex-parser')" 2>/dev/null || echo "  ❌ latex-parser"
	@cd services/llm-generation && poetry run python -c "import fastapi; print('  ✅ llm-generation')" 2>/dev/null || echo "  ❌ llm-generation"
	@cd services/response-formatter && poetry run python -c "import fastapi; print('  ✅ response-formatter')" 2>/dev/null || echo "  ❌ response-formatter"
	@cd services/api-gateway && poetry run python -c "import fastapi; print('  ✅ api-gateway')" 2>/dev/null || echo "  ❌ api-gateway"

# Run benchmark tests
test-benchmarks:
	@echo "🧪 Running benchmark tests..."
	pytest tests/benchmarks/ -v --tb=short
	@echo "✅ Benchmark tests complete!"

# Quick dependency check (fast)
check-deps:
	@echo "⚡ Quick dependency check..."
	@python -c "import httpx, sympy, numpy, beir; print('✅ Critical dependencies OK')" || echo "❌ Missing critical dependencies - run 'make install'"

# ============================================================================
# 💻 Local Development Commands (No Docker)
# ============================================================================

# Check infrastructure services
check-infra:
	@echo "🔍 Checking infrastructure services..."
	@echo ""
	@echo "Redis:"
	@redis-cli ping > /dev/null 2>&1 && echo "  ✅ Redis is running" || echo "  ❌ Redis not running - Start with: brew services start redis"
	@echo ""
	@echo "PostgreSQL:"
	@pg_isready -h localhost -p 5432 > /dev/null 2>&1 && echo "  ✅ PostgreSQL is running" || echo "  ❌ PostgreSQL not running - Start with: brew services start postgresql@16"
	@echo ""
	@echo "RabbitMQ:"
	@curl -s http://localhost:15672 > /dev/null 2>&1 && echo "  ✅ RabbitMQ is running" || echo "  ⚠️  RabbitMQ may not be running - Start with: brew services start rabbitmq"
	@echo ""
	@echo "Qdrant (optional):"
	@curl -s http://localhost:6333/healthz > /dev/null 2>&1 && echo "  ✅ Qdrant is running" || echo "  ⚠️  Qdrant not running (optional) - Start with: docker run -d -p 6333:6333 qdrant/qdrant"
	@echo ""
	@echo "Ollama (optional):"
	@curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && echo "  ✅ Ollama is running" || echo "  ⚠️  Ollama not running (optional) - Start with: ollama serve"
	@echo ""

# Show instructions for starting services locally
dev-start:
	@echo "💻 Running RAG System Locally (Without Docker)"
	@echo "=============================================="
	@echo ""
	@echo "📋 Prerequisites:"
	@echo "  1. Install dependencies: make install"
	@echo "  2. Start infrastructure: make check-infra"
	@echo ""
	@echo "🚀 Start each service in a separate terminal:"
	@echo ""
	@echo "  Terminal 1: make dev-query-analysis"
	@echo "  Terminal 2: make dev-retrieval"
	@echo "  Terminal 3: make dev-ranking"
	@echo "  Terminal 4: make dev-latex"
	@echo "  Terminal 5: make dev-llm"
	@echo "  Terminal 6: make dev-formatter"
	@echo "  Terminal 7: make dev-gateway"
	@echo ""
	@echo "📖 Full guide: See LOCAL_SETUP_GUIDE.md"
	@echo ""
	@echo "💡 Tip: Use tmux or screen to manage multiple terminals"
	@echo ""

# Run Query Analysis service locally
dev-query-analysis:
	@echo "🚀 Starting Query Analysis Service on port 8101..."
	@cd services/query-analysis && PYTHONPATH=../.. poetry run uvicorn app.main:app --host 0.0.0.0 --port 8101 --reload

# Run Document Retrieval service locally
dev-retrieval:
	@echo "🚀 Starting Document Retrieval Service on port 8102..."
	@cd services/document-retrieval && PYTHONPATH=../.. poetry run uvicorn app.main:app --host 0.0.0.0 --port 8102 --reload

# Run Document Ranking service locally
dev-ranking:
	@echo "🚀 Starting Document Ranking Service on port 8103..."
	@cd services/document-ranking && PYTHONPATH=../.. poetry run uvicorn app.main:app --host 0.0.0.0 --port 8103 --reload

# Run LaTeX Parser service locally
dev-latex:
	@echo "🚀 Starting LaTeX Parser Service on port 8104..."
	@cd services/latex-parser && PYTHONPATH=../.. poetry run uvicorn app.main:app --host 0.0.0.0 --port 8104 --reload

# Run LLM Generation service locally
dev-llm:
	@echo "🚀 Starting LLM Generation Service on port 8105..."
	@cd services/llm-generation && PYTHONPATH=../.. poetry run uvicorn app.main:app --host 0.0.0.0 --port 8105 --reload

# Run Response Formatter service locally
dev-formatter:
	@echo "🚀 Starting Response Formatter Service on port 8106..."
	@cd services/response-formatter && PYTHONPATH=../.. poetry run uvicorn app.main:app --host 0.0.0.0 --port 8106 --reload

# Run API Gateway service locally
dev-gateway:
	@echo "🚀 Starting API Gateway Service on port 8100..."
	@cd services/api-gateway && PYTHONPATH=../.. poetry run uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload
