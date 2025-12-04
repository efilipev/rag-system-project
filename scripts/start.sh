#!/bin/bash
# RAG System Full Startup Script
# Usage: ./scripts/start.sh [docker|local] [--skip-wiki] [--quick]

set -e

# Configuration
MODE="${1:-docker}"
SKIP_WIKI=false
QUICK_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-wiki)
            SKIP_WIKI=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        docker|local)
            MODE=$arg
            shift
            ;;
    esac
done

echo ""
echo "========================================"
echo "  RAG System - Full Startup"
echo "  Mode: $MODE"
echo "========================================"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for a service
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=1

    echo -n "  Waiting for $name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo " [OK]"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo " [FAIL]"
    return 1
}

# Step 1: Check prerequisites
echo "[1/6] Checking prerequisites..."

if ! command_exists python3; then
    echo "[FAIL] Python 3 is required"
    exit 1
fi
echo "  [OK] Python 3 found"

if ! command_exists node; then
    echo "[FAIL] Node.js is required"
    exit 1
fi
echo "  [OK] Node.js found"

if ! command_exists npm; then
    echo "[FAIL] npm is required"
    exit 1
fi
echo "  [OK] npm found"

if [ "$MODE" = "docker" ]; then
    if ! command_exists docker; then
        echo "[FAIL] Docker is required for docker mode"
        exit 1
    fi
    echo "  [OK] Docker found"

    if ! command_exists docker-compose; then
        echo "[FAIL] docker-compose is required"
        exit 1
    fi
    echo "  [OK] docker-compose found"
fi

# Step 2: Install dependencies
echo ""
echo "[2/6] Installing dependencies..."

if [ "$QUICK_MODE" = false ]; then
    # Check if node_modules exists for frontend
    if [ ! -d "clients/frontend/node_modules" ]; then
        echo "  Installing frontend dependencies..."
        cd clients/frontend && npm install && cd ../..
    else
        echo "  [OK] Frontend dependencies already installed"
    fi

    # Check Python dependencies for wiki ingestion
    if ! python3 -c "import datasets, sentence_transformers, qdrant_client" 2>/dev/null; then
        echo "  Installing Python dependencies for wiki ingestion..."
        pip install datasets tqdm sentence-transformers qdrant-client --quiet
    else
        echo "  [OK] Python dependencies already installed"
    fi
else
    echo "  Skipping dependency check (quick mode)"
fi

# Step 3: Start services
echo ""
echo "[3/6] Starting services..."

if [ "$MODE" = "docker" ]; then
    echo "  Building and starting Docker containers..."
    docker-compose up -d --build

    echo "  Waiting for services to be ready..."
    sleep 10

    wait_for_service "http://localhost:8000/health" "API Gateway" 60
    wait_for_service "http://localhost:6335/collections" "Qdrant" 30
    wait_for_service "http://localhost:11434/api/tags" "Ollama" 60
else
    echo "  Starting local services..."

    # Check infrastructure
    echo "  Checking infrastructure..."

    if ! redis-cli ping > /dev/null 2>&1; then
        echo "  [WARN] Redis not running. Start with: brew services start redis"
    fi

    if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
        echo "  [WARN] PostgreSQL not running. Start with: brew services start postgresql@16"
    fi

    if ! curl -sf http://localhost:6335/collections > /dev/null 2>&1; then
        echo "  [WARN] Qdrant not running. Start with: docker run -d -p 6335:6333 qdrant/qdrant"
    fi

    # Start services in background
    echo "  Starting microservices..."

    cd services/query-analysis && PYTHONPATH=../.. poetry run uvicorn src.main:app --host 0.0.0.0 --port 8101 > /tmp/query-analysis.log 2>&1 &
    cd ../document-retrieval && PYTHONPATH=../.. poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 > /tmp/document-retrieval.log 2>&1 &
    cd ../document-ranking && PYTHONPATH=../.. poetry run uvicorn src.main:app --host 0.0.0.0 --port 8103 > /tmp/document-ranking.log 2>&1 &
    cd ../latex-parser && PYTHONPATH=../.. poetry run uvicorn src.main:app --host 0.0.0.0 --port 8104 > /tmp/latex-parser.log 2>&1 &
    cd ../llm-generation && PYTHONPATH=../.. poetry run uvicorn src.main:app --host 0.0.0.0 --port 8105 > /tmp/llm-generation.log 2>&1 &
    cd ../response-formatter && PYTHONPATH=../.. poetry run uvicorn src.main:app --host 0.0.0.0 --port 8106 > /tmp/response-formatter.log 2>&1 &
    sleep 5
    cd ../api-gateway && PYTHONPATH=../.. poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 > /tmp/api-gateway.log 2>&1 &
    cd ../..

    echo "  Waiting for services..."
    sleep 10
    wait_for_service "http://localhost:8000/health" "API Gateway" 30
fi

# Step 4: Initialize database
echo ""
echo "[4/6] Initializing database..."

if [ "$MODE" = "docker" ]; then
    docker-compose exec -T postgres psql -U raguser -d postgres -c "CREATE DATABASE ragdb;" 2>/dev/null || echo "  [OK] ragdb already exists"
else
    psql -h localhost -U raguser -d postgres -c "CREATE DATABASE ragdb;" 2>/dev/null || echo "  [OK] ragdb already exists"
fi

# Step 5: Setup Ollama
echo ""
echo "[5/6] Setting up Ollama..."

if [ "$MODE" = "docker" ]; then
    if docker-compose exec -T ollama ollama list 2>/dev/null | grep -q "llama3.2:1b"; then
        echo "  [OK] llama3.2:1b model already available"
    else
        echo "  Pulling llama3.2:1b model (this may take a few minutes)..."
        docker-compose exec -T ollama ollama pull llama3.2:1b || echo "  [WARN] Could not pull model"
    fi
else
    if ollama list 2>/dev/null | grep -q "llama3.2:1b"; then
        echo "  [OK] llama3.2:1b model already available"
    else
        echo "  Pulling llama3.2:1b model..."
        ollama pull llama3.2:1b || echo "  [WARN] Could not pull model"
    fi
fi

# Step 6: Check/Ingest Wikipedia data
echo ""
echo "[6/6] Checking Wikipedia embeddings..."

WIKI_DATA_EXISTS=false
if curl -sf "http://localhost:6335/collections/wikipedia" > /dev/null 2>&1; then
    POINTS=$(curl -sf "http://localhost:6335/collections/wikipedia" | grep -o '"points_count":[0-9]*' | grep -o '[0-9]*' || echo "0")
    if [ "$POINTS" -gt 0 ]; then
        echo "  [OK] Wikipedia collection exists with $POINTS points"
        WIKI_DATA_EXISTS=true
    fi
fi

if [ "$WIKI_DATA_EXISTS" = false ] && [ "$SKIP_WIKI" = false ]; then
    echo "  [WARN] Wikipedia data not found."
    echo ""
    read -p "  Would you like to ingest Wikipedia data now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Starting Wikipedia ingestion (1000 articles for quick start)..."
        python3 scripts/ingest_wikipedia.py --sample 1000 --qdrant-port 6335
    else
        echo "  Skipping Wikipedia ingestion. Run 'make ingest-wiki' later."
    fi
elif [ "$SKIP_WIKI" = true ]; then
    echo "  Skipping Wikipedia check (--skip-wiki flag)"
fi

# Start frontend
echo ""
echo "Starting frontend..."
cd clients/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

# Final summary
echo ""
echo "========================================"
echo "  RAG System is Ready!"
echo "========================================"
echo ""
echo "Access Points:"
echo "   - Frontend:     http://localhost:3005"
echo "   - API Gateway:  http://localhost:8000"
echo "   - API Docs:     http://localhost:8000/docs"
if [ "$MODE" = "docker" ]; then
    echo "   - Grafana:      http://localhost:3001 (admin/admin)"
    echo "   - RabbitMQ:     http://localhost:15672 (raguser/ragpassword)"
fi
echo ""
echo "Useful Commands:"
echo "   - make logs          - View service logs"
echo "   - make health        - Check service health"
echo "   - make ingest-wiki   - Ingest Wikipedia data"
echo "   - make down          - Stop all services"
echo ""
echo "Frontend running at http://localhost:3005"
echo ""

# Keep script running to show frontend output
wait $FRONTEND_PID
