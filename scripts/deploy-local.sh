#!/bin/bash

# Deploy RAG System Locally with Docker Compose
# Uses Ollama for local LLM (no OpenAI API key needed)

set -e

echo "🚀 Deploying RAG System Locally..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Step 1: Build Docker images for all services
echo "📦 Step 1/5: Building Docker images..."
echo "   This may take 10-15 minutes on first run..."
docker-compose build

echo ""
echo "✅ Build complete!"
echo ""

# Step 2: Start infrastructure services first
echo "🏗️  Step 2/5: Starting infrastructure services..."
docker-compose up -d postgres redis rabbitmq qdrant minio ollama

echo "⏳ Waiting for infrastructure to be ready (30 seconds)..."
sleep 30

echo "✅ Infrastructure services running"
echo ""

# Step 3: Pull Ollama model
echo "🦙 Step 3/5: Setting up Ollama model..."
echo "   Pulling llama3.2:3b (this may take 5-10 minutes)..."

# Wait for Ollama to be fully ready
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "   Waiting for Ollama..."
    sleep 5
done

# Pull the model
docker exec rag-ollama ollama pull llama3.2:3b

echo "✅ Ollama model ready!"
echo ""

# Step 4: Start application services
echo "🎯 Step 4/5: Starting application services..."
docker-compose up -d query-analysis document-retrieval document-ranking latex-parser llm-generation response-formatter

echo "⏳ Waiting for services to be ready (20 seconds)..."
sleep 20

echo "✅ Application services running"
echo ""

# Step 5: Start monitoring stack
echo "📊 Step 5/5: Starting monitoring stack..."
docker-compose up -d prometheus grafana

echo ""
echo "✅✅✅ RAG System is now running! ✅✅✅"
echo ""
echo "═══════════════════════════════════════════════════════"
echo "📍 Service URLs:"
echo "═══════════════════════════════════════════════════════"
echo "  API Gateway:           http://localhost:8000"
echo "  Query Analysis:        http://localhost:8101"
echo "  Document Retrieval:    http://localhost:8102"
echo "  Document Ranking:      http://localhost:8103"
echo "  LaTeX Parser:          http://localhost:8104"
echo "  LLM Generation:        http://localhost:8105"
echo "  Response Formatter:    http://localhost:8106"
echo ""
echo "  Ollama API:            http://localhost:11434"
echo "  PostgreSQL:            localhost:5432"
echo "  Redis:                 localhost:6379"
echo "  RabbitMQ Management:   http://localhost:15672"
echo "  MinIO Console:         http://localhost:9001"
echo ""
echo "  Prometheus:            http://localhost:9090"
echo "  Grafana:               http://localhost:3000"
echo "     (admin/admin)"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "🧪 Run health checks:"
echo "   ./scripts/health-check.sh"
echo ""
echo "📚 Load sample documents:"
echo "   ./scripts/load-sample-docs.sh"
echo ""
echo "🔍 View logs:"
echo "   docker-compose logs -f [service-name]"
echo ""
echo "🛑 Stop everything:"
echo "   docker-compose down"
echo ""
