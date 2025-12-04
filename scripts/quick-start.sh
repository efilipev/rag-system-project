#!/bin/bash

# RAG System Quick Start Script
# This script sets up and starts the entire RAG system

set -e

echo "================================================"
echo "RAG System - Quick Start"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if Docker is running
echo "1. Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi
echo -e "${GREEN}[OK] Docker is running${NC}"

# Check if .env exists
echo ""
echo "2. Checking environment configuration..."
if [ ! -f .env ]; then
    echo -e "${YELLOW}[WARN] .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo "Please edit .env and add your OPENAI_API_KEY"
    echo "Then run this script again."
    exit 0
else
    echo -e "${GREEN}[OK] .env file exists${NC}"
fi

# Check if OPENAI_API_KEY is set
if grep -q "OPENAI_API_KEY=your-openai-api-key-here" .env; then
    echo -e "${YELLOW}[WARN] OPENAI_API_KEY not configured in .env${NC}"
    echo "The LLM Generation service may not work without a valid API key."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Pull latest images
echo ""
echo "3. Pulling Docker images..."
docker-compose pull

# Build services
echo ""
echo "4. Building services (this may take a few minutes)..."
docker-compose build

# Start infrastructure services first
echo ""
echo "5. Starting infrastructure services..."
docker-compose up -d postgres redis rabbitmq qdrant chroma minio

echo "Waiting for infrastructure services to be ready (30 seconds)..."
sleep 30

# Initialize database
echo ""
echo "6. Initializing database..."
docker-compose exec -T postgres psql -U raguser -d postgres -c "SELECT 1 FROM pg_database WHERE datname = 'ragdb'" | grep -q 1 || \
    docker-compose exec -T postgres psql -U raguser -d postgres -c "CREATE DATABASE ragdb;"

# Start application services
echo ""
echo "7. Starting application services..."
docker-compose up -d

# Wait for services to start
echo ""
echo "Waiting for services to start (30 seconds)..."
sleep 30

# Start monitoring services
echo ""
echo "8. Starting monitoring services..."
docker-compose up -d prometheus grafana

# Show status
echo ""
echo "9. Checking service status..."
docker-compose ps

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Services are now running. Run the verification script to check:"
echo "  ./scripts/verify-setup.sh"
echo ""
echo "Or check individual service health:"
echo "  curl http://localhost:8101/health  # Query Analysis"
echo "  curl http://localhost:8102/health  # Document Retrieval"
echo "  curl http://localhost:8103/health  # Document Ranking"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Management interfaces:"
echo "  - RabbitMQ: http://localhost:15672"
echo "  - MinIO: http://localhost:9001"
echo "  - Grafana: http://localhost:3000"
echo "  - Prometheus: http://localhost:9090"
echo ""
echo "To stop all services:"
echo "  docker-compose down"
echo ""
