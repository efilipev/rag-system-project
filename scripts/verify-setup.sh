#!/bin/bash

# RAG System Setup Verification Script
# This script verifies that all services are running correctly

set -e

echo "================================================"
echo "RAG System - Setup Verification"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service_name=$1
    local port=$2
    local endpoint=${3:-/health}

    echo -n "Checking $service_name... "

    if curl -sf "http://localhost:$port$endpoint" > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC}"
        return 0
    else
        echo -e "${RED}[FAIL]${NC}"
        return 1
    fi
}

# Function to check port availability
check_port() {
    local port=$1
    local service=$2

    if nc -z localhost $port 2>/dev/null; then
        echo -e "Port $port ($service): ${GREEN}[OK] Open${NC}"
        return 0
    else
        echo -e "Port $port ($service): ${RED}[FAIL] Closed${NC}"
        return 1
    fi
}

echo "1. Checking if Docker is running..."
if docker info > /dev/null 2>&1; then
    echo -e "${GREEN}[OK] Docker is running${NC}"
else
    echo -e "${RED}[FAIL] Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo ""
echo "2. Checking if services are up..."
docker-compose ps

echo ""
echo "3. Checking infrastructure services..."

# Check PostgreSQL
echo -n "PostgreSQL... "
if docker-compose exec -T postgres pg_isready -U raguser > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC}"
else
    echo -e "${RED}[FAIL]${NC}"
fi

# Check Redis
echo -n "Redis... "
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC}"
else
    echo -e "${RED}[FAIL]${NC}"
fi

# Check RabbitMQ
echo -n "RabbitMQ... "
if docker-compose exec -T rabbitmq rabbitmq-diagnostics ping > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC}"
else
    echo -e "${RED}[FAIL]${NC}"
fi

# Check Qdrant
echo -n "Qdrant... "
if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC}"
else
    echo -e "${RED}[FAIL]${NC}"
fi

# Check Chroma
echo -n "Chroma... "
if curl -sf http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC}"
else
    echo -e "${RED}[FAIL]${NC}"
fi

echo ""
echo "4. Checking microservices..."

# Wait a bit for services to be ready
sleep 5

# Check each microservice
check_service "Query Analysis Service" 8101
check_service "Document Retrieval Service" 8102
check_service "Document Ranking Service" 8103
check_service "LaTeX Parser Service" 8104
check_service "LLM Generation Service" 8105
check_service "Response Formatter Service" 8106

echo ""
echo "5. Checking monitoring services..."
check_service "Prometheus" 9090 "/-/healthy"
check_service "Grafana" 3000 "/api/health"

echo ""
echo "6. Checking management interfaces..."
echo "RabbitMQ Management: http://localhost:15672"
echo "  User: raguser, Password: ragpassword"
echo ""
echo "MinIO Console: http://localhost:9001"
echo "  User: minioadmin, Password: minioadmin123"
echo ""
echo "Grafana: http://localhost:3000"
echo "  User: admin, Password: admin"
echo ""
echo "Prometheus: http://localhost:9090"
echo ""

echo "================================================"
echo "Verification Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Test the Query Analysis Service:"
echo '   curl -X POST http://localhost:8101/api/v1/analyze -H "Content-Type: application/json" -d '"'"'{"query": "test query"}'"'"''
echo ""
echo "2. View service logs:"
echo "   docker-compose logs -f query-analysis"
echo ""
echo "3. Access monitoring dashboards:"
echo "   - Grafana: http://localhost:3000"
echo "   - Prometheus: http://localhost:9090"
echo ""
