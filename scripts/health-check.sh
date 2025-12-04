#!/bin/bash

# Health Check Script for RAG System
# Checks all services are running and healthy

set -e

echo "🏥 RAG System Health Check"
echo "═══════════════════════════════════════════════════════"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check HTTP endpoint
check_http() {
    local name=$1
    local url=$2
    local timeout=${3:-5}

    printf "%-25s" "$name:"
    if timeout $timeout curl -sf "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ Unhealthy${NC}"
        return 1
    fi
}

# Function to check TCP port
check_tcp() {
    local name=$1
    local host=$2
    local port=$3

    printf "%-25s" "$name:"
    if nc -z -w5 "$host" "$port" 2>/dev/null; then
        echo -e "${GREEN}✅ Running${NC}"
        return 0
    else
        echo -e "${RED}❌ Down${NC}"
        return 1
    fi
}

total=0
healthy=0

# Check infrastructure services
echo "📦 Infrastructure Services:"
echo "───────────────────────────────────────────────────────"

check_tcp "PostgreSQL" localhost 5432 && ((healthy++)) || true
((total++))

check_tcp "Redis" localhost 6379 && ((healthy++)) || true
((total++))

check_tcp "RabbitMQ" localhost 5672 && ((healthy++)) || true
((total++))

check_http "RabbitMQ Management" "http://localhost:15672" && ((healthy++)) || true
((total++))

check_http "Qdrant" "http://localhost:6333/healthz" && ((healthy++)) || true
((total++))

check_http "MinIO" "http://localhost:9000/minio/health/live" && ((healthy++)) || true
((total++))

check_http "Ollama" "http://localhost:11434/api/tags" 10 && ((healthy++)) || true
((total++))

echo ""
echo "🎯 Application Services:"
echo "───────────────────────────────────────────────────────"

check_http "Query Analysis" "http://localhost:8101/health" && ((healthy++)) || true
((total++))

check_http "Document Retrieval" "http://localhost:8102/health" && ((healthy++)) || true
((total++))

check_http "Document Ranking" "http://localhost:8103/health" && ((healthy++)) || true
((total++))

check_http "LaTeX Parser" "http://localhost:8104/health" && ((healthy++)) || true
((total++))

check_http "LLM Generation" "http://localhost:8105/health" 10 && ((healthy++)) || true
((total++))

check_http "Response Formatter" "http://localhost:8106/health" && ((healthy++)) || true
((total++))

echo ""
echo "📊 Monitoring Stack:"
echo "───────────────────────────────────────────────────────"

check_http "Prometheus" "http://localhost:9090/-/healthy" && ((healthy++)) || true
((total++))

check_http "Grafana" "http://localhost:3000/api/health" && ((healthy++)) || true
((total++))

echo ""
echo "═══════════════════════════════════════════════════════"
echo ""

# Calculate percentage
percentage=$((healthy * 100 / total))

if [ $percentage -eq 100 ]; then
    echo -e "${GREEN}✅ System Status: ALL HEALTHY ($healthy/$total)${NC}"
    echo ""
    echo "🎉 All services are running!"
    exit 0
elif [ $percentage -ge 80 ]; then
    echo -e "${YELLOW}⚠️  System Status: DEGRADED ($healthy/$total - $percentage%)${NC}"
    echo ""
    echo "Some services are down. Check logs:"
    echo "  docker-compose logs [service-name]"
    exit 1
else
    echo -e "${RED}❌ System Status: CRITICAL ($healthy/$total - $percentage%)${NC}"
    echo ""
    echo "Multiple services are down. Try restarting:"
    echo "  docker-compose restart"
    exit 2
fi
