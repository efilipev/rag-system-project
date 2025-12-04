#!/bin/bash

# RAG System Kubernetes Deployment Script
# This script automates the deployment of the RAG system to Kubernetes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="rag-system"
TIMEOUT="300s"

# Functions
function print_header() {
    echo -e "\n${GREEN}===================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}===================================================${NC}\n"
}

function print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function wait_for_pods() {
    local label=$1
    local timeout=$2
    print_info "Waiting for pods with label $label to be ready (timeout: $timeout)..."
    kubectl wait --for=condition=ready pod -l $label -n $NAMESPACE --timeout=$timeout || {
        print_error "Pods with label $label failed to become ready"
        return 1
    }
}

# Main deployment process
print_header "RAG System Kubernetes Deployment"

# Step 1: Create namespace
print_header "Step 1: Creating namespace"
kubectl apply -f namespaces/rag-system.yaml
print_info "Namespace created"

# Step 2: Check for secrets
print_header "Step 2: Checking secrets"
if kubectl get secret rag-app-secrets -n $NAMESPACE >/dev/null 2>&1; then
    print_warning "Secrets already exist. Skipping secret creation."
    print_warning "To recreate secrets, run: kubectl delete secret rag-app-secrets -n $NAMESPACE"
else
    print_error "Secrets not found!"
    print_info "Please create secrets manually using the following command:"
    echo ""
    echo "kubectl create secret generic rag-app-secrets \\"
    echo "  --from-literal=postgres-user=raguser \\"
    echo "  --from-literal=postgres-password=\$(openssl rand -base64 32) \\"
    echo "  --from-literal=rabbitmq-user=raguser \\"
    echo "  --from-literal=rabbitmq-password=\$(openssl rand -base64 32) \\"
    echo "  --from-literal=minio-access-key=minioadmin \\"
    echo "  --from-literal=minio-secret-key=\$(openssl rand -base64 32) \\"
    echo "  --from-literal=jwt-secret-key=\$(python3 -c \"import secrets; print(secrets.token_urlsafe(64))\") \\"
    echo "  --from-literal=openai-api-key=sk-YOUR_KEY_HERE \\"
    echo "  --from-literal=admin-token=\$(openssl rand -base64 32) \\"
    echo "  --namespace=$NAMESPACE"
    echo ""
    exit 1
fi

# Step 3: Create ConfigMaps
print_header "Step 3: Creating ConfigMaps"
kubectl apply -f configmaps/app-config.yaml
print_info "ConfigMaps created"

# Step 4: Create PersistentVolumeClaims
print_header "Step 4: Creating PersistentVolumeClaims"
kubectl apply -f persistentvolumeclaims/
print_info "PVCs created"

# Wait for PVCs to be bound
print_info "Waiting for PVCs to be bound..."
sleep 5
kubectl get pvc -n $NAMESPACE

# Step 5: Deploy infrastructure services
print_header "Step 5: Deploying infrastructure services"

print_info "Deploying PostgreSQL..."
kubectl apply -f deployments/postgres.yaml
kubectl apply -f services/postgres-service.yaml

print_info "Deploying Redis..."
kubectl apply -f deployments/redis.yaml
kubectl apply -f services/redis-service.yaml

print_info "Deploying RabbitMQ..."
kubectl apply -f deployments/rabbitmq.yaml
kubectl apply -f services/rabbitmq-service.yaml

print_info "Deploying Qdrant..."
kubectl apply -f deployments/qdrant.yaml
kubectl apply -f services/qdrant-service.yaml

print_info "Deploying MinIO..."
kubectl apply -f deployments/minio.yaml
kubectl apply -f services/minio-service.yaml

# Wait for infrastructure to be ready
print_info "Waiting for infrastructure services to be ready..."
wait_for_pods "tier=database" $TIMEOUT
wait_for_pods "tier=cache" $TIMEOUT
wait_for_pods "tier=messaging" $TIMEOUT
wait_for_pods "tier=vectordb" $TIMEOUT
wait_for_pods "tier=storage" $TIMEOUT

print_info "All infrastructure services are ready!"

# Step 6: Deploy application services
print_header "Step 6: Deploying application services"

print_info "Deploying Query Analysis service..."
kubectl apply -f deployments/query-analysis.yaml
kubectl apply -f services/query-analysis-service.yaml

print_info "Deploying Document Retrieval service..."
kubectl apply -f deployments/document-retrieval.yaml
kubectl apply -f services/document-retrieval-service.yaml

print_info "Deploying Document Ranking service..."
kubectl apply -f deployments/document-ranking.yaml
kubectl apply -f services/document-ranking-service.yaml

print_info "Deploying LaTeX Parser service..."
kubectl apply -f deployments/latex-parser.yaml
kubectl apply -f services/latex-parser-service.yaml

print_info "Deploying LLM Generation service..."
kubectl apply -f deployments/llm-generation.yaml
kubectl apply -f services/llm-generation-service.yaml

print_info "Deploying Response Formatter service..."
kubectl apply -f deployments/response-formatter.yaml
kubectl apply -f services/response-formatter-service.yaml

# Wait for application services
print_info "Waiting for application services to be ready..."
wait_for_pods "tier=application" $TIMEOUT

print_info "All application services are ready!"

# Step 7: Deploy API Gateway
print_header "Step 7: Deploying API Gateway"
kubectl apply -f deployments/api-gateway.yaml
kubectl apply -f services/api-gateway-service.yaml

wait_for_pods "tier=gateway" $TIMEOUT

print_info "API Gateway is ready!"

# Step 8: Deploy Horizontal Pod Autoscalers
print_header "Step 8: Deploying Horizontal Pod Autoscalers"
kubectl apply -f hpa/
print_info "HPAs deployed"

# Step 9: Deploy Ingress (if exists)
if [ -f ingress/api-gateway-ingress.yaml ]; then
    print_header "Step 9: Deploying Ingress"
    print_warning "Make sure to update the domain in ingress/api-gateway-ingress.yaml"
    read -p "Do you want to deploy Ingress? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl apply -f ingress/api-gateway-ingress.yaml
        print_info "Ingress deployed"
    else
        print_info "Skipping Ingress deployment"
    fi
fi

# Final status
print_header "Deployment Complete!"

echo ""
print_info "Deployment Status:"
echo ""

echo "Pods:"
kubectl get pods -n $NAMESPACE

echo ""
echo "Services:"
kubectl get svc -n $NAMESPACE

echo ""
echo "HPAs:"
kubectl get hpa -n $NAMESPACE

echo ""
print_info "To access the API Gateway:"
EXTERNAL_IP=$(kubectl get svc api-gateway-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
if [ "$EXTERNAL_IP" = "pending" ] || [ -z "$EXTERNAL_IP" ]; then
    print_warning "External IP is pending. Run: kubectl get svc api-gateway-service -n $NAMESPACE --watch"
else
    echo "  curl http://$EXTERNAL_IP/api/v1/health"
fi

echo ""
print_info "To view logs:"
echo "  kubectl logs -f deployment/api-gateway -n $NAMESPACE"

echo ""
print_info "To port-forward (for local testing):"
echo "  kubectl port-forward svc/api-gateway-service 8000:80 -n $NAMESPACE"
echo "  Then access: http://localhost:8000/api/v1/health"

print_header "Deployment finished successfully!"
