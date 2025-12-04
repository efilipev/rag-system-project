# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the RAG System to a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured to access your cluster
- Helm 3 (optional, for Helm-based deployment)
- Container registry with Docker images
- Storage provisioner for PersistentVolumes
- NGINX Ingress Controller (for Ingress)
- cert-manager (for TLS certificates)

## Directory Structure

```
k8s/
├── namespaces/          # Namespace definitions
├── configmaps/          # Application configuration
├── secrets/             # Secret templates (DO NOT commit actual secrets!)
├── persistentvolumeclaims/  # Storage claims
├── deployments/         # Deployment manifests
├── services/            # Service definitions
├── ingress/             # Ingress configuration
├── hpa/                 # Horizontal Pod Autoscalers
└── README.md            # This file
```

## Quick Start

### 1. Build and Push Docker Images

First, build and push all service images to your container registry:

```bash
# Set your registry
export REGISTRY=your-registry.example.com

# Build and push all images
for service in query-analysis document-retrieval document-ranking latex-parser llm-generation response-formatter api-gateway; do
  cd services/$service
  docker build -t $REGISTRY/rag-system/$service:latest .
  docker push $REGISTRY/rag-system/$service:latest
  cd ../..
done
```

Update image references in deployment manifests to point to your registry.

### 2. Create Namespace

```bash
kubectl apply -f namespaces/rag-system.yaml
```

### 3. Create Secrets

**IMPORTANT**: Never commit actual secrets to git!

Generate secure secrets:

```bash
# Generate JWT secret
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")

# Create secrets
kubectl create secret generic rag-app-secrets \
  --from-literal=postgres-user=raguser \
  --from-literal=postgres-password=$(openssl rand -base64 32) \
  --from-literal=rabbitmq-user=raguser \
  --from-literal=rabbitmq-password=$(openssl rand -base64 32) \
  --from-literal=minio-access-key=minioadmin \
  --from-literal=minio-secret-key=$(openssl rand -base64 32) \
  --from-literal=jwt-secret-key=$JWT_SECRET \
  --from-literal=openai-api-key=sk-YOUR_ACTUAL_KEY_HERE \
  --from-literal=admin-token=$(openssl rand -base64 32) \
  --namespace=rag-system
```

**Production**: Use a secrets manager (HashiCorp Vault, AWS Secrets Manager, etc.)

### 4. Create ConfigMaps

```bash
kubectl apply -f configmaps/app-config.yaml

# If you have the init-db.sql script
kubectl create configmap postgres-init-scripts \
  --from-file=init-db.sql=../infrastructure/scripts/init-db.sql \
  --namespace=rag-system
```

### 5. Create PersistentVolumeClaims

```bash
kubectl apply -f persistentvolumeclaims/
```

Verify PVCs are bound:

```bash
kubectl get pvc -n rag-system
```

### 6. Deploy Infrastructure Services

Deploy in order to respect dependencies:

```bash
# Database
kubectl apply -f deployments/postgres.yaml
kubectl apply -f services/postgres-service.yaml

# Cache
kubectl apply -f deployments/redis.yaml
kubectl apply -f services/redis-service.yaml

# Message Queue
kubectl apply -f deployments/rabbitmq.yaml
kubectl apply -f services/rabbitmq-service.yaml

# Vector Database
kubectl apply -f deployments/qdrant.yaml
kubectl apply -f services/qdrant-service.yaml

# Object Storage
kubectl apply -f deployments/minio.yaml
kubectl apply -f services/minio-service.yaml
```

Wait for all infrastructure pods to be ready:

```bash
kubectl wait --for=condition=ready pod -l tier=database -n rag-system --timeout=300s
kubectl wait --for=condition=ready pod -l tier=cache -n rag-system --timeout=300s
kubectl wait --for=condition=ready pod -l tier=messaging -n rag-system --timeout=300s
kubectl wait --for=condition=ready pod -l tier=vectordb -n rag-system --timeout=300s
kubectl wait --for=condition=ready pod -l tier=storage -n rag-system --timeout=300s
```

### 7. Deploy Application Services

```bash
# Deploy all application services
kubectl apply -f deployments/query-analysis.yaml
kubectl apply -f deployments/document-retrieval.yaml
kubectl apply -f deployments/document-ranking.yaml
kubectl apply -f deployments/latex-parser.yaml
kubectl apply -f deployments/llm-generation.yaml
kubectl apply -f deployments/response-formatter.yaml

# Deploy corresponding services
kubectl apply -f services/query-analysis-service.yaml
kubectl apply -f services/document-retrieval-service.yaml
kubectl apply -f services/document-ranking-service.yaml
kubectl apply -f services/latex-parser-service.yaml
kubectl apply -f services/llm-generation-service.yaml
kubectl apply -f services/response-formatter-service.yaml
```

Wait for application pods:

```bash
kubectl wait --for=condition=ready pod -l tier=application -n rag-system --timeout=300s
```

### 8. Deploy API Gateway

```bash
kubectl apply -f deployments/api-gateway.yaml
kubectl apply -f services/api-gateway-service.yaml
```

### 9. Configure Ingress (Optional)

If using Ingress:

```bash
# Update api-gateway-ingress.yaml with your domain
kubectl apply -f ingress/api-gateway-ingress.yaml
```

### 10. Enable Autoscaling

```bash
kubectl apply -f hpa/
```

## Verification

### Check Pod Status

```bash
kubectl get pods -n rag-system
```

All pods should be in `Running` state.

### Check Services

```bash
kubectl get svc -n rag-system
```

### Check Ingress

```bash
kubectl get ingress -n rag-system
```

### Test API Gateway

Get the external IP or LoadBalancer endpoint:

```bash
kubectl get svc api-gateway-service -n rag-system
```

Test the health endpoint:

```bash
# If using LoadBalancer
curl http://<EXTERNAL-IP>/api/v1/health

# If using Ingress
curl https://api.example.com/api/v1/health
```

### View Logs

```bash
# API Gateway logs
kubectl logs -f deployment/api-gateway -n rag-system

# All logs
kubectl logs -f -l tier=application -n rag-system --max-log-requests=10
```

## Scaling

### Manual Scaling

```bash
# Scale API Gateway to 5 replicas
kubectl scale deployment api-gateway --replicas=5 -n rag-system

# Scale all application services
kubectl scale deployment -l tier=application --replicas=5 -n rag-system
```

### Autoscaling

HPA manifests are in `hpa/` directory. They automatically scale based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

View HPA status:

```bash
kubectl get hpa -n rag-system
```

## Updates and Rollouts

### Rolling Update

```bash
# Update image
kubectl set image deployment/api-gateway \
  api-gateway=your-registry.example.com/rag-system/api-gateway:v1.1.0 \
  -n rag-system

# Check rollout status
kubectl rollout status deployment/api-gateway -n rag-system
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/api-gateway -n rag-system

# Rollback to specific revision
kubectl rollout undo deployment/api-gateway --to-revision=2 -n rag-system
```

### View Rollout History

```bash
kubectl rollout history deployment/api-gateway -n rag-system
```

## Monitoring

### Resource Usage

```bash
# CPU and memory usage
kubectl top pods -n rag-system
kubectl top nodes
```

### Events

```bash
# Watch events
kubectl get events -n rag-system --watch

# Events for specific deployment
kubectl describe deployment api-gateway -n rag-system
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod
kubectl describe pod <pod-name> -n rag-system

# Check logs
kubectl logs <pod-name> -n rag-system

# Check previous logs if crashed
kubectl logs <pod-name> -n rag-system --previous
```

### Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints -n rag-system

# Test service from within cluster
kubectl run test-pod --image=curlimages/curl -i --tty --rm -n rag-system -- sh
# Then: curl http://api-gateway-service:80/api/v1/health
```

### Database Connection Issues

```bash
# Exec into postgres pod
kubectl exec -it deployment/postgres -n rag-system -- psql -U raguser -d ragdb

# Test from application pod
kubectl exec -it deployment/api-gateway -n rag-system -- sh
# Then: nc -zv postgres-service 5432
```

### Secret Issues

```bash
# List secrets
kubectl get secrets -n rag-system

# Describe secret (base64 encoded values)
kubectl get secret rag-app-secrets -n rag-system -o yaml

# Decode a secret value
kubectl get secret rag-app-secrets -n rag-system -o jsonpath='{.data.jwt-secret-key}' | base64 -d
```

## Cleanup

### Delete All Resources

```bash
# Delete all resources in namespace
kubectl delete all --all -n rag-system

# Delete PVCs (WARNING: This deletes data!)
kubectl delete pvc --all -n rag-system

# Delete secrets and configmaps
kubectl delete secrets --all -n rag-system
kubectl delete configmaps --all -n rag-system

# Delete namespace
kubectl delete namespace rag-system
```

## Production Considerations

### Security

1. **Enable Network Policies**:
   ```bash
   kubectl apply -f network-policies/
   ```

2. **Use Pod Security Standards**:
   - Set pod security policies
   - Run containers as non-root
   - Use read-only root filesystems

3. **Secrets Management**:
   - Use External Secrets Operator
   - Integrate with HashiCorp Vault
   - Or use cloud provider secrets manager

4. **RBAC**:
   - Create service accounts
   - Define roles and role bindings
   - Follow principle of least privilege

### High Availability

1. **Multi-zone deployment**:
   - Spread pods across availability zones
   - Use pod anti-affinity rules

2. **Database HA**:
   - Use managed database services (AWS RDS, Cloud SQL)
   - Or deploy PostgreSQL with replication

3. **Redis Cluster**:
   - Deploy Redis in cluster mode
   - Or use managed service (ElastiCache, Cloud Memorystore)

### Backups

1. **Velero** for cluster backups:
   ```bash
   velero backup create rag-system-backup --include-namespaces rag-system
   ```

2. **Database backups**:
   - Use CronJob for pg_dump
   - Store in S3/GCS

3. **Volume Snapshots**:
   - Use CSI snapshot controller
   - Schedule regular snapshots

### Monitoring

1. **Prometheus + Grafana**:
   - Deploy Prometheus Operator
   - Configure ServiceMonitors
   - Create dashboards

2. **Logging**:
   - Deploy Fluentd/Fluent Bit
   - Ship logs to ELK or Loki
   - Set up log retention policies

3. **Tracing**:
   - Deploy Jaeger
   - Configure services for OpenTelemetry

## Environment-Specific Configurations

### Development

```bash
# Lower resource limits
# Disable autoscaling
# Use NodePort services
# Single replica
```

### Staging

```bash
# Moderate resources
# Enable autoscaling
# Use LoadBalancer
# 2-3 replicas
```

### Production

```bash
# Full resources
# Enable autoscaling
# Use Ingress with TLS
# 3+ replicas
# Multi-zone deployment
```

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager](https://cert-manager.io/)
- [Helm](https://helm.sh/)
- [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator)
