# RAG System Helm Chart

A Helm chart for deploying the RAG (Retrieval-Augmented Generation) System to Kubernetes.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.8+
- PersistentVolume provisioner support in the underlying infrastructure
- Container images for all services pushed to a registry

## Installing the Chart

### Quick Start

```bash
# Add your values
cp values.yaml my-values.yaml
# Edit my-values.yaml with your configuration

# Install the chart
helm install rag-system . -f my-values.yaml --namespace rag-system --create-namespace
```

### With Custom Values

```bash
helm install rag-system . \
  --namespace rag-system \
  --create-namespace \
  --set secrets.openaiApiKey=sk-your-key \
  --set secrets.jwtSecretKey=your-secret-key \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=api.example.com
```

## Uninstalling the Chart

```bash
helm uninstall rag-system --namespace rag-system
```

## Configuration

The following table lists the configurable parameters of the RAG System chart and their default values.

### Global Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imageRegistry` | Container image registry | `docker.io` |
| `global.storageClass` | Storage class for PVs | `standard` |
| `global.namespace` | Kubernetes namespace | `rag-system` |

### Application Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `config.logLevel` | Log level | `INFO` |
| `config.embeddingModel` | Embedding model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `config.llm.provider` | LLM provider | `openai` |
| `config.llm.defaultModel` | Default LLM model | `gpt-3.5-turbo` |

### Secrets

**IMPORTANT**: Change all default secret values in production!

| Parameter | Description | Default |
|-----------|-------------|---------|
| `secrets.postgres.password` | PostgreSQL password | `changeme` |
| `secrets.jwtSecretKey` | JWT secret key | `changeme` |
| `secrets.openaiApiKey` | OpenAI API key | `sk-changeme` |

### Service Configuration

Each service has the following configurable parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `<service>.enabled` | Enable/disable service | `true` |
| `<service>.replicaCount` | Number of replicas | varies |
| `<service>.image.repository` | Image repository | varies |
| `<service>.image.tag` | Image tag | `latest` |
| `<service>.resources.requests.memory` | Memory request | varies |
| `<service>.resources.limits.cpu` | CPU limit | varies |

### Ingress

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable Ingress | `false` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.hosts[0].host` | Hostname | `api.example.com` |

## Examples

### Development Environment

```bash
helm install rag-system . \
  --namespace rag-system-dev \
  --create-namespace \
  --set apiGateway.replicaCount=1 \
  --set queryAnalysis.replicaCount=1 \
  --set documentRetrieval.replicaCount=1 \
  --set config.security.enableAuthentication=false \
  --set ingress.enabled=false
```

### Production Environment

```bash
helm install rag-system . \
  --namespace rag-system-prod \
  --create-namespace \
  -f production-values.yaml \
  --set secrets.jwtSecretKey=$JWT_SECRET \
  --set secrets.openaiApiKey=$OPENAI_KEY \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=api.prod.example.com
```

### Staging Environment

```bash
helm install rag-system . \
  --namespace rag-system-staging \
  --create-namespace \
  -f staging-values.yaml
```

## Upgrading

### Update Configuration

```bash
helm upgrade rag-system . \
  --namespace rag-system \
  -f my-values.yaml \
  --reuse-values
```

### Update Images

```bash
helm upgrade rag-system . \
  --namespace rag-system \
  --set apiGateway.image.tag=v1.1.0 \
  --set queryAnalysis.image.tag=v1.1.0
```

### Rollback

```bash
# List releases
helm history rag-system --namespace rag-system

# Rollback to revision 1
helm rollback rag-system 1 --namespace rag-system
```

## Monitoring

Check deployment status:

```bash
# Get all resources
helm status rag-system --namespace rag-system

# Watch pods
kubectl get pods -n rag-system --watch

# Check HPA status
kubectl get hpa -n rag-system

# View logs
kubectl logs -f deployment/api-gateway -n rag-system
```

## Troubleshooting

### Pods not starting

```bash
# Describe pod
kubectl describe pod <pod-name> -n rag-system

# Check events
kubectl get events -n rag-system --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n rag-system
```

### Service not accessible

```bash
# Test from inside cluster
kubectl run test --rm -i --tty --image=curlimages/curl -n rag-system -- sh
# Then: curl http://api-gateway-service/api/v1/health

# Port forward for local testing
kubectl port-forward svc/api-gateway-service 8000:80 -n rag-system
```

### Storage issues

```bash
# Check PVC status
kubectl get pvc -n rag-system

# Describe PVC
kubectl describe pvc postgres-pvc -n rag-system
```

## Custom Values Files

Create environment-specific values files:

### development-values.yaml

```yaml
config:
  logLevel: DEBUG
  security:
    enableAuthentication: false
    enableRateLimiting: false

apiGateway:
  replicaCount: 1
  autoscaling:
    enabled: false

queryAnalysis:
  replicaCount: 1
  autoscaling:
    enabled: false
```

### production-values.yaml

```yaml
config:
  logLevel: INFO
  security:
    enableAuthentication: true
    enableRateLimiting: true
  cors:
    allowedOrigins:
      - "https://app.prod.example.com"

apiGateway:
  replicaCount: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10

ingress:
  enabled: true
  hosts:
    - host: api.prod.example.com
      paths:
        - path: /
          pathType: Prefix
```

## Security Best Practices

1. **Never commit secrets to git**
2. **Use external secrets manager** (HashiCorp Vault, AWS Secrets Manager)
3. **Enable RBAC** and create service accounts
4. **Use network policies** to restrict pod communication
5. **Enable Pod Security Standards**
6. **Scan images** for vulnerabilities
7. **Use TLS/SSL** for all external communication

## Support

For issues and questions:
- Check the main project README
- Review Kubernetes deployment logs
- Consult the Helm documentation

## License

[Add your license here]
