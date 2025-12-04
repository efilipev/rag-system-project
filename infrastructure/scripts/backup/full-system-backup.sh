#!/bin/bash

# Full RAG System Backup Script
# This script creates a complete backup of all RAG system components

set -e

# Configuration
NAMESPACE="${NAMESPACE:-rag-system}"
BACKUP_DIR="${BACKUP_DIR:-/backup/rag-system}"
S3_BUCKET="${S3_BUCKET:-rag-backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="rag_system_backup_${TIMESTAMP}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}RAG System Full Backup${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${YELLOW}Timestamp: ${TIMESTAMP}${NC}"
echo -e "${YELLOW}Namespace: ${NAMESPACE}${NC}"
echo -e "${YELLOW}Backup Directory: ${BACKUP_DIR}/${BACKUP_NAME}${NC}"
echo ""

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# 1. Backup PostgreSQL
echo -e "${GREEN}[1/6] Backing up PostgreSQL...${NC}"
export BACKUP_DIR="${BACKUP_DIR}/${BACKUP_NAME}/postgres"
mkdir -p "$BACKUP_DIR"
bash "$(dirname $0)/postgres-backup.sh"

# 2. Backup Qdrant
echo -e "${GREEN}[2/6] Backing up Qdrant...${NC}"
export BACKUP_DIR="${BACKUP_DIR}/${BACKUP_NAME}/qdrant"
mkdir -p "$BACKUP_DIR"
bash "$(dirname $0)/qdrant-backup.sh"

# 3. Backup Redis (if needed)
echo -e "${GREEN}[3/6] Backing up Redis...${NC}"
REDIS_POD=$(kubectl get pod -n $NAMESPACE -l app=redis -o jsonpath='{.items[0].metadata.name}')
if [ -n "$REDIS_POD" ]; then
    kubectl exec -n $NAMESPACE $REDIS_POD -- redis-cli BGSAVE
    sleep 5
    kubectl cp "$NAMESPACE/$REDIS_POD:/data/dump.rdb" "${BACKUP_DIR}/${BACKUP_NAME}/redis/dump.rdb"
    echo -e "${GREEN}Redis backup complete${NC}"
else
    echo -e "${YELLOW}Redis pod not found, skipping${NC}"
fi

# 4. Backup Kubernetes Resources
echo -e "${GREEN}[4/6] Backing up Kubernetes resources...${NC}"
K8S_BACKUP_DIR="${BACKUP_DIR}/${BACKUP_NAME}/kubernetes"
mkdir -p "$K8S_BACKUP_DIR"

# Export deployments
kubectl get deployments -n $NAMESPACE -o yaml > "${K8S_BACKUP_DIR}/deployments.yaml"

# Export services
kubectl get services -n $NAMESPACE -o yaml > "${K8S_BACKUP_DIR}/services.yaml"

# Export configmaps
kubectl get configmaps -n $NAMESPACE -o yaml > "${K8S_BACKUP_DIR}/configmaps.yaml"

# Export PVCs
kubectl get pvc -n $NAMESPACE -o yaml > "${K8S_BACKUP_DIR}/pvcs.yaml"

# Export ingress
kubectl get ingress -n $NAMESPACE -o yaml > "${K8S_BACKUP_DIR}/ingress.yaml" 2>/dev/null || true

# Export HPAs
kubectl get hpa -n $NAMESPACE -o yaml > "${K8S_BACKUP_DIR}/hpa.yaml" 2>/dev/null || true

echo -e "${GREEN}Kubernetes resources backed up${NC}"

# 5. Backup MinIO/S3 (if applicable)
echo -e "${GREEN}[5/6] Backing up MinIO...${NC}"
MINIO_BACKUP_DIR="${BACKUP_DIR}/${BACKUP_NAME}/minio"
mkdir -p "$MINIO_BACKUP_DIR"

# This would require mc (MinIO client) to be installed
# mc mirror minio/rag-documents ${MINIO_BACKUP_DIR}/
echo -e "${YELLOW}MinIO backup requires manual intervention or mc CLI${NC}"

# 6. Create backup manifest
echo -e "${GREEN}[6/6] Creating backup manifest...${NC}"
cat > "${BACKUP_DIR}/${BACKUP_NAME}/manifest.json" <<EOF
{
  "backup_name": "${BACKUP_NAME}",
  "timestamp": "${TIMESTAMP}",
  "namespace": "${NAMESPACE}",
  "components": {
    "postgres": true,
    "qdrant": true,
    "redis": $([ -n "$REDIS_POD" ] && echo "true" || echo "false"),
    "kubernetes": true,
    "minio": false
  },
  "retention_policy": {
    "daily": 7,
    "weekly": 4,
    "monthly": 12
  }
}
EOF

# Create tarball
echo -e "${GREEN}Creating backup archive...${NC}"
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}/"

BACKUP_SIZE=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)
echo -e "${GREEN}Backup archive created: ${BACKUP_NAME}.tar.gz (${BACKUP_SIZE})${NC}"

# Upload to S3
if command -v aws &> /dev/null && [ -n "$S3_BUCKET" ]; then
    echo -e "${YELLOW}Uploading to S3...${NC}"
    aws s3 cp "${BACKUP_NAME}.tar.gz" "s3://${S3_BUCKET}/full-backups/${BACKUP_NAME}.tar.gz"
    echo -e "${GREEN}Upload complete${NC}"
fi

# Clean up temporary directory
rm -rf "${BACKUP_NAME}/"

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Full Backup Completed Successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Backup file: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz${NC}"
echo -e "${GREEN}Size: ${BACKUP_SIZE}${NC}"
echo ""
