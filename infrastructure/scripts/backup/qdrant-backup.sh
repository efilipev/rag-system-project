#!/bin/bash

# Qdrant Backup Script
# This script creates snapshots of Qdrant collections

set -e

# Configuration
NAMESPACE="${NAMESPACE:-rag-system}"
BACKUP_DIR="${BACKUP_DIR:-/backup/qdrant}"
S3_BUCKET="${S3_BUCKET:-rag-backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
QDRANT_URL="${QDRANT_URL:-http://qdrant-service:6333}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting Qdrant backup...${NC}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Get Qdrant pod name
QDRANT_POD=$(kubectl get pod -n $NAMESPACE -l app=qdrant -o jsonpath='{.items[0].metadata.name}')

if [ -z "$QDRANT_POD" ]; then
    echo -e "${RED}Error: Qdrant pod not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Qdrant pod: $QDRANT_POD${NC}"

# Get list of collections
echo -e "${YELLOW}Fetching collections list...${NC}"
COLLECTIONS=$(kubectl exec -n $NAMESPACE $QDRANT_POD -- \
    sh -c "curl -s ${QDRANT_URL}/collections | jq -r '.result.collections[].name'")

if [ -z "$COLLECTIONS" ]; then
    echo -e "${YELLOW}No collections found${NC}"
    exit 0
fi

echo -e "${GREEN}Collections found: $(echo $COLLECTIONS | wc -w)${NC}"

# Create snapshot for each collection
for COLLECTION in $COLLECTIONS; do
    echo -e "${YELLOW}Creating snapshot for collection: ${COLLECTION}${NC}"

    # Create snapshot via API
    SNAPSHOT_NAME=$(kubectl exec -n $NAMESPACE $QDRANT_POD -- \
        sh -c "curl -s -X POST ${QDRANT_URL}/collections/${COLLECTION}/snapshots | jq -r '.result.name'")

    echo -e "${GREEN}Snapshot created: ${SNAPSHOT_NAME}${NC}"

    # Download snapshot
    BACKUP_FILE="${BACKUP_DIR}/${COLLECTION}_${TIMESTAMP}.snapshot"
    kubectl exec -n $NAMESPACE $QDRANT_POD -- \
        cat "/qdrant/storage/snapshots/${COLLECTION}/${SNAPSHOT_NAME}" > "$BACKUP_FILE"

    echo -e "${GREEN}Snapshot downloaded: ${BACKUP_FILE}${NC}"

    # Upload to S3
    if command -v aws &> /dev/null && [ -n "$S3_BUCKET" ]; then
        aws s3 cp "$BACKUP_FILE" "s3://${S3_BUCKET}/qdrant/${COLLECTION}_${TIMESTAMP}.snapshot"
        echo -e "${GREEN}Uploaded to S3${NC}"
    fi

    # Delete snapshot from Qdrant
    kubectl exec -n $NAMESPACE $QDRANT_POD -- \
        sh -c "curl -s -X DELETE ${QDRANT_URL}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}"
done

echo -e "${GREEN}Qdrant backup completed successfully!${NC}"
