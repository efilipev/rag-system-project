#!/bin/bash

# PostgreSQL Backup Script
# This script creates backups of the RAG system PostgreSQL database

set -e

# Configuration
NAMESPACE="${NAMESPACE:-rag-system}"
BACKUP_DIR="${BACKUP_DIR:-/backup/postgres}"
S3_BUCKET="${S3_BUCKET:-rag-backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="postgres_backup_${TIMESTAMP}.sql.gz"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting PostgreSQL backup...${NC}"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Get PostgreSQL pod name
POSTGRES_POD=$(kubectl get pod -n $NAMESPACE -l app=postgres -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POSTGRES_POD" ]; then
    echo -e "${RED}Error: PostgreSQL pod not found${NC}"
    exit 1
fi

echo -e "${YELLOW}PostgreSQL pod: $POSTGRES_POD${NC}"

# Create backup
echo -e "${YELLOW}Creating database dump...${NC}"
kubectl exec -n $NAMESPACE $POSTGRES_POD -- \
    sh -c 'pg_dump -U $POSTGRES_USER $POSTGRES_DB' | \
    gzip > "${BACKUP_DIR}/${BACKUP_FILE}"

# Verify backup
if [ ! -f "${BACKUP_DIR}/${BACKUP_FILE}" ]; then
    echo -e "${RED}Error: Backup file not created${NC}"
    exit 1
fi

BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_FILE}" | cut -f1)
echo -e "${GREEN}Backup created: ${BACKUP_FILE} (${BACKUP_SIZE})${NC}"

# Upload to S3 (if S3_BUCKET is set and aws CLI is available)
if command -v aws &> /dev/null && [ -n "$S3_BUCKET" ]; then
    echo -e "${YELLOW}Uploading to S3: s3://${S3_BUCKET}/postgres/${NC}"
    aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}" "s3://${S3_BUCKET}/postgres/${BACKUP_FILE}"
    echo -e "${GREEN}Upload complete${NC}"
fi

# Clean up old backups (local)
echo -e "${YELLOW}Cleaning up backups older than ${RETENTION_DAYS} days...${NC}"
find "$BACKUP_DIR" -name "postgres_backup_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete

# Clean up old backups (S3)
if command -v aws &> /dev/null && [ -n "$S3_BUCKET" ]; then
    RETENTION_DATE=$(date -d "${RETENTION_DAYS} days ago" +%Y-%m-%d)
    aws s3 ls "s3://${S3_BUCKET}/postgres/" | while read -r line; do
        CREATE_DATE=$(echo $line | awk '{print $1}')
        FILE_NAME=$(echo $line | awk '{print $4}')
        if [[ "$CREATE_DATE" < "$RETENTION_DATE" ]]; then
            aws s3 rm "s3://${S3_BUCKET}/postgres/${FILE_NAME}"
            echo -e "${YELLOW}Deleted old backup: ${FILE_NAME}${NC}"
        fi
    done
fi

echo -e "${GREEN}PostgreSQL backup completed successfully!${NC}"
echo -e "${GREEN}Backup file: ${BACKUP_DIR}/${BACKUP_FILE}${NC}"

# Create backup metadata
cat > "${BACKUP_DIR}/${BACKUP_FILE}.meta" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "backup_file": "${BACKUP_FILE}",
  "size": "${BACKUP_SIZE}",
  "namespace": "${NAMESPACE}",
  "pod": "${POSTGRES_POD}",
  "retention_days": ${RETENTION_DAYS}
}
EOF

echo -e "${GREEN}Backup metadata saved${NC}"
