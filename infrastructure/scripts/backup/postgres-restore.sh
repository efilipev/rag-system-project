#!/bin/bash

# PostgreSQL Restore Script
# This script restores the RAG system PostgreSQL database from a backup

set -e

# Configuration
NAMESPACE="${NAMESPACE:-rag-system}"
BACKUP_DIR="${BACKUP_DIR:-/backup/postgres}"
S3_BUCKET="${S3_BUCKET:-rag-backups}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if backup file is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Backup file not specified${NC}"
    echo "Usage: $0 <backup_file>"
    echo ""
    echo "Available backups:"
    ls -lh "$BACKUP_DIR"/postgres_backup_*.sql.gz 2>/dev/null || echo "No local backups found"
    exit 1
fi

BACKUP_FILE="$1"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

# Download from S3 if not found locally
if [ ! -f "$BACKUP_PATH" ] && command -v aws &> /dev/null && [ -n "$S3_BUCKET" ]; then
    echo -e "${YELLOW}Backup not found locally. Downloading from S3...${NC}"
    aws s3 cp "s3://${S3_BUCKET}/postgres/${BACKUP_FILE}" "$BACKUP_PATH"
fi

# Verify backup file exists
if [ ! -f "$BACKUP_PATH" ]; then
    echo -e "${RED}Error: Backup file not found: $BACKUP_PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}WARNING: This will restore the database and overwrite all existing data!${NC}"
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo -e "${YELLOW}Restore cancelled${NC}"
    exit 0
fi

echo -e "${GREEN}Starting PostgreSQL restore...${NC}"

# Get PostgreSQL pod name
POSTGRES_POD=$(kubectl get pod -n $NAMESPACE -l app=postgres -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POSTGRES_POD" ]; then
    echo -e "${RED}Error: PostgreSQL pod not found${NC}"
    exit 1
fi

echo -e "${YELLOW}PostgreSQL pod: $POSTGRES_POD${NC}"

# Terminate existing connections
echo -e "${YELLOW}Terminating existing database connections...${NC}"
kubectl exec -n $NAMESPACE $POSTGRES_POD -- \
    sh -c 'psql -U $POSTGRES_USER -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '\''$POSTGRES_DB'\'' AND pid <> pg_backend_pid();"'

# Drop and recreate database
echo -e "${YELLOW}Dropping existing database...${NC}"
kubectl exec -n $NAMESPACE $POSTGRES_POD -- \
    sh -c 'psql -U $POSTGRES_USER -d postgres -c "DROP DATABASE IF EXISTS $POSTGRES_DB;"'

echo -e "${YELLOW}Creating new database...${NC}"
kubectl exec -n $NAMESPACE $POSTGRES_POD -- \
    sh -c 'psql -U $POSTGRES_USER -d postgres -c "CREATE DATABASE $POSTGRES_DB;"'

# Restore database
echo -e "${YELLOW}Restoring database from backup...${NC}"
gunzip -c "$BACKUP_PATH" | kubectl exec -i -n $NAMESPACE $POSTGRES_POD -- \
    sh -c 'psql -U $POSTGRES_USER $POSTGRES_DB'

echo -e "${GREEN}PostgreSQL restore completed successfully!${NC}"

# Verify restore
echo -e "${YELLOW}Verifying restore...${NC}"
TABLE_COUNT=$(kubectl exec -n $NAMESPACE $POSTGRES_POD -- \
    sh -c 'psql -U $POSTGRES_USER $POSTGRES_DB -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '\''public'\'';"')

echo -e "${GREEN}Tables restored: ${TABLE_COUNT}${NC}"
echo -e "${GREEN}Restore verification complete${NC}"
