#!/bin/bash
set -e

echo "Waiting for Qdrant to be ready..."
for i in {1..30}; do
    if curl -f http://qdrant:6333/healthz > /dev/null 2>&1; then
        echo "Qdrant is ready!"
        break
    fi
    echo "Waiting for Qdrant... (attempt $i/30)"
    sleep 2
done

echo "Waiting for Chroma to be ready..."
# Chroma takes a while to start, just wait a fixed time
echo "Giving Chroma 10 seconds to initialize..."
sleep 10
echo "Chroma should be ready now!"

echo "Starting document-retrieval service..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
