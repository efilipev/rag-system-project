#!/bin/bash

# Setup Ollama for RAG System
# This script pulls the required LLM model into Ollama

set -e

echo "Setting up Ollama for RAG System..."

# Wait for Ollama to be ready
echo "Waiting for Ollama service to be ready..."
until curl -sf http://localhost:11434/api/tags > /dev/null; do
    echo "   Waiting for Ollama..."
    sleep 5
done

echo "[OK] Ollama is ready!"

# Pull the model (llama3.2:3b - small, fast, good quality)
echo "Pulling Llama 3.2 3B model (this may take a few minutes)..."
docker exec rag-ollama ollama pull llama3.2:3b

echo "[OK] Model pulled successfully!"

# Verify the model is available
echo "Verifying model..."
docker exec rag-ollama ollama list

echo ""
echo "[OK] Ollama setup complete!"
echo "   Model: llama3.2:3b"
echo "   API: http://localhost:11434"
echo ""
echo "You can now use the RAG System with local LLM!"
