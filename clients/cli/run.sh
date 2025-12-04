#!/bin/bash
# Run the RAG CLI client

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Installing dependencies..."
    poetry install --no-interaction
fi

# Run the CLI
./.venv/bin/python rag_cli.py "$@"
