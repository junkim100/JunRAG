#!/bin/bash
# Start script for JunRAG Web UI

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Load environment variables from .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Check for required environment variables (warnings only, .env may have them)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set (check .env file)"
fi

if [ -z "$QDRANT_API_KEY" ]; then
    echo "Warning: QDRANT_API_KEY not set (check .env file)"
fi

# Check GPU availability
uv run python -c "
import torch
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
if num_gpus < 8:
    print(f'ERROR: WebUIPipeline requires 8 GPUs, but only {num_gpus} are available.')
    exit(1)
print(f'✓ Found {num_gpus} GPUs')
"

if [ $? -ne 0 ]; then
    echo "GPU check failed. Exiting."
    exit 1
fi

# Start the Gradio server
cd "$PROJECT_ROOT"
uv run "$SCRIPT_DIR/app.py"
