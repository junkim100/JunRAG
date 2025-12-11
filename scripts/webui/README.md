# JunRAG Web UI

A Gradio-based web interface for JunRAG with pre-loaded models for optimal performance.

## Requirements

- **8 GPUs** (exactly 8 required)
  - GPUs 0-3: Embedders (one per GPU)
  - GPUs 4-7: Rerankers (one per GPU)
- Python 3.10+
- All dependencies from `pyproject.toml`
- Environment variables:
  - `OPENAI_API_KEY`: For query decomposition and final generation
  - `QDRANT_API_KEY`: For Qdrant access

## Installation

```bash
# Install dependencies (including gradio)
uv sync

# Or with pip
pip install -e ".[dev]"
pip install gradio>=4.0.0
```

## Usage

### Start the Server

```bash
# Using the startup script
./scripts/webui/start_server.sh

# Or directly with Python
python scripts/webui/app.py
```

The server will start on `http://0.0.0.0:7860`

### Using the UI

1. **Initialize Pipeline**:
   - Configure model settings in the sidebar
   - Click "Initialize Pipeline" to pre-load all models (this may take a few minutes)

2. **Configure Settings**:
   - Select metadata file from dropdown
   - Adjust retrieval and reranking parameters
   - Set reasoning effort level

3. **Ask Questions**:
   - Type your query in the chat input
   - Click "Send" or press Enter
   - View the answer with timing information

## Features

- **Pre-loaded Models**: All embedders and rerankers are loaded at startup for fast inference
- **Settings Sidebar**: Configure all pipeline parameters without restarting
- **Metadata Selection**: Choose from available metadata files
- **Timing Information**: See detailed timing breakdown for each query
- **ChatGPT-like Interface**: Clean, modern UI similar to ChatGPT/Gemini

## GPU Allocation

The pipeline requires exactly 8 GPUs:
- **GPUs 0-3**: Embedding models (one model per GPU)
- **GPUs 4-7**: Reranking models (one model per GPU)

This ensures:
- No GPU conflicts
- Maximum parallelism
- Fast inference (no model loading during queries)

## Troubleshooting

### "WebUIPipeline requires exactly 8 GPUs"

Make sure you have 8 GPUs available:
```bash
nvidia-smi
```

### Models not loading

Check GPU memory:
```bash
nvidia-smi
```

Reduce `gpu_memory_utilization` in settings if needed.

### Metadata files not found

Ensure metadata files are in `data/metadata/` directory.

