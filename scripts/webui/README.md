# JunRAG Web UI

A Gradio-based web interface for JunRAG using the Sequential Decomposition Pipeline.

## Requirements

- **GPUs**: Configurable (1-8 GPUs supported)
  - GPUs are split between retriever (embedders) and reranker
  - For `tensor_parallel_size=8`: 4 GPUs for embedders, 4 GPUs for rerankers
  - For `tensor_parallel_size=4`: 2 GPUs for embedders, 2 GPUs for rerankers
  - Minimum: 1 GPU (will be split between retriever and reranker)
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
uv pip install -e ".[dev]"
```

## Usage

### Start the Server

```bash
# Using the startup script
./scripts/webui/start_server.sh

# Or directly with Python
uv run scripts/webui/app.py
```

The server will start on `http://0.0.0.0:7860`

### Using the UI

1. **Initialize Pipeline**:
   - Configure model settings in the sidebar
   - Set tensor parallel size (number of GPUs to use)
   - Click "Initialize Pipeline" to initialize the pipeline (models load on first query)

2. **Configure Settings**:
   - Select metadata file from dropdown
   - Adjust retrieval and reranking parameters
   - Set reasoning effort level

3. **Ask Questions**:
   - Type your query in the chat input
   - Click "Send" or press Enter
   - View the answer, decomposition chains, and timing information

## Features

- **Sequential Decomposition**: Uses sequential decomposition pipeline with placeholder-based query processing
- **Decomposition Chains Display**: Shows all subqueries, rewritten queries, and intermediate answers
- **Settings Sidebar**: Configure all pipeline parameters without restarting
- **Metadata Selection**: Choose from available metadata files
- **Timing Information**: See detailed timing breakdown for each query
- **ChatGPT-like Interface**: Clean, modern UI similar to ChatGPT/Gemini
- **Flexible GPU Configuration**: Works with 1-8 GPUs (configurable)

## GPU Allocation

The pipeline splits available GPUs between retriever and reranker:

- **Retriever GPUs**: Floor of `tensor_parallel_size / 2`
- **Reranker GPUs**: Ceiling of `tensor_parallel_size / 2` (priority for odd counts)

Examples:

- `tensor_parallel_size=8`: 4 GPUs for embedders, 4 GPUs for rerankers
- `tensor_parallel_size=4`: 2 GPUs for embedders, 2 GPUs for rerankers
- `tensor_parallel_size=1`: 0 GPUs for embedders, 1 GPU for rerankers (minimum 1 each)

Models are loaded on first query and kept in memory for subsequent queries.

## Troubleshooting

### Models not loading

Check GPU memory:

```bash
nvidia-smi
```

Reduce `gpu_memory_utilization` in settings if needed.

### Metadata files not found

Ensure metadata files are in `data/metadata/` directory.

### GPU memory errors

Reduce `tensor_parallel_size` or `gpu_memory_utilization` in settings.
