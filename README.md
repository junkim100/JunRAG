# JunRAG

A modular RAG (Retrieval-Augmented Generation) library with query decomposition support for complex multi-hop questions.

## Features

- **Query Decomposition**: Break complex multi-hop queries into single-hop sub-queries using GPT-4o
- **Sequential Decomposition**: Placeholder-based sequential query processing with answer chaining
- **Parallel Processing**: Concurrent embedding, retrieval, and reranking with per-subquery GPU assignment
- **vLLM-Powered**: Embedding and reranking both use vLLM for high-throughput inference
- **Dynamic Top-K Selection**: Two-stage adaptive chunk selection based on query complexity and overlap
- **Per-Subquery GPU Assignment**: Each sub-query uses a dedicated GPU for embedding and reranking
- **Multiple Pipelines**: Naive, Parallel, Sequential Decomposition, and WebUI pipeline implementations
- **Web UI**: Gradio-based interface with ChatGPT-like UI, multiple pipeline modes, and settings sidebar
- **Smart Model Loading**: Models load on first query and persist across queries for efficiency
- **Modular Components**: Use building blocks independently or as complete pipelines
- **NVML Error Handling**: Robust handling of GPU initialization issues with automatic workarounds

## Supported Models

| Component | Default Model | Backend | Alternatives |
|-----------|--------------|---------|--------------|
| Embedding | `jinaai/jina-embeddings-v3` | vLLM | Any pooling model |
| Reranking | `Qwen/Qwen3-Reranker-4B` | vLLM | `Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen3-Reranker-8B`, `jinaai/jina-reranker-v3` (transformers) |
| Generation | `gpt-5.1-2025-11-13` | OpenAI API | Any OpenAI model (supports reasoning models) |
| Decomposition | `gpt-4o` | OpenAI API | Any OpenAI model |

## Installation

```bash
cd junrag
uv sync --extra tools  # Include tools dependencies
uv pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
# Required
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key

# Optional
HF_TOKEN=your_huggingface_token
UPSTAGE_API_KEY=your_upstage_api_key  # For evaluation script (optional)
UPSTAGE_BASE_URL=https://api.upstage.ai/v1  # For evaluation script (optional)
```

## GPU Usage

JunRAG supports flexible GPU configurations:

- **Naive Pipeline**: Uses `tensor_parallel_size` for tensor parallelism (splitting a single model across multiple GPUs)
- **Parallel Pipeline**: Uses `tensor_parallel_size` to specify the number of GPUs available for per-subquery assignment (each sub-query gets a dedicated GPU)
- **Sequential Decomposition Pipeline**: Splits `tensor_parallel_size` GPUs between retriever (floor) and reranker (ceil, priority for odd counts)
- **WebUI Pipeline**: Configurable GPUs (1-8 GPUs, split between retriever and reranker)

**Important:** vLLM is imported at module level to avoid NVML initialization errors when `CUDA_VISIBLE_DEVICES` is set. If you encounter GPU-related errors, ensure:
- NVIDIA drivers are properly installed
- `nvidia-smi` works correctly
- You have sufficient GPU memory for your models

---

## Section 1: Creating the Vector Database

### Quick Start: Prepare All Collections

Run the master script to create all 4 collections (test_late, test_contextual, val_late, val_contextual):

```bash
uv run tools/prepare_vectordb.py --qdrant_url https://your-qdrant.cloud.qdrant.io --tensor_parallel_size 8
```

This runs the entire pipeline:
1. Download FRAMES dataset → `data/dataset/test.json`, `data/dataset/val.json`
2. Fetch Wikipedia articles → `data/markdown/*.md` (with deduplication)
3. Create embeddings → `data/embeddings/*.json`
4. Upload to Qdrant → `data/metadata/*.json`

### Step-by-Step Guide

#### Step 1: Download Dataset

```bash
uv run tools/dataset/download_dataset.py
```

**Output:**
- `data/dataset/test.json` (100 Q&A items with wiki_links)
- `data/dataset/val.json` (100 Q&A items with wiki_links)

#### Step 2: Fetch Wikipedia Content

```bash
uv run tools/preprocessing/fetch_wikipedia.py --max_workers 20
```

**Output:**
- `data/markdown/test_canonical_*.md` (unique Wikipedia articles, deduplicated)
- `data/markdown/test_url_mapping.json` (maps item_id_article_idx to canonical files)
- `data/markdown/val_canonical_*.md` (unique Wikipedia articles, deduplicated)
- `data/markdown/val_url_mapping.json` (maps item_id_article_idx to canonical files)

#### Step 3: Create Embeddings

Markdown files are used directly for embedding

Choose between **Late Chunking** or **Contextual Embedding**:

```bash
# Late chunking (splits documents into chunks, preserves cross-chunk context)
uv run tools/embedding/embed_late.py --dataset test --tensor_parallel_size 8
uv run tools/embedding/embed_late.py --dataset val --tensor_parallel_size 8

# Contextual embedding (one embedding per document)
uv run tools/embedding/embed_contextual.py --dataset test --tensor_parallel_size 8
uv run tools/embedding/embed_contextual.py --dataset val --tensor_parallel_size 8
```

**Output:**
- `data/embeddings/test_late.json`
- `data/embeddings/test_contextual.json`
- `data/embeddings/val_late.json`
- `data/embeddings/val_contextual.json`

#### Step 4: Upload to Qdrant

```bash
uv run tools/qdrant/upload_to_qdrant.py --qdrant_url https://your-qdrant.cloud.qdrant.io --dataset test --embedding_type late --recreate
uv run tools/qdrant/upload_to_qdrant.py --qdrant_url https://your-qdrant.cloud.qdrant.io --dataset test --embedding_type contextual --recreate
uv run tools/qdrant/upload_to_qdrant.py --qdrant_url https://your-qdrant.cloud.qdrant.io --dataset val --embedding_type late --recreate
uv run tools/qdrant/upload_to_qdrant.py --qdrant_url https://your-qdrant.cloud.qdrant.io --dataset val --embedding_type contextual --recreate
```

**Output:**
- `data/metadata/test_late_metadata.json`
- `data/metadata/test_contextual_metadata.json`
- `data/metadata/val_late_metadata.json`
- `data/metadata/val_contextual_metadata.json`

---

## Section 2: Running Pipelines

JunRAG provides multiple pipelines for different use cases.

### Pipeline Overview

| Pipeline | Use Case | Flow | GPU Requirements |
|----------|----------|------|------------------|
| **Naive** | Simple, single-hop questions | Query → Retrieve → Rerank → Generate | Configurable (1-8 GPUs for tensor parallelism) |
| **Parallel** | Complex, multi-hop questions | Query → Decompose → Parallel Embed → Parallel Retrieve+Rerank → Dynamic Top-K → Generate | Configurable (1-8 GPUs for per-subquery assignment) |
| **Sequential Decomposition** | Complex, multi-hop questions with answer dependencies | Query → Decompose → [Sequential: Embed → Retrieve → Rerank → Rewrite] → Generate | Configurable (split between retriever and reranker) |
| **WebUI** | Web interface with sequential decomposition | Query → Decompose → [Sequential: Embed → Retrieve → Rerank → Rewrite] → Generate | Configurable (1-8 GPUs, split between retriever and reranker) |

---

### 2.1 Naive Pipeline

The naive pipeline is designed for simple, single-hop questions. It processes queries sequentially without decomposition.

```bash
uv run junrag naive \
    --query "What is machine learning?" \
    --metadata_path data/metadata/test_late_metadata.json \
    --tensor_parallel_size 8
```

**Note:** `tensor_parallel_size` here refers to tensor parallelism within a single model (splitting a model across multiple GPUs).

#### Python API

```python
from junrag.pipelines import NaivePipeline

pipeline = NaivePipeline(
    metadata_path="data/metadata/test_late_metadata.json",
    tensor_parallel_size=8,  # Tensor parallelism: model split across 8 GPUs
)

result = pipeline.run("What is machine learning?")
print(result.answer)
```

---

### 2.2 Parallel Pipeline (with Query Decomposition)

The parallel pipeline is designed for complex multi-hop questions. It decomposes queries into sub-queries and processes them in parallel with dedicated GPU assignment.

**GPU Assignment Strategy:**
- Each sub-query is assigned a dedicated GPU from the available pool
- `tensor_parallel_size` specifies the number of GPUs available for assignment
- GPUs are assigned round-robin: subquery 1 → GPU 0, subquery 2 → GPU 1, etc.
- If you have more sub-queries than GPUs, GPUs are reused in round-robin fashion
- Each sub-query uses its assigned GPU for both embedding and reranking

**Example:**
```bash
uv run junrag parallel \
    --query "What was the GDP of the country where Einstein was born in 1950?" \
    --metadata_path data/metadata/test_late_metadata.json \
    --tensor_parallel_size 8 \
    --retrieval_top_k 20
```

**Pipeline Flow:**
1. **Decompose** query into single-hop sub-queries (using GPT-4o)
2. **Pre-initialize** models on assigned GPUs in parallel
3. **Embed** all sub-queries in parallel (each on its assigned GPU)
4. **Retrieve + Rerank** chunks for each sub-query in parallel
   - Each subquery: retrieve → rerank sequentially on assigned GPU
   - Different subqueries run in parallel, maximizing GPU utilization
   - Semaphore limits concurrent Qdrant calls to prevent server overload
5. **Merge and select** dynamic top-K chunks (two-stage approach)
6. **Generate** final answer using LLM

### Dynamic K: Two-Stage Calculation

The parallel pipeline uses a two-stage approach to select the optimal number of chunks:

**Stage 1: Calculate initial K (accounting for expected overlap)**
```
K_initial = min(MAX_CAP, max(MIN_FLOOR, N_sub × chunks_per_subquery × 1.4))
```
- The `1.4` overlap factor accounts for ~30% expected duplicate chunks across sub-queries

**Stage 2: Adjust after deduplication**
- After merging and deduplicating chunks, if `n_unique_chunks < K_initial`, use all unique chunks
- Otherwise, use `K_initial` chunks

Where:
- `MAX_CAP = 25` (maximum chunks for final selection, default)
- `MIN_FLOOR = 5` (minimum chunks for final selection, default)
- `N_sub` = number of sub-queries
- `chunks_per_subquery = 10` (number of reranked chunks per sub-query, default)

#### Python API

```python
from junrag.pipelines import ParallelPipeline

pipeline = ParallelPipeline(
    metadata_path="data/metadata/test_late_metadata.json",
    tensor_parallel_size=8,  # Number of GPUs available for per-subquery assignment
    max_concurrent_retrievals=8,  # Max parallel Qdrant retrieval operations
    chunks_per_subquery=10,  # Chunks per sub-query after reranking
    max_cap=25,  # Maximum chunks for final selection
    min_floor=5,  # Minimum chunks for final selection
)

result = pipeline.run("What was the GDP of the country where Einstein was born in 1950?")
print(f"Sub-queries: {result.sub_queries}")
print(f"Answer: {result.answer}")
print(f"Dynamic K: {result.dynamic_k_final} chunks selected from {result.n_unique_chunks} unique chunks")
```

#### Key Parameters

- `tensor_parallel_size`: Number of GPUs available for per-subquery assignment (not tensor parallelism within a model)
- `retrieval_top_k`: Number of chunks to retrieve per sub-query (default: 50)
- `chunks_per_subquery`: Number of chunks per sub-query after reranking (default: 10)
- `max_concurrent_retrievals`: Maximum parallel Qdrant retrieval operations (default: 8)
- `max_cap`: Maximum chunks for final selection (default: 25)
- `min_floor`: Minimum chunks for final selection (default: 5)

---

### 2.3 Sequential Decomposition Pipeline

The sequential decomposition pipeline is designed for complex multi-hop questions where sub-queries depend on answers from previous sub-queries. Unlike the parallel pipeline, it processes sub-queries sequentially, using placeholders that are filled with actual answers from retrieval.

**Key Features:**
- **Placeholder-based decomposition**: Sub-queries (except the first) contain `[answer]` placeholders referencing previous answers
- **Sequential processing**: Each sub-query is processed one at a time, with placeholders filled before retrieval
- **GPU split strategy**: Retriever and reranker share GPUs (retriever gets floor, reranker gets ceil with priority for odd counts)
- **Model reuse**: Models are loaded once and reused across all sequential sub-queries

**Example:**
```bash
uv run junrag sequential_decomposition \
    --query "As of August 1, 2024, which country were holders of the FIFA World Cup the last time the UEFA Champions League was won by a club from London?" \
    --metadata_path data/metadata/test_late_metadata.json \
    --tensor_parallel_size 8 \
    --retrieval_top_k 50
```

**Pipeline Flow:**
1. **Decompose** query into sequential sub-queries with `[answer]` placeholders
   - First sub-query: No placeholder (self-contained)
   - Subsequent sub-queries: Contain `[answer]` placeholder for previous answer
2. **Load models** once (retriever and reranker with GPU split)
3. **For each sub-query sequentially:**
   - If sub-query has `[answer]` placeholder: Rewrite it using answer from previous retrieval
   - Embed the sub-query
   - Retrieve chunks from vector DB
   - Rerank chunks
   - Extract answer for next sub-query (if not last)
4. **Generate** final answer using all accumulated context

**GPU Split Strategy:**
- For `tensor_parallel_size=8`: Retriever gets 4 GPUs, Reranker gets 4 GPUs
- For `tensor_parallel_size=7`: Retriever gets 3 GPUs, Reranker gets 4 GPUs (priority to reranker)
- Models are loaded once and reused for all sub-queries

**Reranker GPU Assignment:**
- Reranker GPU allocation depends on `n_chains` (number of sequential chains):
  - `n_chains=1`: Uses all reranker GPUs with `tensor_parallel_size=4` (e.g., GPUs 4,5,6,7)
  - `n_chains=2`: Splits into 2 rerankers with `tensor_parallel_size=2` each (e.g., GPUs 4,5 and 6,7)
  - `n_chains>2`: Round-robin assignment when chains exceed available reranker GPUs
- **Smart Conflict Detection**: During evaluation, rerankers are kept loaded when possible and only cleaned up when GPU conflicts occur (e.g., switching from `n_chains=2` to `n_chains=1` requires all GPUs)
- **Model Persistence**: Use `cleanup_models=False` in `pipeline.run()` to keep models loaded across multiple queries (useful for evaluation)

**Example Decomposition:**
```
Original Query: "Who was older, the guitar player for the Dugites from 1982-1983 or the lead singer of The Sports?"

Decomposed Sub-queries:
1. "Who was the guitar player for the Dugites from 1982-1983?"
2. "When was [answer] born?"  # [answer] = answer to query 1
3. "Who was the lead singer of The Sports?"
4. "When was [answer] born?"  # [answer] = answer to query 3
```

#### Python API

```python
from junrag.pipelines import SequentialDecompositionPipeline

pipeline = SequentialDecompositionPipeline(
    metadata_path="data/metadata/test_late_metadata.json",
    tensor_parallel_size=8,  # Split between retriever and reranker
    rerank_per_subquery=10,  # Chunks per sub-query after reranking
    decomposition_model="gpt-4o",  # Model for decomposition and answer extraction
)

result = pipeline.run("Complex multi-hop question with dependencies")
print(f"Original sub-queries: {result.original_sub_queries}")
print(f"Rewritten sub-queries: {result.sub_queries}")
print(f"Answer: {result.answer}")
```

#### Key Parameters

- `tensor_parallel_size`: Total GPUs (split between retriever and reranker)
- `retrieval_top_k`: Number of chunks to retrieve per sub-query (default: 50)
- `rerank_per_subquery`: Number of chunks per sub-query after reranking (default: 10)
- `decomposition_model`: Model for query decomposition and answer extraction (default: "gpt-4o")
- `cleanup_models`: Whether to cleanup models after each query (default: `True`). Set to `False` for evaluation to keep models loaded across multiple queries

#### When to Use Sequential Decomposition vs Parallel

- **Use Sequential Decomposition** when:
  - Sub-queries depend on answers from previous sub-queries
  - You want to use actual retrieved answers to refine subsequent queries
  - You have limited GPUs and want to maximize model reuse

- **Use Parallel** when:
  - Sub-queries are independent and can be processed simultaneously
  - You want maximum throughput with parallel processing
  - You have enough GPUs for per-subquery assignment

---

### 2.4 Web UI Pipeline (Gradio Interface)

The WebUI uses the Sequential Decomposition Pipeline with a modern web interface.

**Requirements:**
- **GPUs**: Configurable (1-8 GPUs supported)
  - GPUs are split between retriever (embedders) and reranker
  - For `tensor_parallel_size=8`: 4 GPUs for embedders, 4 GPUs for rerankers
  - For `tensor_parallel_size=4`: 2 GPUs for embedders, 2 GPUs for rerankers
  - Minimum: 1 GPU (will be split between retriever and reranker)
- Models load automatically on first query

**Start the Web UI:**

```bash
# Using the startup script
./scripts/webui/start_server.sh

# Or directly
uv run scripts/webui/app.py
```

The server will start on `http://0.0.0.0:7860`

**Features:**
- Clean, modern chat interface with conversation history
- **Sequential Decomposition**: Uses sequential decomposition pipeline with placeholder-based query processing
- **Decomposition Chains Display**: Shows all subqueries, rewritten queries, and intermediate answers
- **Settings Sidebar**: Configure all pipeline parameters without restarting
  - Model configuration (embedding, reranker, LLM, decomposition models)
  - Tensor parallel size (number of GPUs to use)
  - Metadata file selection (reads from `data/metadata/`)
  - Query settings (retrieval_top_k, rerank_per_subquery, reasoning_effort)
- **Timing Information**: Detailed timing breakdown for each query
- **Real-time Configuration**: Change settings and metadata without restarting

**Usage:**
1. Initialize Pipeline: Configure model settings and click "Initialize Pipeline" (models load on first query)
2. Configure Settings: Select metadata file and adjust parameters
3. Ask Questions: Type queries and get answers with decomposition chains and timing information

See `scripts/webui/README.md` for detailed documentation.

#### Python API

The WebUI uses `SequentialDecompositionPipeline` internally. You can use it directly:

```python
from junrag.pipelines import SequentialDecompositionPipeline

pipeline = SequentialDecompositionPipeline(
    embedding_model="jinaai/jina-embeddings-v3",
    reranker_model="Qwen/Qwen3-Reranker-4B",
    llm_model="gpt-5.1-2025-11-13",
    decomposition_model="gpt-4o",
    tensor_parallel_size=8,  # Split between retriever and reranker
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    rerank_per_subquery=10,
)

# Run queries (models load automatically on first query)
result = pipeline.run("Complex multi-hop question")
print(result.answer)
print(f"Chains: {result.chains}")
```

---

## Section 3: Evaluation

JunRAG includes an evaluation script to test pipeline performance on datasets.

### Running Evaluation

```bash
uv run tools/evaluate.py \
    --pipeline sequential_decomposition \
    --decomposition_model gpt-4o \
    --tensor_parallel_size 8 \
    --metadata_path data/metadata/test_late_metadata.json
```

**Key Features:**
- **Model Persistence**: Models are kept loaded across evaluation items to avoid reload overhead
- **Smart Reranker Management**: Rerankers are only reloaded when GPU conflicts occur (e.g., when `n_chains` changes)
- **Comprehensive Metrics**: Tracks answer quality, `used_internal_knowledge` flags, and timing information
- **Batch Processing**: Processes entire datasets with progress tracking

**Note:** The evaluation script uses `UPSTAGE_API_KEY` and `UPSTAGE_BASE_URL` (optional) for answer evaluation. If not provided, evaluation will still work but may use a different evaluation method.

**Output:**
- Results saved to `results/evaluation_results_{pipeline}_{model}_{config}.json`
- Each result includes: question, answer, ground truth, evaluation metrics, `used_internal_knowledge`, and timing breakdown

**Evaluation Optimization:**
- The evaluation script automatically uses `cleanup_models=False` for the sequential decomposition pipeline
- This keeps embedders and rerankers loaded across items, significantly speeding up evaluation
- Rerankers are intelligently cleaned up only when GPU conflicts occur (e.g., switching from `n_chains=2` to `n_chains=1`)

---

## Build Your Own Pipeline

You can combine JunRAG's modular components to create a custom pipeline tailored to your needs:

```python
from junrag.components import (
    EmbeddingModel,
    retrieve_chunks,
    Reranker,
    QueryDecomposer,
    LLMGenerator,
)

# 1. Decompose queries (for complex multi-hop questions)
from junrag.components.decomposition import QueryDecomposer
from junrag.components.sequential_decomposition import SequentialQueryDecomposer

# Standard decomposition (independent sub-queries)
decomposer = QueryDecomposer(model="gpt-4o")
sub_queries = decomposer.decompose("Complex multi-hop question")

# Sequential decomposition (with placeholders)
seq_decomposer = SequentialQueryDecomposer(model="gpt-4o")
seq_sub_queries = seq_decomposer.decompose("Complex multi-hop question with dependencies")

# 2. Embed queries (with vLLM)
# Note: tensor_parallel_size here is for tensor parallelism (splitting model across GPUs)
embedder = EmbeddingModel(model="jinaai/jina-embeddings-v3", tensor_parallel_size=8)
query_embedding = embedder.embed_query("Simple query")

# 3. Retrieve relevant chunks from your Qdrant collection
chunks = retrieve_chunks(query_embedding=query_embedding, collection_name="test_late", top_k=20)

# 4. Rerank retrieved chunks (optional, with vLLM or transformers)
# Note: tensor_parallel_size here is for tensor parallelism (splitting model across GPUs)
reranker = Reranker(model="Qwen/Qwen3-Reranker-4B", tensor_parallel_size=8)
reranked = reranker.rerank("query", chunks, top_k=5)

# 5. Generate answer using an LLM
generator = LLMGenerator(model="gpt-4o")
result = generator.generate("query", reranked)
print(result["answer"])
```

Feel free to mix and match components or insert custom logic between stages to fit your use case!

---

## Project Structure

```
junrag/
├── src/junrag/                     # Library (pip installable)
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── py.typed                    # Type hint marker
│   ├── components/                 # RAG building blocks
│   │   ├── __init__.py
│   │   ├── embedding.py            # Query embedding (vLLM)
│   │   ├── retrieval.py            # Qdrant vector search
│   │   ├── reranking.py            # Reranking (vLLM/transformers)
│   │   ├── decomposition.py        # Query decomposition (OpenAI)
│   │   ├── sequential_decomposition.py  # Sequential decomposition with placeholders
│   │   └── generation.py           # LLM generation (OpenAI)
│   ├── pipelines/                  # Pipeline implementations
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base pipeline
│   │   ├── naive.py                # Simple linear pipeline
│   │   ├── parallel.py             # Advanced pipeline with decomposition
│   │   ├── sequential_decomposition.py  # Sequential pipeline with placeholder-based decomposition
│   │   └── webui.py                # Web UI pipeline with pre-loaded models
│   └── cli/                        # Command-line interface
│       ├── __init__.py
│       └── pipeline.py             # CLI entry point
│
├── tools/                          # Data preparation scripts
│   ├── prepare_vectordb.py         # Master script (runs entire pipeline)
│   ├── dataset/
│   │   └── download_dataset.py     # Download FRAMES dataset
│   ├── preprocessing/
│   │   └── fetch_wikipedia.py      # Fetch Wikipedia → markdown
│   ├── embedding/
│   │   ├── embed_late.py           # Late chunking embeddings
│   │   └── embed_contextual.py     # Contextual embeddings
│   └── qdrant/
│       └── upload_to_qdrant.py     # Upload to Qdrant
│
├── scripts/                        # Utility scripts
│   ├── run_pipeline.py             # Standalone script
│   └── webui/                      # Web UI scripts
│       ├── app.py                  # Gradio web interface
│       ├── start_server.sh         # Server startup script
│       └── README.md               # Web UI documentation
│
├── data/                           # Data files (gitignored)
│   ├── dataset/                    # Downloaded Q&A dataset
│   │   ├── test.json
│   │   └── val.json
│   ├── markdown/                   # Fetched Wikipedia articles (deduplicated)
│   │   ├── test_canonical_*.md     # Unique articles (one per URL)
│   │   ├── test_url_mapping.json   # Maps item_id_article_idx to canonical files
│   │   ├── val_canonical_*.md
│   │   └── val_url_mapping.json
│   ├── embeddings/                 # Embedded documents
│   │   ├── test_late.json
│   │   ├── test_contextual.json
│   │   └── ...
│   └── metadata/                   # Qdrant collection metadata
│       ├── test_late_metadata.json
│       ├── test_contextual_metadata.json
│       ├── val_late_metadata.json
│       └── val_contextual_metadata.json
│
├── results/                        # Pipeline results (gitignored)
│   └── {pipeline}_{timestamp}.json  # Auto-generated result files (e.g., naive_20251212_010640.json)
│
├── pyproject.toml                  # Package configuration
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
