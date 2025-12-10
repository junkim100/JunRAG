# JunRAG

A modular RAG (Retrieval-Augmented Generation) library with query decomposition support for complex multi-hop questions.

## Features

- **Query Decomposition**: Break complex multi-hop queries into single-hop sub-queries using GPT-4o
- **Parallel Processing**: Concurrent retrieval and reranking with semaphore-based rate limiting
- **vLLM-Powered**: Embedding and reranking both use vLLM for high-throughput inference
- **Dynamic Top-K Selection**: Adaptive chunk selection based on query complexity
- **Multiple Pipelines**: Naive (simple) and Full (advanced) pipeline implementations
- **Modular Components**: Use building blocks independently or as complete pipelines

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
```

---

## Section 1: Creating the Vector Database

### Quick Start: Prepare All Collections

Run the master script to create all 4 collections (test_late, test_contextual, val_late, val_contextual):

```bash
uv run tools/prepare_all.py --qdrant_url https://your-qdrant.cloud.qdrant.io --tensor_parallel_size 8
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

JunRAG provides two pipelines for different use cases.

### Pipeline Overview

| Pipeline | Use Case | Flow |
|----------|----------|------|
| **Naive** | Simple, single-hop questions | Query → Retrieve → Rerank → Generate |
| **Full** | Complex, multi-hop questions | Query → Decompose → Parallel Retrieve → Parallel Rerank → Dynamic Top-K → Generate |

---

### 2.1 Naive Pipeline

```bash
uv run junrag naive \
    --query "What is machine learning?" \
    --metadata_path data/metadata/test_late_metadata.json \
    --tensor_parallel_size 8
```

#### Python API

```python
from junrag.pipelines import NaivePipeline

pipeline = NaivePipeline(
    metadata_path="data/metadata/test_late_metadata.json",
    tensor_parallel_size=8,
)

result = pipeline.run("What is machine learning?")
print(result["answer"])
```

---

### 2.2 Full Pipeline (with Query Decomposition)

```bash
uv run junrag full \
    --query "What was the GDP of the country where Einstein was born in 1950?" \
    --metadata_path data/metadata/test_late_metadata.json \
    --tensor_parallel_size 8
```

### Dynamic K: Two-Stage Calculation

The full pipeline uses a two-stage approach to select the optimal number of chunks:

**Stage 1: Calculate initial K (accounting for expected overlap)**
```
K_initial = min(MAX_CAP, max(MIN_FLOOR, N_sub × chunks_per_subquery × 1.4))
```
- The `1.4` overlap factor accounts for ~30% expected duplicate chunks across sub-queries

**Stage 2: Adjust after deduplication**
- After merging and deduplicating chunks, if `n_unique_chunks < K_initial`, use all unique chunks
- Otherwise, use `K_initial` chunks

Where:
- `MAX_CAP = 25` (maximum chunks for final selection)
- `MIN_FLOOR = 5` (minimum chunks for final selection)
- `N_sub` = number of sub-queries
- `chunks_per_subquery = 10` (number of reranked chunks per sub-query)

#### Python API

```python
from junrag.pipelines import FullPipeline

pipeline = FullPipeline(
    metadata_path="data/metadata/test_late_metadata.json",
    tensor_parallel_size=8,
    max_concurrent_retrievals=5,
    max_concurrent_reranks=3,
)

result = pipeline.run("What was the GDP of the country where Einstein was born in 1950?")
print(f"Sub-queries: {result['sub_queries']}")
print(f"Answer: {result['answer']}")
```

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
decomposer = QueryDecomposer(model="gpt-4o")
sub_queries = decomposer.decompose("Complex multi-hop question")

# 2. Embed queries (with vLLM)
embedder = EmbeddingModel(model="jinaai/jina-embeddings-v3", tensor_parallel_size=8)
query_embedding = embedder.embed_query("Simple query")

# 3. Retrieve relevant chunks from your Qdrant collection
chunks = retrieve_chunks(query_embedding=query_embedding, collection_name="test_late", top_k=20)

# 4. Rerank retrieved chunks (optional, with vLLM or transformers)
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
│   │   └── generation.py           # LLM generation (OpenAI)
│   ├── pipelines/                  # Pipeline implementations
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base pipeline
│   │   ├── naive.py                # Simple linear pipeline
│   │   └── full.py                 # Advanced pipeline with decomposition
│   └── cli/                        # Command-line interface
│       ├── __init__.py
│       └── pipeline.py             # CLI entry point
│
├── tools/                          # Data preparation scripts
│   ├── prepare_all.py              # Master script (runs entire pipeline)
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
├── scripts/
│   └── run_pipeline.py             # Standalone script
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
│       └── ...
│
├── pyproject.toml                  # Package configuration
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
