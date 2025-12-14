"""
Pydantic models for JunRAG data structures.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ChunkMetadata(BaseModel):
    """Metadata for a retrieved chunk."""

    dataset: Optional[str] = Field(default=None, description="Dataset name")
    item_id: Optional[str] = Field(default=None, description="Item ID")
    source: Optional[str] = Field(default=None, description="Source identifier")
    markdown_file: Optional[str] = Field(default=None, description="Markdown file path")

    class Config:
        """Pydantic config."""

        extra = "allow"


class Chunk(BaseModel):
    """A retrieved chunk with metadata and scores."""

    rank: Optional[int] = Field(default=None, description="Rank in results")
    chunk_id: Optional[str] = Field(default=None, description="Chunk identifier")
    retrieval_score: Optional[float] = Field(
        default=None, description="Retrieval similarity score"
    )
    reranker_score: Optional[float] = Field(
        default=None, description="Reranker relevance score"
    )
    text: str = Field(default="", description="Chunk text content")
    metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata, description="Chunk metadata"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Raw payload from vector DB"
    )
    source_subquery: Optional[str] = Field(
        default=None, description="Source sub-query (for parallel pipeline)"
    )
    final_rank: Optional[int] = Field(
        default=None, description="Final rank after merging (for parallel pipeline)"
    )

    class Config:
        """Pydantic config."""

        extra = "allow"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""

    model: str = Field(
        default="jinaai/jina-embeddings-v3", description="Embedding model name"
    )
    tensor_parallel_size: int = Field(
        default=1, ge=1, description="Number of GPUs for tensor parallelism"
    )
    gpu_memory_utilization: float = Field(
        default=0.9, gt=0.0, le=1.0, description="GPU memory fraction"
    )
    max_model_len: int = Field(
        default=8192, gt=0, description="Maximum sequence length"
    )

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"


class RerankerConfig(BaseModel):
    """Configuration for reranker model."""

    model: str = Field(
        default="Qwen/Qwen3-Reranker-4B", description="Reranker model name"
    )
    device: Optional[str] = Field(default=None, description="Device (cuda/cpu)")
    use_multi_gpu: bool = Field(
        default=True, description="Use DataParallel for transformers backend"
    )
    batch_size: int = Field(default=64, gt=0, description="Batch size for inference")
    tensor_parallel_size: int = Field(
        default=1, ge=1, description="Number of GPUs for vLLM"
    )
    gpu_memory_utilization: float = Field(
        default=0.8, gt=0.0, le=1.0, description="GPU memory fraction for vLLM"
    )
    max_model_len: int = Field(
        default=8192, gt=0, description="Maximum sequence length"
    )
    instruction: Optional[str] = Field(
        default=None, description="Custom instruction for Qwen3 reranker"
    )

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"


class GenerationConfig(BaseModel):
    """Configuration for LLM generation."""

    model: str = Field(default="gpt-5.1-2025-11-13", description="OpenAI model name")
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Generation temperature"
    )
    max_tokens: int = Field(default=128, gt=0, description="Maximum tokens in response")
    reasoning_effort: str = Field(
        default="medium",
        description="Reasoning effort for reasoning models (low, medium, high)",
    )

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, v: str) -> str:
        """Validate reasoning effort value."""
        valid_values = {"low", "medium", "high"}
        if v not in valid_values:
            raise ValueError(f"reasoning_effort must be one of {valid_values}, got {v}")
        return v

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"


class DecompositionConfig(BaseModel):
    """Configuration for query decomposition."""

    model: str = Field(default="gpt-4o", description="OpenAI model name")
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Generation temperature"
    )
    max_tokens: int = Field(default=500, gt=0, description="Maximum tokens in response")

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"


class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""

    collection_name: str = Field(description="Qdrant collection name")
    url: Optional[str] = Field(default=None, description="Qdrant server URL")
    api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    top_k: int = Field(default=10, gt=0, description="Number of results to return")
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum score threshold"
    )
    filter_conditions: Optional[Dict[str, Any]] = Field(
        default=None, description="Filter conditions dict"
    )

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens used")
    completion_tokens: int = Field(
        default=0, ge=0, description="Completion tokens used"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"


class GenerationResult(BaseModel):
    """Result from LLM generation."""

    answer: str = Field(description="Generated answer")
    model: str = Field(description="Model used for generation")
    usage: UsageInfo = Field(description="Token usage information")

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"


class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""

    embedding_model: str = Field(description="Embedding model name")
    reranker_model: str = Field(description="Reranker model name")
    llm_model: str = Field(description="LLM model name")
    retrieval_top_k: int = Field(gt=0, description="Retrieval top_k")
    rerank_top_k: int = Field(gt=0, description="Rerank top_k")
    decomposition_model: Optional[str] = Field(
        default=None, description="Decomposition model name"
    )
    max_cap: Optional[int] = Field(default=None, gt=0, description="Maximum chunks")
    min_floor: Optional[int] = Field(default=None, gt=0, description="Minimum chunks")
    chunks_per_subquery: Optional[int] = Field(
        default=None, gt=0, description="Chunks per sub-query"
    )
    max_concurrent_retrievals: Optional[int] = Field(
        default=None, gt=0, description="Max concurrent retrievals"
    )
    tensor_parallel_size: Optional[int] = Field(
        default=None, ge=1, description="Tensor parallel size"
    )

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"


class PipelineResult(BaseModel):
    """Result from pipeline execution."""

    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer")
    pipeline: str = Field(
        description="Pipeline type (naive/parallel/sequential_decomposition/webui)"
    )
    config: PipelineConfig = Field(description="Pipeline configuration")
    usage: UsageInfo = Field(description="Token usage information")
    retrieved_chunks: Optional[List[Chunk]] = Field(
        default=None, description="Retrieved chunks (naive pipeline)"
    )
    reranked_chunks: Optional[List[Chunk]] = Field(
        default=None, description="Reranked chunks (naive pipeline)"
    )
    sub_queries: Optional[List[str]] = Field(
        default=None,
        description="Decomposed sub-queries (parallel/sequential_decomposition pipeline)",
    )
    original_sub_queries: Optional[List[str]] = Field(
        default=None,
        description="Original sub-queries with placeholders (sequential_decomposition pipeline)",
    )
    n_subqueries: Optional[int] = Field(
        default=None,
        description="Number of sub-queries (parallel/sequential_decomposition pipeline)",
    )
    n_chains: Optional[int] = Field(
        default=None,
        description="Number of independent sub-query chains (sequential_decomposition pipeline)",
    )
    chains: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Per-chain execution details (sequential_decomposition pipeline)",
    )
    used_internal_knowledge: Optional[bool] = Field(
        default=None,
        description="Whether internal knowledge was used at least once (sequential_decomposition pipeline)",
    )
    dynamic_k_initial: Optional[int] = Field(
        default=None, description="Initial dynamic K (parallel pipeline)"
    )
    dynamic_k_final: Optional[int] = Field(
        default=None, description="Final dynamic K (parallel pipeline)"
    )
    n_unique_chunks: Optional[int] = Field(
        default=None, description="Number of unique chunks (parallel pipeline)"
    )
    retrieved_per_subquery: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Retrieved chunks per sub-query (parallel/sequential_decomposition pipeline)",
    )
    reranked_per_subquery: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Reranked chunks per sub-query (parallel/sequential_decomposition pipeline)",
    )
    final_chunks: Optional[List[Chunk]] = Field(
        default=None,
        description="Final selected chunks (parallel/sequential_decomposition pipeline)",
    )

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "allow"
