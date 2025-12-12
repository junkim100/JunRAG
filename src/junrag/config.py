"""
Configuration management for JunRAG.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Configuration container for JunRAG."""

    # Qdrant settings
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant server URL"
    )
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")

    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # Model settings
    embedding_model: str = Field(
        default="jinaai/jina-embeddings-v3", description="Embedding model name"
    )
    reranker_model: str = Field(
        default="Qwen/Qwen3-Reranker-4B", description="Reranker model name"
    )
    llm_model: str = Field(
        default="gpt-5.1-2025-11-13", description="LLM model for generation"
    )
    decomposition_model: str = Field(
        default="gpt-5-mini-2025-08-07", description="Model for query decomposition"
    )

    # Embedding settings
    tensor_parallel_size: int = Field(
        default=1, ge=1, description="Number of GPUs for tensor parallelism"
    )
    gpu_memory_utilization: float = Field(
        default=0.9, gt=0.0, le=1.0, description="GPU memory fraction to use"
    )
    max_model_len: int = Field(
        default=8192, gt=0, description="Maximum sequence length"
    )

    # Retrieval settings
    retrieval_top_k: int = Field(
        default=20, gt=0, description="Number of chunks to retrieve"
    )
    rerank_top_k: int = Field(
        default=5, gt=0, description="Number of chunks after reranking"
    )
    rerank_batch_size: int = Field(default=64, gt=0, description="Reranking batch size")
    use_multi_gpu: bool = Field(default=True, description="Use multi-GPU for reranking")

    # Parallel pipeline settings
    max_cap: int = Field(
        default=25, gt=0, description="Maximum chunks for parallel pipeline"
    )
    min_floor: int = Field(
        default=5, gt=0, description="Minimum chunks for parallel pipeline"
    )
    chunks_per_subquery: int = Field(
        default=10, gt=0, description="Chunks per sub-query after reranking (M)"
    )

    # Generation settings
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="LLM temperature"
    )
    max_tokens: int = Field(
        default=16384, gt=0, description="Maximum tokens in response"
    )
    reasoning_effort: str = Field(
        default="medium",
        description="Reasoning effort for reasoning models (low, medium, high)",
    )

    # Concurrency settings
    max_concurrent_retrievals: int = Field(
        default=5, gt=0, description="Max parallel Qdrant retrievals"
    )
    max_concurrent_reranks: int = Field(
        default=3, gt=0, description="Max parallel reranks"
    )

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, v: str) -> str:
        """Validate reasoning effort value."""
        valid_values = {"low", "medium", "high"}
        if v not in valid_values:
            raise ValueError(f"reasoning_effort must be one of {valid_values}, got {v}")
        return v

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    class Config:
        """Pydantic config."""

        frozen = False
        extra = "forbid"


def get_config(env_file: Optional[str] = None) -> Config:
    """Get configuration instance."""
    return Config.from_env(env_file)
