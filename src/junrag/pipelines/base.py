"""
Base pipeline class for JunRAG.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

from dotenv import load_dotenv

from junrag.components.embedding import EmbeddingModel
from junrag.components.retrieval import load_metadata
from junrag.components.reranking import Reranker
from junrag.components.generation import LLMGenerator
from junrag.models import PipelineResult


class BasePipeline(ABC):
    """Abstract base class for RAG pipelines."""

    def __init__(
        self,
        # Qdrant settings
        metadata_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        # Model settings
        embedding_model: str = "jinaai/jina-embeddings-v3",
        reranker_model: str = "Qwen/Qwen3-Reranker-4B",
        llm_model: str = "gpt-5.1-2025-11-13",
        # Embedding settings
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        # Retrieval settings
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
        rerank_batch_size: int = 64,
        use_multi_gpu: bool = True,
        # Generation settings
        temperature: float = 0.1,
        max_tokens: int = 16384,
        reasoning_effort: str = "medium",
        openai_api_key: Optional[str] = None,
        # Environment
        env_file: Optional[str] = None,
    ):
        """Initialize base pipeline with common settings."""
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        # Store settings
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.llm_model_name = llm_model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.rerank_batch_size = rerank_batch_size
        self.use_multi_gpu = use_multi_gpu
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        # Load metadata if provided
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = load_metadata(metadata_path)

        # Resolve Qdrant settings
        self.collection_name = collection_name or self.metadata.get("collection_name")
        self.qdrant_url = (
            qdrant_url
            or self.metadata.get("qdrant_url")
            or os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        self.qdrant_api_key = (
            qdrant_api_key
            or self.metadata.get("qdrant_api_key")
            or os.getenv("QDRANT_API_KEY")
        )

        # OpenAI API key
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Lazy initialization
        self._embedding_model = None
        self._reranker = None
        self._generator = None

    @property
    def embedder(self) -> EmbeddingModel:
        """Lazy load embedding model (uses vLLM)."""
        if self._embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = EmbeddingModel(
                model=self.embedding_model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )
        return self._embedding_model

    @property
    def reranker(self) -> Reranker:
        """Lazy load reranker (vLLM for Qwen3, transformers for Jina)."""
        if self._reranker is None:
            self._reranker = Reranker(
                model=self.reranker_model_name,
                batch_size=self.rerank_batch_size,
                use_multi_gpu=self.use_multi_gpu,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )
        return self._reranker

    @property
    def generator(self) -> LLMGenerator:
        """Lazy load LLM generator (uses OpenAI API)."""
        if self._generator is None:
            self._generator = LLMGenerator(
                model=self.llm_model_name,
                api_key=self.openai_api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
        return self._generator

    @abstractmethod
    def run(self, query: str, **kwargs) -> PipelineResult:
        """Run the pipeline. Must be implemented by subclasses."""
        pass
