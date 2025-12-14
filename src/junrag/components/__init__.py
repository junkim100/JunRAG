"""
RAG building blocks for JunRAG.

Components:
- embedding: Query embedding using vLLM (tensor parallelism supported)
- retrieval: Vector database retrieval from Qdrant
- reranking: Reranking with auto-detection:
    - Qwen3-Reranker: vLLM (generative logprob scoring)
    - Jina-Reranker: transformers (cross-encoder)
- decomposition: Query decomposition using OpenAI GPT-4o
- generation: LLM-based answer generation using OpenAI API
"""

from junrag.components.embedding import EmbeddingModel, embed_query
from junrag.components.retrieval import retrieve_chunks, get_qdrant_client
from junrag.components.reranking import Reranker, rerank_chunks
from junrag.components.decomposition import QueryDecomposer, decompose_query
from junrag.components.sequential_decomposition import (
    SequentialQueryDecomposer,
)
from junrag.components.generation import LLMGenerator, generate_answer

__all__ = [
    "EmbeddingModel",
    "embed_query",
    "retrieve_chunks",
    "get_qdrant_client",
    "Reranker",
    "rerank_chunks",
    "QueryDecomposer",
    "decompose_query",
    "SequentialQueryDecomposer",
    "LLMGenerator",
    "generate_answer",
]
