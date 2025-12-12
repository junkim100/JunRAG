"""
Pipeline implementations for JunRAG.

Pipelines:
- NaivePipeline: Simple linear flow (Query → Retrieve → Rerank → Generate)
- ParallelPipeline: Advanced flow with decomposition and parallel processing
- SequentialDecompositionPipeline: Sequential flow with placeholder-based decomposition
"""

from junrag.pipelines.naive import NaivePipeline
from junrag.pipelines.parallel import ParallelPipeline
from junrag.pipelines.webui import WebUIPipeline
from junrag.pipelines.sequential_decomposition import SequentialDecompositionPipeline

__all__ = [
    "NaivePipeline",
    "ParallelPipeline",
    "WebUIPipeline",
    "SequentialDecompositionPipeline",
]
