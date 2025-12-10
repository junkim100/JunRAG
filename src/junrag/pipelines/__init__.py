"""
Pipeline implementations for JunRAG.

Pipelines:
- NaivePipeline: Simple linear flow (Query → Retrieve → Rerank → Generate)
- FullPipeline: Advanced flow with decomposition and parallel processing
"""

from junrag.pipelines.naive import NaivePipeline
from junrag.pipelines.full import FullPipeline

__all__ = ["NaivePipeline", "FullPipeline"]
