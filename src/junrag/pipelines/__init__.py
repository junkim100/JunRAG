"""
Pipeline implementations for JunRAG.

Pipelines:
- NaivePipeline: Simple linear flow (Query → Retrieve → Rerank → Generate)
- ParallelPipeline: Advanced flow with decomposition and parallel processing
"""

from junrag.pipelines.naive import NaivePipeline
from junrag.pipelines.parallel import ParallelPipeline
from junrag.pipelines.webui import WebUIPipeline

__all__ = ["NaivePipeline", "ParallelPipeline", "WebUIPipeline"]
