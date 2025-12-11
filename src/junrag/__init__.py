"""
JunRAG - A modular RAG library with query decomposition support.
"""

__version__ = "0.1.0"

from junrag.config import Config
from junrag.pipelines import NaivePipeline, ParallelPipeline

__all__ = ["Config", "NaivePipeline", "ParallelPipeline", "__version__"]
