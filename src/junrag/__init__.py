"""
JunRAG - A modular RAG library with query decomposition support.
"""

__version__ = "0.1.0"

from junrag.config import Config
from junrag.pipelines import NaivePipeline, FullPipeline

__all__ = ["Config", "NaivePipeline", "FullPipeline", "__version__"]
