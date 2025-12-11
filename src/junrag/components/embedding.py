"""
Embedding component for JunRAG.
Handles query and document embedding using vLLM.
"""

import logging
import os
from typing import List, Optional, Union

# Set spawn method for vLLM multiprocessing before torch import
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch

from junrag.models import EmbeddingConfig

# Configure logging
logger = logging.getLogger(__name__)

# Import vLLM at module level to avoid NVML errors during import
# when CUDA_VISIBLE_DEVICES is set later. vLLM's platform detection
# runs at import time and queries all GPUs, so we do this before
# any CUDA_VISIBLE_DEVICES manipulation.
try:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
        from vllm import LLM as vLLM_LLM
    _VLLM_AVAILABLE = True
except Exception as e:
    # If import fails, we'll handle it in __init__
    _VLLM_AVAILABLE = False
    _VLLM_IMPORT_ERROR = e
    vLLM_LLM = None


class EmbeddingModel:
    """vLLM-based embedding model wrapper."""

    def __init__(
        self,
        model: Union[str, EmbeddingConfig] = "jinaai/jina-embeddings-v3",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
    ):
        """
        Initialize embedding model.

        Args:
            model: Model name or path, or EmbeddingConfig instance
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
        """
        # Handle Pydantic config or individual parameters
        if isinstance(model, EmbeddingConfig):
            config = model
            self.model_name = config.model
            tensor_parallel_size = config.tensor_parallel_size
            gpu_memory_utilization = config.gpu_memory_utilization
            max_model_len = config.max_model_len
        else:
            self.model_name = model

        # Validate tensor_parallel_size
        # Check CUDA_VISIBLE_DEVICES to get the actual number of visible GPUs
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            # If CUDA_VISIBLE_DEVICES is set, count the number of GPUs specified
            visible_gpu_list = [
                gpu.strip() for gpu in cuda_visible.split(",") if gpu.strip()
            ]
            available_gpus = len(visible_gpu_list) if visible_gpu_list else 0
        else:
            available_gpus = (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            )

        if tensor_parallel_size > available_gpus:
            logger.warning(
                f"Requested tensor_parallel_size={tensor_parallel_size} but only "
                f"{available_gpus} GPUs available. Using {max(1, available_gpus)} instead."
            )
            tensor_parallel_size = max(1, available_gpus)

        if not torch.cuda.is_available():
            logger.warning(
                "CUDA not available. Embedding model will run on CPU (may be slow)."
            )

        # Validate gpu_memory_utilization
        if not 0.0 < gpu_memory_utilization <= 1.0:
            logger.warning(
                f"Invalid gpu_memory_utilization={gpu_memory_utilization}. "
                "Must be between 0 and 1. Using 0.9."
            )
            gpu_memory_utilization = 0.9

        # Initialize vLLM with error handling
        # vLLM is imported at module level to avoid NVML errors when CUDA_VISIBLE_DEVICES is set
        if not _VLLM_AVAILABLE:
            if _VLLM_IMPORT_ERROR:
                logger.error(
                    f"vLLM import failed at module level: {_VLLM_IMPORT_ERROR}"
                )
                raise ImportError(
                    "vLLM is required for embedding but failed to import. "
                    f"Original error: {_VLLM_IMPORT_ERROR}"
                ) from _VLLM_IMPORT_ERROR
            else:
                raise ImportError(
                    "vLLM is required for embedding. Install with: pip install vllm"
                )

        try:
            logger.info(f"Initializing embedding model: {model}")
            logger.info(f"  tensor_parallel_size: {tensor_parallel_size}")
            logger.info(f"  gpu_memory_utilization: {gpu_memory_utilization}")
            logger.info(f"  max_model_len: {max_model_len}")
            logger.debug(
                f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
            )

            # Use the module-level imported LLM class
            self.llm = vLLM_LLM(
                model=model,
                trust_remote_code=True,
                runner="pooling",
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            logger.info(f"Embedding model '{model}' loaded successfully")

        except ImportError as e:
            logger.error("vLLM not installed. Install with: pip install vllm")
            raise ImportError(
                "vLLM is required for embedding. Install with: pip install vllm"
            ) from e

        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"GPU out of memory loading embedding model. "
                f"Try reducing gpu_memory_utilization (current: {gpu_memory_utilization}) "
                f"or max_model_len (current: {max_model_len})."
            )
            raise MemoryError(
                f"GPU out of memory. Reduce gpu_memory_utilization or max_model_len."
            ) from e

        except Exception as e:
            error_msg = str(e).lower()
            error_type = type(e).__name__

            # Check for NVML-related errors during LLM initialization (not import)
            if (
                "nvml" in error_msg
                or "nvmlerror" in error_msg
                or "NVMLError" in error_type
            ):
                logger.warning(
                    f"NVML error during vLLM LLM initialization: {error_type}: {e}. "
                    "This may be non-fatal. vLLM should still work for inference."
                )
                # For now, we'll still raise the error as it indicates a real problem
                # But provide helpful context

            if "not found" in error_msg or "does not exist" in error_msg:
                logger.error(
                    f"Model '{model}' not found. Check model name or HuggingFace access."
                )
                raise ValueError(
                    f"Model '{model}' not found. Verify model name."
                ) from e
            elif "trust_remote_code" in error_msg:
                logger.error(
                    f"Model '{model}' requires trust_remote_code but failed to load."
                )
                raise
            else:
                # Log full exception details for debugging
                import traceback

                logger.error(
                    f"Failed to initialize embedding model: {error_type}: {e}\n"
                    f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise RuntimeError(
                    f"Failed to initialize embedding model: {error_type}: {e}"
                ) from e

    def _add_prefix(self, text: str, task: str) -> str:
        """Add task-specific prefix for Jina v3."""
        if "jina-embeddings-v3" not in self.model_name.lower():
            return text

        if task == "retrieval.query":
            return "Represent the query for retrieval: " + text
        elif task == "retrieval.passage":
            return "Represent the document for retrieval: " + text
        return text

    def embed_query(self, query: str, task: str = "retrieval.query") -> np.ndarray:
        """
        Embed a single query.

        Args:
            query: Query text
            task: Embedding task type

        Returns:
            Query embedding vector

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If embedding fails
        """
        # Validate input
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty or whitespace only")

        try:
            prefixed_query = self._add_prefix(query, task)
            outputs = self.llm.encode([prefixed_query], pooling_task="embed")

            if not outputs:
                raise RuntimeError("Embedding model returned empty output")

            output_data = outputs[0].outputs.data
            if hasattr(output_data, "cpu"):
                output_data = output_data.cpu()

            embedding = np.asarray(output_data).flatten()

            # Validate embedding
            if embedding.size == 0:
                raise RuntimeError("Generated embedding is empty")
            if np.isnan(embedding).any():
                logger.warning("Embedding contains NaN values")

            return embedding

        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU out of memory during embedding")
            raise MemoryError("GPU out of memory during embedding") from e

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Embedding failed: {e}") from e

    def embed_queries(
        self, queries: List[str], task: str = "retrieval.query"
    ) -> List[np.ndarray]:
        """
        Embed multiple queries.

        Args:
            queries: List of query texts
            task: Embedding task type

        Returns:
            List of query embedding vectors

        Raises:
            ValueError: If queries list is empty or contains invalid items
            RuntimeError: If embedding fails
        """
        # Validate input
        if not queries:
            raise ValueError("Queries list cannot be empty")

        if not isinstance(queries, list):
            raise ValueError("Queries must be a list of strings")

        # Filter and validate queries
        valid_queries = []
        invalid_indices = []
        for i, q in enumerate(queries):
            if q and isinstance(q, str) and q.strip():
                valid_queries.append(q.strip())
            else:
                invalid_indices.append(i)

        if invalid_indices:
            logger.warning(
                f"Skipping {len(invalid_indices)} invalid queries at indices: {invalid_indices}"
            )

        if not valid_queries:
            raise ValueError("No valid queries to embed")

        try:
            prefixed_queries = [self._add_prefix(q, task) for q in valid_queries]
            outputs = self.llm.encode(prefixed_queries, pooling_task="embed")

            if not outputs:
                raise RuntimeError("Embedding model returned empty output")

            if len(outputs) != len(valid_queries):
                logger.warning(
                    f"Expected {len(valid_queries)} embeddings, got {len(outputs)}"
                )

            embeddings = []
            for i, output in enumerate(outputs):
                output_data = output.outputs.data
                if hasattr(output_data, "cpu"):
                    output_data = output_data.cpu()
                embedding = np.asarray(output_data).flatten()

                if embedding.size == 0:
                    logger.warning(f"Empty embedding at index {i}")
                if np.isnan(embedding).any():
                    logger.warning(f"Embedding at index {i} contains NaN values")

                embeddings.append(embedding)

            return embeddings

        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"GPU out of memory embedding {len(valid_queries)} queries. "
                "Try reducing batch size."
            )
            raise MemoryError("GPU out of memory during batch embedding") from e

        except Exception as e:
            logger.error(f"Failed to embed queries: {e}")
            raise RuntimeError(f"Batch embedding failed: {e}") from e


# Singleton instance for convenience
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    config: Optional[EmbeddingConfig] = None, **kwargs
) -> EmbeddingModel:
    """Get or create embedding model singleton."""
    global _embedding_model
    if _embedding_model is None:
        if config is not None:
            _embedding_model = EmbeddingModel(config)
        else:
            _embedding_model = EmbeddingModel(**kwargs)
    return _embedding_model


def embed_query(
    query: str,
    model: Optional[EmbeddingModel] = None,
    config: Optional[EmbeddingConfig] = None,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function to embed a query.

    Args:
        query: Query text
        model: Optional pre-initialized model
        config: Optional EmbeddingConfig instance
        **kwargs: Arguments for EmbeddingModel if creating new

    Returns:
        Query embedding vector
    """
    if model is None:
        if config is not None:
            model = get_embedding_model(config=config)
        else:
            model = get_embedding_model(**kwargs)
    return model.embed_query(query)
