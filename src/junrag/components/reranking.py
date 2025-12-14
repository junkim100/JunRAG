"""
Reranking component for JunRAG.

Supports two reranker types:
1. Qwen3-Reranker: Uses vLLM with generative logprob-based scoring (yes/no tokens)
2. Jina-Reranker: Uses transformers with cross-encoder classification

The component auto-detects the model type and uses the appropriate backend.
"""

import logging
import math
import os
from typing import Dict, List, Optional, Union

# Set spawn method for vLLM multiprocessing before torch import
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from transformers import AutoTokenizer

from junrag.models import Chunk, RerankerConfig

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
        from vllm import LLM as vLLM_LLM, SamplingParams as vLLM_SamplingParams
    _VLLM_AVAILABLE = True
except Exception as e:
    # If import fails, we'll handle it in __init__
    _VLLM_AVAILABLE = False
    _VLLM_IMPORT_ERROR = e
    vLLM_LLM = None
    vLLM_SamplingParams = None


class Reranker:
    """
    Universal reranker supporting both vLLM (Qwen3) and transformers (Jina) backends.

    - Qwen3-Reranker: vLLM with tensor parallelism (recommended)
    - Jina-Reranker: transformers with DataParallel
    """

    def __init__(
        self,
        model: Union[str, RerankerConfig] = "Qwen/Qwen3-Reranker-4B",
        device: Optional[str] = None,
        use_multi_gpu: bool = True,
        batch_size: int = 64,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 8192,
        instruction: Optional[str] = None,
    ):
        """
        Initialize reranker.

        Args:
            model: Reranker model name or RerankerConfig instance
            device: Device to use (cuda/cpu) - for transformers backend
            use_multi_gpu: Use DataParallel for transformers backend
            batch_size: Batch size for inference
            tensor_parallel_size: Number of GPUs for vLLM (default: 1)
            gpu_memory_utilization: GPU memory fraction for vLLM
            max_model_len: Maximum sequence length
            instruction: Custom instruction for Qwen3 reranker
        """
        # Handle Pydantic config or individual parameters
        if isinstance(model, RerankerConfig):
            config = model
            self.model_name = config.model
            device = config.device
            use_multi_gpu = config.use_multi_gpu
            batch_size = config.batch_size
            tensor_parallel_size = config.tensor_parallel_size
            gpu_memory_utilization = config.gpu_memory_utilization
            max_model_len = config.max_model_len
            instruction = config.instruction
        else:
            self.model_name = model

        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.instruction = (
            instruction
            or "Given a web search query, retrieve relevant passages that answer the query"
        )

        # Validate parameters
        if batch_size <= 0:
            logger.warning(f"Invalid batch_size={batch_size}. Using 64.")
            self.batch_size = 64

        if not 0.0 < gpu_memory_utilization <= 1.0:
            logger.warning(
                f"Invalid gpu_memory_utilization={gpu_memory_utilization}. Using 0.8."
            )
            gpu_memory_utilization = 0.8

        # Check GPU availability
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if tensor_parallel_size > available_gpus and available_gpus > 0:
            logger.warning(
                f"Requested tensor_parallel_size={tensor_parallel_size} but only "
                f"{available_gpus} GPUs available. Using {available_gpus}."
            )
            tensor_parallel_size = available_gpus

        # Detect backend based on model name
        self.use_vllm = "qwen" in model.lower() and "reranker" in model.lower()

        logger.info(f"Initializing reranker: {model}")
        logger.info(f"  Backend: {'vLLM' if self.use_vllm else 'transformers'}")

        try:
            if self.use_vllm:
                self._init_vllm(
                    model, tensor_parallel_size, gpu_memory_utilization, max_model_len
                )
            else:
                self._init_transformers(model, device, use_multi_gpu, batch_size)
        except Exception as e:
            logger.error(f"Failed to initialize reranker '{model}': {e}")
            raise

    def _init_vllm(
        self,
        model: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
    ):
        """Initialize vLLM backend for Qwen3-Reranker."""
        if not _VLLM_AVAILABLE:
            if _VLLM_IMPORT_ERROR:
                logger.error(
                    f"vLLM import failed at module level: {_VLLM_IMPORT_ERROR}"
                )
                raise ImportError(
                    "vLLM is required for Qwen3 reranker but failed to import. "
                    f"Original error: {_VLLM_IMPORT_ERROR}"
                ) from _VLLM_IMPORT_ERROR
            else:
                raise ImportError(
                    "vLLM is required for Qwen3 reranker. Install with: pip install vllm"
                )

        # Import TokensPrompt (not needed at module level)
        try:
            from vllm.inputs.data import TokensPrompt
        except ImportError as e:
            logger.error(f"Failed to import TokensPrompt from vLLM: {e}")
            raise ImportError("Failed to import TokensPrompt from vLLM") from e

        self.TokensPrompt = TokensPrompt

        # Use provided tensor_parallel_size (defaults to 1, matching embedding model)
        tp_size = tensor_parallel_size

        logger.info(f"Loading reranker: {model}")
        logger.info(f"  Backend: vLLM (generative)")
        logger.info(f"  Tensor parallel: {tp_size} GPUs")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True
            )
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model}: {e}")
            raise ValueError(f"Failed to load tokenizer: {e}") from e

        try:
            # Use module-level imported LLM class
            self.model = vLLM_LLM(
                model=model,
                tensor_parallel_size=tp_size,
                max_model_len=max_model_len,
                enable_prefix_caching=True,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"GPU out of memory loading reranker. "
                f"Try reducing gpu_memory_utilization (current: {gpu_memory_utilization}) "
                f"or max_model_len (current: {max_model_len})."
            )
            raise MemoryError("GPU out of memory loading reranker model") from e
        except Exception as e:
            error_msg = str(e).lower()
            error_type = type(e).__name__

            # Check for vLLM memory errors (not just torch.cuda.OutOfMemoryError)
            is_memory_error = (
                "memory" in error_msg
                or "out of memory" in error_msg
                or "free memory" in error_msg
                or "gpu memory utilization" in error_msg
                or "MemoryError" in error_type
                or "OutOfMemoryError" in error_type
            )

            if is_memory_error:
                logger.error(
                    f"GPU memory error loading reranker: {error_type}: {e}. "
                    f"Current gpu_memory_utilization: {gpu_memory_utilization}. "
                    "Try reducing gpu_memory_utilization or freeing GPU memory."
                )
                raise MemoryError(
                    f"GPU memory error loading reranker: {e}. "
                    "Try reducing gpu_memory_utilization or freeing GPU memory."
                ) from e

            if "not found" in error_msg:
                logger.error(f"Model '{model}' not found. Check model name.")
                raise ValueError(f"Model '{model}' not found") from e
            logger.error(f"Failed to initialize vLLM reranker: {e}")
            raise

        # Setup tokens for yes/no scoring
        try:
            self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[
                0
            ]
            self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[
                0
            ]
        except Exception as e:
            logger.error(f"Failed to tokenize yes/no tokens: {e}")
            raise RuntimeError("Failed to setup yes/no tokens for reranker") from e

        # Suffix for Qwen3 chat template
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )

        # Use module-level imported SamplingParams class
        self.sampling_params = vLLM_SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )

        logger.info(f"vLLM reranker '{model}' loaded successfully")

    def _init_transformers(
        self,
        model: str,
        device: Optional[str],
        use_multi_gpu: bool,
        batch_size: int,
    ):
        """Initialize transformers backend for Jina-Reranker."""
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError as e:
            logger.error("transformers not installed")
            raise ImportError(
                "transformers is required for reranking. Install with: pip install transformers"
            ) from e

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            # Validate device
            if device not in ["cuda", "cpu"] and not device.startswith("cuda:"):
                logger.warning(
                    f"Unknown device '{device}'. Falling back to auto-detection."
                )
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

        if self.device == "cpu":
            logger.warning("Reranker running on CPU. This may be slow.")

        # Check for multiple GPUs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_data_parallel = use_multi_gpu and self.num_gpus > 1

        logger.info(f"Loading reranker: {model}")
        logger.info(f"  Backend: transformers (cross-encoder)")
        logger.info(f"  Device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model}: {e}")
            raise ValueError(f"Failed to load tokenizer: {e}") from e

        # Set padding token if not defined
        pad_token_added = False
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # Ensure pad_token_id is set
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.debug("Set pad_token to eos_token")
            else:
                # Fallback: add a new padding token
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                pad_token_added = True
                logger.debug("Added new [PAD] token")

        # Validate pad_token_id is set
        if self.tokenizer.pad_token_id is None:
            logger.error(f"Tokenizer {model} has no padding token")
            raise ValueError(
                f"Tokenizer {model} does not have a padding token defined. "
                "Please ensure the tokenizer has a pad_token or eos_token."
            )

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model, trust_remote_code=True
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU out of memory loading reranker model")
            raise MemoryError("GPU out of memory loading reranker model") from e
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                logger.error(f"Model '{model}' not found")
                raise ValueError(f"Model '{model}' not found") from e
            logger.error(f"Failed to load reranker model: {e}")
            raise

        # Resize token embeddings if we added a new pad token
        if pad_token_added:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Update model config with padding token ID
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None and pad_token_id >= 0:
            if hasattr(self.model, "config"):
                # Set pad_token_id in config
                self.model.config.pad_token_id = pad_token_id
                # Also ensure it's set as an attribute
                setattr(self.model.config, "pad_token_id", pad_token_id)
            # Also set on the model itself if it has the attribute
            if hasattr(self.model, "pad_token_id"):
                self.model.pad_token_id = pad_token_id

        self.model.eval()

        try:
            if self.use_data_parallel:
                logger.info(f"  Multi-GPU: {self.num_gpus} GPUs with DataParallel")
                self.model = self.model.to(self.device)
                self.model = torch.nn.DataParallel(self.model)
                self.effective_batch_size = batch_size * self.num_gpus
            else:
                self.model = self.model.to(self.device)
                self.effective_batch_size = batch_size

                # Try torch.compile for optimization
                if hasattr(torch, "compile") and self.device == "cuda":
                    try:
                        self.model = torch.compile(self.model)
                        logger.info("  Optimization: torch.compile enabled")
                    except Exception as e:
                        logger.debug(f"torch.compile failed (non-critical): {e}")
        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU out of memory moving model to device")
            raise MemoryError("GPU out of memory") from e

        logger.info(f"Transformers reranker '{model}' loaded successfully")

    def _format_qwen3_instruction(self, query: str, doc: str) -> List[Dict]:
        """Format instruction for Qwen3-Reranker."""
        return [
            {
                "role": "system",
                "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
            },
            {
                "role": "user",
                "content": f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
            },
        ]

    def _rerank_vllm(
        self, query: str, chunks: List[Dict], top_k: int, text_key: str
    ) -> List[Dict]:
        """Rerank using vLLM backend (Qwen3-Reranker)."""
        # Extract documents
        documents = []
        for chunk in chunks:
            text = chunk.get(text_key, "") or chunk.get("chunk_text", "") or ""
            documents.append(text)

        # Format messages
        pairs = [(query, doc) for doc in documents]
        messages = [self._format_qwen3_instruction(q, d) for q, d in pairs]

        # Apply chat template
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Add suffix tokens
        max_len = self.max_model_len - len(self.suffix_tokens)
        inputs = [
            self.TokensPrompt(prompt_token_ids=tokens[:max_len] + self.suffix_tokens)
            for tokens in tokenized
        ]

        # Generate and compute scores
        outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=False)

        scores = []
        for output in outputs:
            final_logits = output.outputs[0].logprobs[-1]

            true_logit = final_logits.get(self.true_token)
            false_logit = final_logits.get(self.false_token)

            true_logit = true_logit.logprob if true_logit else -10
            false_logit = false_logit.logprob if false_logit else -10

            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)

        # Add scores to chunks
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = chunk.copy()
            chunk_copy["reranker_score"] = float(score)
            scored_chunks.append(chunk_copy)

        # Sort and take top_k
        scored_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
        reranked = scored_chunks[:top_k]

        # Update ranks
        for rank, chunk in enumerate(reranked, start=1):
            chunk["rank"] = rank

        return reranked

    def _rerank_transformers(
        self, query: str, chunks: List[Dict], top_k: int, text_key: str
    ) -> List[Dict]:
        """Rerank using transformers backend (Jina-Reranker)."""
        # Extract documents
        documents = []
        for chunk in chunks:
            text = chunk.get(text_key, "") or chunk.get("chunk_text", "") or ""
            documents.append(text)

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score in batches
        all_scores = []
        for i in range(0, len(pairs), self.effective_batch_size):
            batch_pairs = pairs[i : i + self.effective_batch_size]

            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            all_scores.extend(scores.cpu().tolist())

        # Add scores to chunks
        scored_chunks = []
        for chunk, score in zip(chunks, all_scores):
            chunk_copy = chunk.copy()
            chunk_copy["reranker_score"] = float(score)
            scored_chunks.append(chunk_copy)

        # Sort and take top_k
        scored_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
        reranked = scored_chunks[:top_k]

        # Update ranks
        for rank, chunk in enumerate(reranked, start=1):
            chunk["rank"] = rank

        return reranked

    def rerank(
        self,
        query: str,
        chunks: Union[List[Dict], List[Chunk]],
        top_k: int = 5,
        text_key: str = "text",
    ) -> List[Chunk]:
        """
        Rerank chunks for a query.

        Args:
            query: Query text
            chunks: List of chunk dictionaries or Chunk models
            top_k: Number of top results to return
            text_key: Key for text content in chunks

        Returns:
            Reranked chunks with reranker_score added

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If reranking fails
        """
        # Validate inputs
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty or whitespace only")

        if not chunks:
            logger.warning("No chunks provided for reranking")
            return []

        if top_k <= 0:
            logger.warning(f"Invalid top_k={top_k}. Using 5.")
            top_k = 5

        # Cap top_k at number of chunks
        if top_k > len(chunks):
            logger.debug(
                f"top_k ({top_k}) > chunks ({len(chunks)}). Returning all chunks."
            )
            top_k = len(chunks)

        # Convert Chunk models to dicts for processing
        chunks_dict = []
        for chunk in chunks:
            if isinstance(chunk, Chunk):
                chunks_dict.append(chunk.model_dump())
            else:
                chunks_dict.append(chunk)

        # Check for empty text in chunks
        empty_count = sum(
            1
            for c in chunks_dict
            if not (c.get(text_key) or c.get("chunk_text") or c.get("content"))
        )
        if empty_count > 0:
            logger.warning(
                f"{empty_count}/{len(chunks_dict)} chunks have no text content. "
                f"Checked keys: '{text_key}', 'chunk_text', 'content'"
            )

        try:
            if self.use_vllm:
                reranked_dicts = self._rerank_vllm(query, chunks_dict, top_k, text_key)
            else:
                reranked_dicts = self._rerank_transformers(
                    query, chunks_dict, top_k, text_key
                )

            # Convert back to Chunk models
            return [Chunk(**chunk) for chunk in reranked_dicts]
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                "GPU out of memory during reranking. "
                "Try reducing batch_size or number of chunks."
            )
            raise MemoryError("GPU out of memory during reranking") from e
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise RuntimeError(f"Reranking failed: {e}") from e


# Singleton for convenience
_reranker: Optional[Reranker] = None


def get_reranker(config: Optional[RerankerConfig] = None, **kwargs) -> Reranker:
    """Get or create reranker singleton."""
    global _reranker
    if _reranker is None:
        if config is not None:
            _reranker = Reranker(config)
        else:
            _reranker = Reranker(**kwargs)
    return _reranker


def rerank_chunks(
    query: str,
    chunks: Union[List[Dict], List[Chunk]],
    top_k: int = 5,
    reranker: Optional[Reranker] = None,
    config: Optional[RerankerConfig] = None,
    **kwargs,
) -> List[Chunk]:
    """
    Convenience function to rerank chunks.

    Args:
        query: Query text
        chunks: List of chunks
        top_k: Number of results
        reranker: Optional pre-initialized reranker
        config: Optional RerankerConfig instance
        **kwargs: Arguments for Reranker if creating new

    Returns:
        Reranked chunks
    """
    if reranker is None:
        if config is not None:
            reranker = get_reranker(config=config)
        else:
            reranker = get_reranker(**kwargs)
    return reranker.rerank(query, chunks, top_k=top_k)
