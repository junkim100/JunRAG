"""
Sequential Decomposition Pipeline for JunRAG.
Processes multi-hop queries by decomposing them into sequential sub-queries with placeholders.

Flow: Query → Decompose → [For each subquery: Embed → Retrieve → Rerank → Extract/Rewrite] → Generate Final Answer

Key differences from Parallel Pipeline:
- Subqueries are processed sequentially, not in parallel
- Each subquery (except the first) contains [answer] placeholder for previous answer
- After retrieval+reranking, LLM rewrites the next subquery with the actual answer
- Final subquery uses a different prompt to generate the final answer

GPU Split Strategy:
- Retriever: floor(available_gpus / 2) GPUs
- Reranker: ceil(available_gpus / 2) GPUs (gets priority for odd GPU counts)
- Models are loaded once and reused across all subqueries
"""

import gc
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

from junrag.pipelines.base import BasePipeline
from junrag.components.sequential_decomposition import SequentialQueryDecomposer
from junrag.components.retrieval import retrieve_chunks
from junrag.components.embedding import EmbeddingModel
from junrag.components.reranking import Reranker
from junrag.models import (
    Chunk,
    PipelineConfig,
    PipelineResult,
    UsageInfo,
)

# Set spawn method for vLLM
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logger = logging.getLogger(__name__)


class SequentialDecompositionPipeline(BasePipeline):
    """
    Sequential Decomposition RAG pipeline.

    Flow:
    1. Decompose query into sequential sub-queries with [answer] placeholders
    2. For each sub-query (in order):
       a. Embed the sub-query
       b. Retrieve chunks from vector DB
       c. Rerank chunks
       d. If not last sub-query: Extract answer and rewrite next sub-query
       e. If last sub-query: Generate final answer
    3. Return final answer with all intermediate results

    GPU Strategy:
    - Retriever GPUs: One embedder per GPU (tensor_parallel_size=1), assigned to chains round-robin
    - Reranker GPUs: Split among chains, each chain gets its own reranker instance
    - GPU split: retriever gets floor(gpus/2), reranker gets ceil(gpus/2)
    - For odd tensor_parallel_size (e.g., 7): retriever=3 GPUs, reranker=4 GPUs
    - For multiple chains: Reranker GPUs are divided among chains (e.g., 2 chains with 4 reranker GPUs: Chain 1 gets GPUs 4-5, Chain 2 gets GPUs 6-7)
    - Chains run in parallel: Each chain uses its assigned embedder and reranker GPUs independently
    """

    def __init__(
        self,
        # Sequential pipeline specific settings
        decomposition_model: str = "gpt-4o",
        rerank_per_subquery: int = 10,
        # Inherited settings
        **kwargs,
    ):
        """
        Initialize sequential decomposition pipeline.

        Args:
            decomposition_model: Model for query decomposition and answer extraction
            rerank_per_subquery: Number of chunks to keep after reranking per sub-query
            **kwargs: Arguments passed to BasePipeline
        """
        super().__init__(**kwargs)

        self.decomposition_model = decomposition_model
        self.rerank_per_subquery = rerank_per_subquery

        # Detect available GPUs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.available_gpus = min(self.tensor_parallel_size, self.num_gpus)
        if self.available_gpus == 0:
            self.available_gpus = 1  # Fallback to at least 1

        # Calculate GPU split: retriever gets floor, reranker gets ceil (priority)
        self.retriever_gpus = self.available_gpus // 2
        self.reranker_gpus = self.available_gpus - self.retriever_gpus

        # Ensure at least 1 GPU for each if available
        if self.retriever_gpus == 0 and self.available_gpus >= 1:
            self.retriever_gpus = 1
            self.reranker_gpus = max(1, self.available_gpus - 1)

        logger.info(f"Sequential pipeline detected {self.num_gpus} GPUs")
        logger.info(
            f"GPU split: retriever={self.retriever_gpus} GPUs, "
            f"reranker={self.reranker_gpus} GPUs (from {self.available_gpus} available)"
        )

        # Lazy initialization
        self._decomposer = None
        self._embedders: Dict[int, EmbeddingModel] = {}
        self._embedder_loading_lock = threading.Lock()
        # Rerankers are keyed by their assigned GPU tuple (e.g., (4,), (6, 7)).
        # This allows multiple chains to share a single reranker when n_chains > n_reranker_gpus
        # (round-robin GPU assignment), avoiding multiple vLLM engines on the same GPU.
        self._rerankers: Dict[Tuple[int, ...], Reranker] = {}
        self._reranker_loading_lock = threading.Lock()
        # Serialize calls per reranker instance (conservative safety for shared rerankers).
        self._reranker_call_locks: Dict[Tuple[int, ...], threading.Lock] = {}
        # Reranker GPU split depends on n_chains; cache must match current run.
        self._reranker_assignment_key: Optional[Tuple[int, Tuple[int, ...]]] = None

    def _get_retriever_gpu_id_list(self) -> List[int]:
        """Get GPU IDs for retriever/embedders."""
        if self.num_gpus <= 0:
            return [0]

        gpu_ids = list(range(min(self.retriever_gpus, self.num_gpus)))
        if not gpu_ids:
            # Fallback (e.g., tensor_parallel_size=1 => retriever_gpus=0)
            gpu_ids = [0]
        return gpu_ids

    def _get_retriever_gpu_ids(self) -> str:
        """Get comma-separated GPU IDs for retriever/embedders."""
        return ",".join(str(g) for g in self._get_retriever_gpu_id_list())

    def _get_reranker_gpu_id_list(self) -> List[int]:
        """Get GPU IDs for reranker."""
        if self.num_gpus <= 0:
            return [0]

        start_gpu = min(self.retriever_gpus, self.num_gpus - 1)
        end_gpu = min(start_gpu + max(self.reranker_gpus, 1), self.num_gpus)
        gpu_ids = list(range(start_gpu, end_gpu))
        if not gpu_ids:
            gpu_ids = [0]
        return gpu_ids

    def _get_reranker_gpu_ids(self) -> str:
        """Get comma-separated GPU IDs for reranker."""
        return ",".join(str(g) for g in self._get_reranker_gpu_id_list())

    def _cleanup_embedders(self) -> None:
        """Best-effort cleanup of cached embedders to free GPU memory."""
        if not hasattr(self, "_embedders") or self._embedders is None:
            self._embedders = {}

        if not self._embedders:
            return

        for embedder in list(self._embedders.values()):
            try:
                # vLLM backend - delete the LLM reference to trigger cleanup
                if hasattr(embedder, "llm"):
                    try:
                        # Delete the reference - vLLM will clean up when object is garbage collected
                        del embedder.llm
                    except Exception:
                        pass
            except Exception:
                # Best-effort cleanup only
                pass

        try:
            self._embedders.clear()
        except Exception:
            self._embedders = {}

    def _cleanup_rerankers(self) -> None:
        """Best-effort cleanup of cached rerankers to free GPU memory."""
        if not hasattr(self, "_rerankers") or self._rerankers is None:
            self._rerankers = {}

        if not self._rerankers:
            # Still clear locks to keep state consistent.
            if (
                hasattr(self, "_reranker_call_locks")
                and self._reranker_call_locks is not None
            ):
                try:
                    self._reranker_call_locks.clear()
                except Exception:
                    self._reranker_call_locks = {}
            return

        for rr in list(self._rerankers.values()):
            try:
                # vLLM backend - delete the model reference to trigger cleanup
                if hasattr(rr, "model"):
                    try:
                        # Delete the reference - vLLM will clean up when object is garbage collected
                        del rr.model
                    except Exception:
                        pass
                if hasattr(rr, "tokenizer"):
                    try:
                        del rr.tokenizer
                    except Exception:
                        pass
            except Exception:
                # Best-effort cleanup only
                pass

        try:
            self._rerankers.clear()
        except Exception:
            self._rerankers = {}
        # Clear associated call locks
        if (
            hasattr(self, "_reranker_call_locks")
            and self._reranker_call_locks is not None
        ):
            try:
                self._reranker_call_locks.clear()
            except Exception:
                self._reranker_call_locks = {}

    def _cleanup_all_models(self) -> None:
        """Comprehensive cleanup of all models to properly shut down vLLM engines."""
        logger.info("Cleaning up all pipeline models...")
        try:
            # Clean up embedders
            self._cleanup_embedders()

            # Clean up rerankers
            self._cleanup_rerankers()

            # Clean up decomposer (if it has any models)
            if hasattr(self, "_decomposer") and self._decomposer is not None:
                # Decomposer uses OpenAI API, no vLLM models to clean
                pass

            # Force garbage collection
            try:
                gc.collect()
            except Exception:
                pass

            # Clear CUDA cache
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    # Synchronize to ensure cleanup completes
                    torch.cuda.synchronize()
                except Exception:
                    pass

            # Small delay to allow vLLM worker processes to terminate cleanly
            time.sleep(2)

            logger.info("Model cleanup completed")
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

    def _ensure_reranker_assignment(
        self, n_chains: int, skip_cleanup: bool = False
    ) -> None:
        """
        Ensure cached rerankers match current n_chains / GPU assignment.

        Reranker GPU splits depend on n_chains (e.g., n_chains=1 uses all reranker GPUs,
        but n_chains=3 splits them). Reusing cached rerankers created under a different
        n_chains can cause repeated vLLM OOM/free-memory startup failures because the
        previous configuration may still be holding memory on GPUs needed by new chains.

        Args:
            n_chains: Number of chains for this query
            skip_cleanup: If True, only cleanup conflicting rerankers (useful for evaluation)
        """
        desired_key = (int(n_chains), tuple(self._get_reranker_gpu_id_list()))
        if self._reranker_assignment_key is None:
            self._reranker_assignment_key = desired_key
            return

        if self._reranker_assignment_key != desired_key:
            prev_n_chains, prev_gpus = self._reranker_assignment_key
            if skip_cleanup:
                # For evaluation: check if existing rerankers conflict with new assignment
                # Only cleanup rerankers that use GPUs needed by the new configuration
                new_gpus_needed = set()
                for chain_idx in range(1, n_chains + 1):
                    new_key = self._get_reranker_key_for_chain(chain_idx, n_chains)
                    new_gpus_needed.update(new_key)

                # Find rerankers that conflict (use any GPU needed by new assignment)
                conflicting_keys = []
                for cached_key in list(self._rerankers.keys()):
                    cached_gpus = set(cached_key)
                    if cached_gpus & new_gpus_needed:  # Overlap detected
                        conflicting_keys.append(cached_key)

                if conflicting_keys:
                    # Clean up only conflicting rerankers to free GPU memory
                    logger.info(
                        f"Reranker configuration changed (n_chains: {prev_n_chains}→{n_chains}). "
                        f"Cleaning up {len(conflicting_keys)} conflicting reranker(s) to avoid GPU OOM."
                    )
                    for key in conflicting_keys:
                        if key in self._rerankers:
                            reranker = self._rerankers.pop(key)
                            # Explicitly delete model references to free GPU memory
                            if hasattr(reranker, "model"):
                                try:
                                    del reranker.model
                                except Exception:
                                    pass
                            if hasattr(reranker, "tokenizer"):
                                try:
                                    del reranker.tokenizer
                                except Exception:
                                    pass
                    # Remove associated locks
                    for key in conflicting_keys:
                        self._reranker_call_locks.pop(key, None)
                    # Force garbage collection and clear CUDA cache
                    import gc
                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    logger.info("Conflicting rerankers cleaned up")
                else:
                    logger.debug(
                        f"Reranker configuration changed (n_chains: {prev_n_chains}→{n_chains}), "
                        "but no conflicts detected. Keeping existing rerankers."
                    )
                self._reranker_assignment_key = desired_key
            else:
                logger.info(
                    "Reranker configuration changed "
                    f"(was n_chains={prev_n_chains}, reranker_gpus={list(prev_gpus)}; "
                    f"now n_chains={n_chains}, reranker_gpus={self._get_reranker_gpu_id_list()}). "
                    "Releasing cached rerankers to avoid GPU OOM."
                )
                self._cleanup_rerankers()
                self._reranker_assignment_key = desired_key

    @property
    def decomposer(self) -> SequentialQueryDecomposer:
        """Lazy load sequential query decomposer."""
        if self._decomposer is None:
            self._decomposer = SequentialQueryDecomposer(
                model=self.decomposition_model,
                api_key=self.openai_api_key,
            )
        return self._decomposer

    @property
    def seq_embedder(self) -> EmbeddingModel:
        """Backwards-compatible accessor (embedder slot 0)."""
        return self._get_embedder_for_slot(0)

    def _get_embedder_for_slot(self, slot_idx: int) -> EmbeddingModel:
        """
        Get (or load) an embedder assigned to a single GPU.

        We create one embedder per retriever GPU (tensor_parallel_size=1) so chains can run
        in parallel up to floor(gpus/2) as requested.
        """
        gpu_list = self._get_retriever_gpu_id_list()
        slot_idx = slot_idx % len(gpu_list)

        if slot_idx in self._embedders:
            return self._embedders[slot_idx]

        with self._embedder_loading_lock:
            if slot_idx in self._embedders:
                return self._embedders[slot_idx]

            gpu_id = gpu_list[slot_idx]
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                logger.info(f"Loading embedder slot={slot_idx} on GPU: {gpu_id}")
                embedder = EmbeddingModel(
                    model=self.embedding_model_name,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                )
            finally:
                if original_cuda_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

            self._embedders[slot_idx] = embedder
            return embedder

    def _get_reranker_for_chain(self, chain_id: int, n_chains: int) -> Reranker:
        """
        Get reranker for a specific chain with dedicated GPU assignment.

        GPU assignment strategy:
        - If n_chains <= n_reranker_gpus: GPUs are split among chains
          (e.g., 2 chains with 4 reranker GPUs: Chain 1→GPUs 4-5, Chain 2→GPUs 6-7)
        - If n_chains > n_reranker_gpus: Round-robin assignment
          (e.g., 5 chains with 4 reranker GPUs: Chain 1→GPU 4, Chain 2→GPU 5, Chain 3→GPU 6, Chain 4→GPU 7, Chain 5→GPU 4)
        """
        # Compute the GPU assignment for this chain; used as cache key so chains can share.
        reranker_key = self._get_reranker_key_for_chain(chain_id, n_chains)
        if reranker_key in self._rerankers:
            return self._rerankers[reranker_key]

        with self._reranker_loading_lock:
            reranker_key = self._get_reranker_key_for_chain(chain_id, n_chains)
            if reranker_key in self._rerankers:
                return self._rerankers[reranker_key]

            gpu_ids = ",".join(str(g) for g in reranker_key)
            tp_size = max(1, len(reranker_key))
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

            # With GPU-keyed caching, we never create multiple reranker instances on the same GPU(s).
            # Keep the requested utilization (and rely on retry fallback values if other processes
            # are using memory).
            adjusted_memory_util = self.gpu_memory_utilization

            # Retry with progressively lower memory utilization if memory errors occur
            memory_util_values = [adjusted_memory_util]
            if adjusted_memory_util > 0.5:
                memory_util_values.append(0.5)
            if adjusted_memory_util > 0.3:
                memory_util_values.append(0.3)

            last_error = None
            reranker = None
            try:
                for attempt, memory_util in enumerate(memory_util_values):
                    try:
                        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                        if attempt > 0:
                            logger.warning(
                                f"Retrying reranker initialization for chain {chain_id} "
                                f"with reduced gpu_memory_utilization={memory_util:.2f} "
                                f"(attempt {attempt + 1}/{len(memory_util_values)})"
                            )
                        else:
                            logger.info(
                                f"Loading reranker for chain {chain_id} on GPUs: {gpu_ids} "
                                f"(tensor_parallel_size={tp_size}, gpu_memory_utilization={memory_util:.2f})"
                            )

                        reranker = Reranker(
                            model=self.reranker_model_name,
                            batch_size=self.rerank_batch_size,
                            use_multi_gpu=tp_size > 1,
                            tensor_parallel_size=tp_size,
                            gpu_memory_utilization=memory_util,
                            max_model_len=self.max_model_len,
                        )
                        # Success - break out of retry loop
                        break
                    except (MemoryError, ValueError, RuntimeError) as e:
                        error_msg = str(e).lower()
                        is_memory_error = (
                            "memory" in error_msg
                            or "out of memory" in error_msg
                            or "free memory" in error_msg
                            or "gpu memory utilization" in error_msg
                        )

                        last_error = e
                        if is_memory_error and attempt < len(memory_util_values) - 1:
                            logger.warning(
                                f"GPU memory error loading reranker for chain {chain_id}: {e}. "
                                f"Retrying with lower memory utilization..."
                            )
                            continue
                        else:
                            # Not a memory error or exhausted retries
                            raise
                    except Exception as e:
                        # Other errors - don't retry
                        raise
                else:
                    # All retries failed
                    if last_error:
                        raise RuntimeError(
                            f"Failed to initialize reranker for chain {chain_id} after "
                            f"{len(memory_util_values)} attempts with memory utilization "
                            f"values {memory_util_values}. Last error: {last_error}"
                        ) from last_error
            finally:
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

            self._rerankers[reranker_key] = reranker
            # Ensure a lock exists for this reranker instance
            if reranker_key not in self._reranker_call_locks:
                self._reranker_call_locks[reranker_key] = threading.Lock()
            return reranker

    def _get_reranker_key_for_chain(
        self, chain_id: int, n_chains: int
    ) -> Tuple[int, ...]:
        """Return the GPU tuple assigned to a chain's reranker (cache key)."""
        all_reranker_gpus = self._get_reranker_gpu_id_list()
        n_reranker_gpus = len(all_reranker_gpus)

        if n_chains <= 1:
            chain_reranker_gpus = all_reranker_gpus
        elif n_chains > n_reranker_gpus:
            # Round-robin assignment (one GPU per chain); chains may share the same GPU.
            gpu_idx = (chain_id - 1) % n_reranker_gpus
            chain_reranker_gpus = [all_reranker_gpus[gpu_idx]]
        else:
            # Split GPUs among chains; disjoint partitions.
            gpus_per_chain = n_reranker_gpus // n_chains
            extra_gpus = n_reranker_gpus % n_chains
            gpus_for_this_chain = gpus_per_chain + (1 if chain_id <= extra_gpus else 0)
            start_idx = sum(
                gpus_per_chain + (1 if i <= extra_gpus else 0)
                for i in range(1, chain_id)
            )
            end_idx = start_idx + gpus_for_this_chain
            chain_reranker_gpus = all_reranker_gpus[start_idx:end_idx]

        return tuple(int(g) for g in chain_reranker_gpus if g is not None)

    @property
    def seq_reranker(self) -> Reranker:
        """Backwards-compatible accessor (reranker for chain 1)."""
        return self._get_reranker_for_chain(1, 1)

    def _format_context(self, chunks: List[Chunk]) -> str:
        """Format chunks into context string for LLM.

        Improved formatting to make numeric/ranking extraction easier:
        - Clear source markers [1], [2], etc.
        - Preserve structure for easier parsing
        """
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.text if isinstance(chunk, Chunk) else chunk.get("text", "")
            if text and text.strip():
                # Format: [1] text content
                # This format makes it easier for the LLM to extract answers and cite sources
                context_parts.append(f"[{i}] {text.strip()}")

        return "\n\n".join(context_parts)

    def _get_fallback_answer(self, reranked: List[Chunk]) -> Dict[str, Any]:
        """Generate a fallback answer when context is empty."""
        if reranked:
            top_chunk = reranked[0]
            chunk_text = (
                top_chunk.text if hasattr(top_chunk, "text") else str(top_chunk)
            )
            if chunk_text and chunk_text.strip():
                first_sentence = chunk_text.split(".")[0].strip()
                if first_sentence and len(first_sentence) > 10:
                    answer = first_sentence[:200]
                else:
                    answer = chunk_text[:200].strip()
                return {"answer": answer, "supporting_sources": []}
        return {
            "answer": "Unable to determine from provided context",
            "supporting_sources": [],
        }

    @staticmethod
    def _has_answer_placeholder(text: str) -> bool:
        return "[answer]" in (text or "").lower()

    def _split_into_chains(self, sub_queries: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Split a flat sub-query list into independent chains.

        Boundary rule:
        - Start a new chain whenever a sub-query does NOT contain "[answer]" (except the very first),
          and all subsequent "[answer]" queries belong to the most recent chain.
        """
        chains: List[List[Dict[str, Any]]] = []
        current: Optional[List[Dict[str, Any]]] = None

        for flat_idx, sq in enumerate(sub_queries):
            has_placeholder = self._has_answer_placeholder(sq)
            if not has_placeholder:
                current = []
                chains.append(current)
            elif current is None:
                # Edge case: placeholder without a prior root query; treat as its own chain
                current = []
                chains.append(current)

            current.append(
                {
                    "flat_idx": flat_idx,
                    "original_subquery": sq,
                }
            )

        return chains

    def _fill_placeholder(self, template: str, answer: str) -> str:
        """Replace [answer] placeholder (case-insensitive) with the provided answer."""
        if not template:
            return template
        if not answer:
            return re.sub(r"\[answer\]", "", template, flags=re.IGNORECASE).strip()
        return re.sub(r"\[answer\]", answer.strip(), template, flags=re.IGNORECASE)

    def run(
        self,
        query: str,
        retrieval_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        cleanup_models: bool = True,
    ) -> PipelineResult:
        """
        Run the sequential decomposition pipeline.

        Args:
            query: User query (can be complex multi-hop)
            retrieval_top_k: Override retrieval top_k per sub-query
            rerank_top_k: Override chunks per sub-query after reranking

        Returns:
            PipelineResult with answer and pipeline details
        """
        if retrieval_top_k:
            self.retrieval_top_k = retrieval_top_k
        if rerank_top_k:
            self.rerank_per_subquery = rerank_top_k

        # Start timing
        pipeline_start_time = time.time()

        print(f"\n{'='*80}")
        print("SEQUENTIAL DECOMPOSITION PIPELINE")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Collection: {self.collection_name}")
        print(
            f"GPU split: retriever={self.retriever_gpus} GPUs, "
            f"reranker={self.reranker_gpus} GPUs"
        )

        # Step 1: Decompose query into sequential sub-queries
        print(f"\n[Step 1] Decomposing query into sequential sub-queries...")
        step1_start = time.time()
        sub_queries = self.decomposer.decompose(query)
        step1_time = time.time() - step1_start
        n_subqueries = len(sub_queries)
        print(f"  Decomposition completed in {step1_time:.2f}s")

        if n_subqueries == 0:
            logger.error("Query decomposition returned 0 sub-queries")
            raise ValueError("Query decomposition failed: No sub-queries generated.")

        print(f"Decomposed into {n_subqueries} sequential sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            has_placeholder = "[answer]" in sq.lower()
            marker = " [has placeholder]" if has_placeholder else ""
            print(f"  {i}. {sq}{marker}")

        # Split into independent chains (multi-chain support)
        chains = self._split_into_chains(sub_queries)
        n_chains = len(chains)
        print(f"\nIdentified {n_chains} chain(s):")
        for chain_idx, chain in enumerate(chains, 1):
            print(f"  Chain {chain_idx} ({len(chain)} steps)")

        # IMPORTANT: n_chains affects reranker GPU splitting. If a previous query loaded
        # rerankers under a different n_chains, those models may still occupy GPUs
        # (e.g., GPU 7) and cause repeated "free memory < desired utilization" failures
        # when trying to initialize additional chain rerankers. Reset as needed.
        # For evaluation (cleanup_models=False), skip aggressive cleanup to keep models loaded
        self._ensure_reranker_assignment(n_chains, skip_cleanup=not cleanup_models)

        # Step 2: Pre-initialize models (embedder pool + reranker)
        # Check if models are already loaded to avoid unnecessary reloading during evaluation
        retriever_gpu_list = self._get_retriever_gpu_id_list()
        n_embedder_slots = max(1, len(retriever_gpu_list))

        # Check if embedders are already loaded
        embedders_loaded = all(
            slot_idx in self._embedders for slot_idx in range(n_embedder_slots)
        )

        # Check if rerankers are already loaded for all chains
        rerankers_loaded = all(
            self._get_reranker_key_for_chain(chain_idx, n_chains) in self._rerankers
            for chain_idx in range(1, n_chains + 1)
        )

        models_already_loaded = embedders_loaded and rerankers_loaded

        # Initialize preinit_time to 0.0 in case models are already loaded
        preinit_time = 0.0

        if not models_already_loaded:
            print(f"\n[Step 2] Loading models...")
            preinit_start = time.time()

            # Load one embedder per retriever GPU (tensor_parallel_size=1)
            for slot_idx in range(n_embedder_slots):
                _ = self._get_embedder_for_slot(slot_idx)
            print(
                f"  Loaded {n_embedder_slots} embedders on GPUs: {self._get_retriever_gpu_ids()}"
            )

            # Pre-load rerankers for all chains (each chain gets its own reranker)
            for chain_idx in range(1, n_chains + 1):
                _ = self._get_reranker_for_chain(chain_idx, n_chains)
            print(
                f"  Rerankers loaded: {len(self._rerankers)} instance(s) for {n_chains} chain(s) "
                f"on GPUs: {self._get_reranker_gpu_ids()}"
            )

            preinit_time = time.time() - preinit_start
            print(f"  Models loaded in {preinit_time:.2f}s")
        else:
            # Models already loaded - just ensure they're accessible (for evaluation efficiency)
            # Silently verify models are ready without reloading
            for slot_idx in range(n_embedder_slots):
                _ = self._get_embedder_for_slot(slot_idx)
            for chain_idx in range(1, n_chains + 1):
                _ = self._get_reranker_for_chain(chain_idx, n_chains)

        # Step 3: Process chains in parallel (round-robin by embedder slot)
        print(
            f"\n[Step 3] Processing chains (parallel across {n_embedder_slots} embedders)..."
        )
        step3_start = time.time()

        def _normalize_chunk(chunk: Any, source_subquery: str) -> Chunk:
            if isinstance(chunk, Chunk):
                c = chunk.model_copy()
                c.source_subquery = source_subquery
                return c
            if isinstance(chunk, dict):
                chunk_copy = chunk.copy()
                chunk_copy["source_subquery"] = source_subquery
                return Chunk(**chunk_copy)
            return Chunk(text=str(chunk), source_subquery=source_subquery)

        def _process_chain(
            chain_id: int,
            chain_items: List[Dict[str, Any]],
            embedder: EmbeddingModel,
            n_chains: int,
        ) -> Dict[str, Any]:
            chain_start = time.time()
            prev_answer = ""
            steps: List[Dict[str, Any]] = []
            chain_evidence: List[Chunk] = []

            for step_idx, item in enumerate(chain_items, 1):
                flat_idx = int(item["flat_idx"])
                original_subquery = str(item["original_subquery"])

                rewritten_subquery = original_subquery
                if self._has_answer_placeholder(original_subquery):
                    rewritten_subquery = self._fill_placeholder(
                        original_subquery, prev_answer
                    )

                # Build a retrieval-optimized query string (fed to embedder + reranker).
                # This is a pure "input shaping" change to improve retrieval accuracy for later steps
                # by preserving the original user intent alongside the current subquery.
                retrieval_query = rewritten_subquery
                try:
                    if query and isinstance(query, str):
                        retrieval_query = (
                            f"{rewritten_subquery}\n\nOriginal user query:\n{query}"
                        )
                except Exception:
                    retrieval_query = rewritten_subquery

                # Embed
                embed_start = time.time()
                query_embedding = embedder.embed_query(retrieval_query)
                embed_time = time.time() - embed_start

                # Retrieve
                retrieve_start = time.time()
                retrieved = retrieve_chunks(
                    query_embedding=query_embedding,
                    collection_name=self.collection_name,
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    top_k=self.retrieval_top_k,
                )
                retrieve_time = time.time() - retrieve_start

                # Rerank (per-chain reranker, no lock needed)
                reranked: List[Chunk] = []
                rerank_time = 0.0
                if retrieved:
                    rerank_start = time.time()
                    chain_reranker = self._get_reranker_for_chain(chain_id, n_chains)
                    reranker_key = self._get_reranker_key_for_chain(chain_id, n_chains)
                    rerank_lock = self._reranker_call_locks.setdefault(
                        reranker_key, threading.Lock()
                    )
                    with rerank_lock:
                        reranked_raw = chain_reranker.rerank(
                            retrieval_query,
                            retrieved,
                            top_k=self.rerank_per_subquery,
                        )
                    rerank_time = time.time() - rerank_start
                    reranked = [
                        _normalize_chunk(c, source_subquery=rewritten_subquery)
                        for c in reranked_raw
                    ]

                # Answer subquery (LLM) + select evidence chunks based on supporting sources
                context = self._format_context(reranked)
                answer_start = time.time()
                if context:
                    answer_data = self.decomposer.answer_subquery(
                        rewritten_subquery, context=context, original_query=query
                    )
                else:
                    answer_data = self._get_fallback_answer(reranked)
                    answer_data["used_internal_knowledge"] = False
                answer_time = time.time() - answer_start

                answer = str(answer_data.get("answer", "") or "").strip()
                supporting_sources = answer_data.get("supporting_sources", []) or []
                used_internal_knowledge = answer_data.get(
                    "used_internal_knowledge", False
                )

                # Chunks explicitly cited by the subquery-answering LLM.
                # NOTE: If `used_internal_knowledge` is True, the answer was produced without
                # relying on retrieved context, so we do NOT mark any chunk as "supporting".
                supporting_chunks_for_step: List[Chunk] = []
                if not used_internal_knowledge and isinstance(supporting_sources, list):
                    for s in supporting_sources:
                        try:
                            si = int(s)
                        except Exception:
                            continue
                        if 1 <= si <= len(reranked):
                            supporting_chunks_for_step.append(reranked[si - 1])

                # Evidence chunks to carry forward to final generation (can exist even when
                # internal knowledge was used, because retrieval still ran).
                evidence_chunks_for_generation: List[Chunk] = []
                if supporting_chunks_for_step:
                    evidence_chunks_for_generation = supporting_chunks_for_step
                elif reranked:
                    evidence_chunks_for_generation = reranked[:1]

                # Fallback: If answer is empty but we have reranked chunks, extract from top chunk
                if not answer and reranked:
                    # Try to extract a meaningful snippet from the top reranked chunk
                    top_chunk = reranked[0]
                    chunk_text = (
                        top_chunk.text if hasattr(top_chunk, "text") else str(top_chunk)
                    )
                    if chunk_text and chunk_text.strip():
                        # Extract first sentence or first 100 chars as fallback answer
                        first_sentence = chunk_text.split(".")[0].strip()
                        if first_sentence and len(first_sentence) > 10:
                            answer = first_sentence[:200]  # Limit to 200 chars
                        else:
                            answer = chunk_text[:200].strip()

                # Final fallback: If still empty, use a default message
                if not answer:
                    answer = "Unable to determine from provided context"

                chain_evidence.extend(evidence_chunks_for_generation)

                steps.append(
                    {
                        "chain_id": chain_id,
                        "step_idx": step_idx,
                        "flat_idx": flat_idx,
                        "original_subquery": original_subquery,
                        "rewritten_subquery": rewritten_subquery,
                        "answer": answer,
                        "supporting_sources": supporting_sources,
                        "supporting_chunk_ids": [
                            c.chunk_id for c in supporting_chunks_for_step if c.chunk_id
                        ],
                        "retrieved_count": len(retrieved),
                        "reranked_count": len(reranked),
                        "embed_time": embed_time,
                        "retrieve_time": retrieve_time,
                        "rerank_time": rerank_time,
                        "answer_time": answer_time,
                        "used_internal_knowledge": used_internal_knowledge,
                    }
                )

                prev_answer = answer

            chain_time = time.time() - chain_start
            return {
                "chain_id": chain_id,
                "n_steps": len(steps),
                "steps": steps,
                "chain_answer": steps[-1]["answer"] if steps else "",
                "runtime": chain_time,
                "_evidence_chunks": chain_evidence,
            }

        # Assign chains to embedder slots round-robin
        slot_to_chain_indices: List[List[int]] = [[] for _ in range(n_embedder_slots)]
        for idx in range(n_chains):
            slot_to_chain_indices[idx % n_embedder_slots].append(idx)

        chain_results: List[Dict[str, Any]] = []

        def _run_slot(slot_idx: int, chain_indices: List[int]) -> List[Dict[str, Any]]:
            embedder = self._get_embedder_for_slot(slot_idx)
            out: List[Dict[str, Any]] = []
            for chain_idx in chain_indices:
                out.append(
                    _process_chain(chain_idx + 1, chains[chain_idx], embedder, n_chains)
                )
            return out

        with ThreadPoolExecutor(max_workers=n_embedder_slots) as ex:
            futures = [
                ex.submit(_run_slot, slot_idx, chain_idxs)
                for slot_idx, chain_idxs in enumerate(slot_to_chain_indices)
                if chain_idxs
            ]
            for fut in as_completed(futures):
                chain_results.extend(fut.result())

        # Sort results to match chain order
        chain_results.sort(key=lambda x: x["chain_id"])

        # Reconstruct flat rewritten sub-queries and collect evidence
        rewritten_subqueries: List[str] = list(sub_queries)
        per_subquery: List[Dict[str, Any]] = []
        evidence_chunks: List[Chunk] = []
        for chain in chain_results:
            evidence_chunks.extend(chain.pop("_evidence_chunks", []))
            for step in chain["steps"]:
                flat_idx = int(step["flat_idx"])
                rewritten_subqueries[flat_idx] = step["rewritten_subquery"]
                per_subquery.append(step)

        step3_time = time.time() - step3_start
        print(f"  Chain processing completed in {step3_time:.2f}s")

        # Step 4: Generate final answer (aggregate all chains)
        print(f"\n[Step 4] Generating final answer (aggregating chains)...")
        step4_start = time.time()

        # Deduplicate evidence chunks for generation
        seen_ids = set()
        unique_chunks: List[Chunk] = []
        for chunk in evidence_chunks:
            chunk_id = chunk.chunk_id
            if chunk_id is None:
                import hashlib

                chunk_id = f"text_{hashlib.md5(chunk.text.encode()).hexdigest()[:8]}"
                chunk.chunk_id = chunk_id
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)

        # Normalize reranker scores for fair comparison across subqueries
        if unique_chunks:
            scores = [
                c.reranker_score for c in unique_chunks if c.reranker_score is not None
            ]
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score
                if score_range > 0:
                    # Normalize to [0, 1] range
                    for chunk in unique_chunks:
                        if chunk.reranker_score is not None:
                            chunk.reranker_score = (
                                chunk.reranker_score - min_score
                            ) / score_range

        unique_chunks.sort(key=lambda x: x.reranker_score or 0, reverse=True)

        # Build chain outputs text (passed to final LLM as a synthetic context chunk)
        chain_output_lines: List[str] = []
        for chain in chain_results:
            chain_output_lines.append(f"Chain {chain['chain_id']}:")
            for step in chain["steps"]:
                # chain_output_lines.append(f"- Q{step['step_idx']} original: {step['original_subquery']}")
                chain_output_lines.append(
                    f"  - Q{step['step_idx']} query: {step['rewritten_subquery']}"
                )
                origin = (
                    "internal_knowledge"
                    if step.get("used_internal_knowledge", False)
                    else "context"
                )
                sources = step.get("supporting_sources", []) or []
                sources_note = (
                    f" sources={sources}"
                    if origin == "context" and isinstance(sources, list) and sources
                    else ""
                )
                chain_output_lines.append(
                    f"  - Q{step['step_idx']} answer ({origin}{sources_note}): {step['answer']}"
                )
            chain_output_lines.append("")

        chain_outputs_text = "\n".join(chain_output_lines).strip()
        chain_outputs_chunk = Chunk(
            chunk_id="chain_outputs",
            text=chain_outputs_text,
            metadata={"source": "chain_outputs"},
            reranker_score=1e9,
        )

        generation_chunks = [chain_outputs_chunk] + unique_chunks
        generation_result = self.generator.generate(query, generation_chunks)
        step4_time = time.time() - step4_start
        print(
            f"Answer generated ({generation_result.usage.total_tokens} tokens) in {step4_time:.2f}s"
        )

        # Update final ranks on evidence chunks only
        for rank, chunk in enumerate(unique_chunks, 1):
            chunk.final_rank = rank

        config = PipelineConfig(
            embedding_model=self.embedding_model_name,
            reranker_model=self.reranker_model_name,
            llm_model=self.llm_model_name,
            decomposition_model=self.decomposition_model,
            retrieval_top_k=self.retrieval_top_k,
            rerank_top_k=self.rerank_per_subquery,
            tensor_parallel_size=self.tensor_parallel_size,
        )

        usage = UsageInfo(**generation_result.usage.model_dump())

        # Check if internal knowledge was used in any step across all chains
        used_internal_knowledge = any(
            step.get("used_internal_knowledge", False)
            for chain in chain_results
            for step in chain.get("steps", [])
        )

        result = PipelineResult(
            query=query,
            answer=generation_result.answer,
            pipeline="sequential_decomposition",
            config=config,
            usage=usage,
            sub_queries=rewritten_subqueries,
            original_sub_queries=sub_queries,
            n_subqueries=n_subqueries,
            n_unique_chunks=len(unique_chunks),
            used_internal_knowledge=used_internal_knowledge,
            retrieved_per_subquery=[
                {
                    "chain_id": s["chain_id"],
                    "flat_idx": s["flat_idx"],
                    "subquery": s["rewritten_subquery"],
                    "count": s["retrieved_count"],
                }
                for s in per_subquery
            ],
            reranked_per_subquery=[
                {
                    "chain_id": s["chain_id"],
                    "flat_idx": s["flat_idx"],
                    "subquery": s["rewritten_subquery"],
                    "count": s["reranked_count"],
                }
                for s in per_subquery
            ],
            final_chunks=unique_chunks,
            chains=chain_results,
            n_chains=n_chains,
        )

        pipeline_end_time = time.time()
        total_runtime = pipeline_end_time - pipeline_start_time

        timing_info = {
            "total_runtime": total_runtime,
            "preinit_models": preinit_time,
            "step1_decomposition": step1_time,
            "step3_chain_processing": step3_time,
            "step4_generation": step4_time,
            "per_subquery": [
                {
                    "chain_id": s["chain_id"],
                    "flat_idx": s["flat_idx"],
                    "embed_time": s["embed_time"],
                    "retrieve_time": s["retrieve_time"],
                    "rerank_time": s["rerank_time"],
                    "answer_time": s["answer_time"],
                }
                for s in per_subquery
            ],
            "per_chain": [
                {"chain_id": c["chain_id"], "runtime": c["runtime"]}
                for c in chain_results
            ],
        }

        print(f"\n{'='*80}")
        print("SEQUENTIAL DECOMPOSITION PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(
            f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)"
        )
        print(f"\nTiming breakdown:")
        print(f"  Model Loading: {preinit_time:.2f}s")
        print(f"  Step 1 (Decomposition): {step1_time:.2f}s")
        print(f"  Step 3 (Chain Processing): {step3_time:.2f}s")
        print(f"  Step 4 (Generation): {step4_time:.2f}s")

        result.timing = timing_info

        # Cleanup models to properly shut down vLLM engines
        # Skip cleanup if cleanup_models=False (useful for evaluation where models should persist)
        if cleanup_models:
            try:
                self._cleanup_all_models()
            except Exception as e:
                logger.warning(f"Error during cleanup (non-fatal): {e}")

        return result
