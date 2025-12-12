"""
Web UI Pipeline for JunRAG.
Optimized for web interface with pre-loaded models on dedicated GPUs.

GPU Allocation:
- 4 GPUs for embedders (one per GPU)
- 4 GPUs for rerankers (one per GPU)
- Query decomposition: OpenAI API
- Final generation: OpenAI API
"""

import logging
import os
import multiprocessing
import threading
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
    ALL_COMPLETED,
)
from typing import Dict, List, Optional

import torch

from junrag.pipelines.base import BasePipeline
from junrag.components.decomposition import QueryDecomposer
from junrag.components.retrieval import retrieve_chunks
from junrag.models import (
    Chunk,
    PipelineConfig,
    PipelineResult,
    UsageInfo,
)

# Set spawn method for vLLM
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logger = logging.getLogger(__name__)


class WebUIPipeline(BasePipeline):
    """
    Web UI optimized pipeline with pre-loaded models on dedicated GPUs.

    GPU Requirements:
    - Exactly 8 GPUs required
    - GPUs 0-3: Embedders (one per GPU)
    - GPUs 4-7: Rerankers (one per GPU)

    Flow:
    1. Pre-load all models on startup
    2. Decompose query (API)
    3. Embed subqueries in parallel (pre-loaded models)
    4. Retrieve and rerank in parallel (pre-loaded models)
    5. Generate answer (API)
    """

    def __init__(
        self,
        # Web UI specific settings
        max_cap: int = 25,
        min_floor: int = 5,
        chunks_per_subquery: int = 10,
        max_concurrent_retrievals: int = 8,
        decomposition_model: str = "gpt-5-mini-2025-08-07",
        # Inherited settings
        **kwargs,
    ):
        """
        Initialize web UI pipeline with pre-loaded models.

        Args:
            max_cap: Maximum chunks for final selection
            min_floor: Minimum chunks for final selection
            chunks_per_subquery: Chunks per sub-query after reranking
            max_concurrent_retrievals: Max parallel Qdrant retrieval operations
            decomposition_model: Model for query decomposition (API)
            **kwargs: Arguments passed to BasePipeline
        """
        super().__init__(**kwargs)

        self.max_cap = max_cap
        self.min_floor = min_floor
        self.chunks_per_subquery = chunks_per_subquery
        self.max_concurrent_retrievals = max_concurrent_retrievals
        self.decomposition_model = decomposition_model

        # Check GPU requirements
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.num_gpus < 8:
            raise RuntimeError(
                f"WebUIPipeline requires exactly 8 GPUs, but only {self.num_gpus} are available. "
                "Please ensure 8 GPUs are available."
            )

        # GPU allocation: 4 for embedders, 4 for rerankers
        self.embedder_gpus = [0, 1, 2, 3]
        self.reranker_gpus = [4, 5, 6, 7]

        logger.info(f"WebUI pipeline detected {self.num_gpus} GPUs")
        logger.info(f"  Embedder GPUs: {self.embedder_gpus}")
        logger.info(f"  Reranker GPUs: {self.reranker_gpus}")

        # Semaphore for Qdrant retrieval rate limiting
        self._retrieval_semaphore = threading.Semaphore(max_concurrent_retrievals)

        # Pre-loaded models (initialized on startup)
        self._embedders = {}  # GPU index -> EmbeddingModel
        self._rerankers = {}  # GPU index -> Reranker
        self._models_loaded = False
        self._model_loading_lock = threading.Lock()

        # Lazy initialization for API-based components
        self._decomposer = None

    @property
    def decomposer(self) -> QueryDecomposer:
        """Lazy load query decomposer (uses API)."""
        if self._decomposer is None:
            self._decomposer = QueryDecomposer(
                model=self.decomposition_model,
                api_key=self.openai_api_key,
            )
        return self._decomposer

    def _get_embedder_for_gpu(self, gpu_idx: int):
        """Get embedder for a specific GPU (must be pre-loaded)."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        if gpu_idx not in self._embedders:
            raise ValueError(f"Embedder not available for GPU {gpu_idx}")
        return self._embedders[gpu_idx]

    def _get_reranker_for_gpu(self, gpu_idx: int):
        """Get reranker for a specific GPU (must be pre-loaded)."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        if gpu_idx not in self._rerankers:
            raise ValueError(f"Reranker not available for GPU {gpu_idx}")
        return self._rerankers[gpu_idx]

    def load_models(self):
        """
        Pre-load all embedders and rerankers on their assigned GPUs.
        This should be called once at startup.
        """
        if self._models_loaded:
            logger.warning("Models already loaded. Skipping.")
            return

        with self._model_loading_lock:
            if self._models_loaded:
                return

            logger.info("Pre-loading embedders and rerankers in parallel...")
            load_start = time.time()

            # Store original CUDA_VISIBLE_DEVICES
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

            # Since vLLM models can't be pickled, we must load them in the main process
            # We'll use threading with a lock to coordinate CUDA_VISIBLE_DEVICES changes
            # and load models sequentially, but prepare everything in parallel
            from junrag.components.embedding import EmbeddingModel
            from junrag.components.reranking import Reranker

            use_vllm = (
                "qwen" in self.reranker_model_name.lower()
                and "reranker" in self.reranker_model_name.lower()
            )

            # Use a lock to ensure only one model loads at a time
            # This prevents GPU memory conflicts when multiple models try to initialize
            loading_lock = threading.Lock()

            def _load_embedder_task(gpu_idx):
                """Load embedder on specific GPU."""
                # Hold lock for entire loading process to ensure sequential loading
                with loading_lock:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                    logger.info(f"Loading embedder on GPU {gpu_idx}...")

                    try:
                        # Model loading - vLLM will use the GPU set by CUDA_VISIBLE_DEVICES
                        embedder = EmbeddingModel(
                            model=self.embedding_model_name,
                            tensor_parallel_size=1,
                            gpu_memory_utilization=self.gpu_memory_utilization,
                            max_model_len=self.max_model_len,
                        )
                        self._embedders[gpu_idx] = embedder
                        logger.info(f"Embedder loaded on GPU {gpu_idx}")
                    except Exception as e:
                        logger.error(f"Failed to load embedder on GPU {gpu_idx}: {e}")
                        raise

            def _load_reranker_task(gpu_idx):
                """Load reranker on specific GPU."""
                if use_vllm:
                    # Hold lock for entire loading process
                    with loading_lock:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                        logger.info(f"Loading reranker on GPU {gpu_idx}...")

                        try:
                            reranker = Reranker(
                                model=self.reranker_model_name,
                                batch_size=self.rerank_batch_size,
                                use_multi_gpu=False,
                                tensor_parallel_size=1,
                                gpu_memory_utilization=self.gpu_memory_utilization,
                                max_model_len=self.max_model_len,
                            )
                            self._rerankers[gpu_idx] = reranker
                            logger.info(f"Reranker loaded on GPU {gpu_idx}")
                        except Exception as e:
                            logger.error(
                                f"Failed to load reranker on GPU {gpu_idx}: {e}"
                            )
                            raise
                else:
                    # For transformers backend, no need for CUDA_VISIBLE_DEVICES lock
                    device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Loading reranker on {device}...")
                    reranker = Reranker(
                        model=self.reranker_model_name,
                        device=device,
                        batch_size=self.rerank_batch_size,
                        use_multi_gpu=False,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_len,
                    )
                    with loading_lock:
                        self._rerankers[gpu_idx] = reranker
                    logger.info(f"Reranker loaded on {device}")

            # Load models sequentially using threads for task management
            # The lock ensures only one model loads at a time, preventing GPU memory conflicts
            with ThreadPoolExecutor(max_workers=8) as executor:
                # Submit all embedder tasks
                embedder_futures = [
                    executor.submit(_load_embedder_task, gpu_idx)
                    for gpu_idx in self.embedder_gpus
                ]
                # Submit all reranker tasks
                reranker_futures = [
                    executor.submit(_load_reranker_task, gpu_idx)
                    for gpu_idx in self.reranker_gpus
                ]

                # Wait for all tasks to complete
                for future in embedder_futures + reranker_futures:
                    future.result()  # This will raise if any task failed

            # Set CUDA_VISIBLE_DEVICES to all GPUs after loading
            # Models are already loaded and bound to their specific GPUs, but we need
            # to make all GPUs visible for any future operations
            all_gpus = ",".join(str(i) for i in range(self.num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus
            logger.info(f"Set CUDA_VISIBLE_DEVICES to all GPUs: {all_gpus}")

            load_time = time.time() - load_start
            self._models_loaded = True
            logger.info(f"All models loaded in {load_time:.2f}s")

    def _calculate_dynamic_k(
        self, n_subqueries: int, overlap_factor: float = 1.4
    ) -> int:
        """Calculate dynamic K for final chunk selection."""
        target_before_dedup = n_subqueries * self.chunks_per_subquery
        k_initial = min(
            self.max_cap, max(self.min_floor, int(target_before_dedup * overlap_factor))
        )
        return k_initial

    def run(
        self,
        query: str,
        retrieval_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run the web UI pipeline.

        Args:
            query: User query
            retrieval_top_k: Override retrieval top_k per sub-query
            rerank_top_k: Override chunks per sub-query after reranking

        Returns:
            PipelineResult with answer and metadata
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if retrieval_top_k:
            self.retrieval_top_k = retrieval_top_k
        if rerank_top_k:
            self.chunks_per_subquery = rerank_top_k

        pipeline_start = time.time()

        # Step 1: Decompose query
        step1_start = time.time()
        sub_queries = self.decomposer.decompose(query)
        step1_time = time.time() - step1_start
        n_subqueries = len(sub_queries)

        if n_subqueries == 0:
            raise ValueError("Query decomposition returned 0 sub-queries")

        # Step 2: Embed subqueries in parallel
        step2_start = time.time()

        def _embed_subquery(subquery: str, subquery_idx: int):
            """Embed a subquery using round-robin GPU assignment."""
            gpu_idx = self.embedder_gpus[subquery_idx % len(self.embedder_gpus)]
            embedder = self._get_embedder_for_gpu(gpu_idx)
            return (subquery_idx, embedder.embed_query(subquery), gpu_idx)

        query_embeddings_dict = {}
        with ThreadPoolExecutor(max_workers=len(self.embedder_gpus)) as executor:
            futures = {
                executor.submit(_embed_subquery, sq, i): i
                for i, sq in enumerate(sub_queries, 1)
            }
            for future in futures:
                subquery_idx, embedding, gpu_idx = future.result()
                query_embeddings_dict[subquery_idx] = embedding

        query_embeddings = [
            query_embeddings_dict[i] for i in range(1, n_subqueries + 1)
        ]
        step2_time = time.time() - step2_start

        # Step 3: Retrieve and rerank in parallel
        step3_start = time.time()

        def _retrieve_and_rerank(subquery: str, query_embedding, subquery_idx: int):
            """Retrieve and rerank for a subquery."""
            subquery_start = time.time()
            gpu_idx = self.reranker_gpus[subquery_idx % len(self.reranker_gpus)]

            # Retrieve
            retrieve_start = time.time()
            with self._retrieval_semaphore:
                chunks = retrieve_chunks(
                    query_embedding=query_embedding,
                    collection_name=self.collection_name,
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    top_k=self.retrieval_top_k,
                )
            retrieve_time = time.time() - retrieve_start
            retrieved_count = len(chunks)

            if not chunks:
                return {
                    "subquery": subquery,
                    "subquery_idx": subquery_idx,
                    "chunks": [],
                    "retrieved_count": retrieved_count,
                    "retrieve_time": retrieve_time,
                    "rerank_time": 0.0,
                    "total_time": time.time() - subquery_start,
                }

            # Rerank
            rerank_start = time.time()
            reranker = self._get_reranker_for_gpu(gpu_idx)
            reranked = reranker.rerank(subquery, chunks, top_k=self.chunks_per_subquery)
            rerank_time = time.time() - rerank_start

            reranked_dicts = [
                chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                for chunk in reranked
            ]

            return {
                "subquery": subquery,
                "subquery_idx": subquery_idx,
                "chunks": reranked_dicts,
                "retrieved_count": retrieved_count,
                "retrieve_time": retrieve_time,
                "rerank_time": rerank_time,
                "total_time": time.time() - subquery_start,
            }

        all_reranked = []
        all_retrieved = []
        with ThreadPoolExecutor(max_workers=len(self.reranker_gpus)) as executor:
            futures = {
                executor.submit(_retrieve_and_rerank, sq, emb, i): (sq, i)
                for i, (sq, emb) in enumerate(zip(sub_queries, query_embeddings), 1)
            }
            for future in futures:
                try:
                    result = future.result()
                    all_reranked.append(result)
                    all_retrieved.append(
                        {
                            "subquery": result["subquery"],
                            "count": result.get("retrieved_count", 0),
                        }
                    )
                except Exception as e:
                    subquery, subquery_idx = futures[future]
                    logger.error(
                        f"Retrieve+rerank failed for subquery {subquery_idx}: {e}"
                    )
                    all_reranked.append(
                        {
                            "subquery": subquery,
                            "subquery_idx": subquery_idx,
                            "chunks": [],
                            "retrieved_count": 0,
                            "retrieve_time": 0.0,
                            "rerank_time": 0.0,
                            "total_time": 0.0,
                        }
                    )

        all_reranked.sort(key=lambda x: x.get("subquery_idx", 0))
        step3_time = time.time() - step3_start

        # Step 4: Merge and select
        step4_start = time.time()
        seen_ids = set()
        merged_chunks = []
        for item in all_reranked:
            for chunk in item["chunks"]:
                if isinstance(chunk, Chunk):
                    chunk_dict = chunk.model_dump()
                else:
                    chunk_dict = chunk.copy()

                chunk_id = chunk_dict.get("chunk_id")
                if chunk_id is None:
                    text = chunk_dict.get("text", "")
                    if text:
                        import hashlib

                        chunk_id = f"text_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                    else:
                        chunk_id = (
                            f"rank_{chunk_dict.get('rank', 0)}_{item['subquery'][:20]}"
                        )

                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    chunk_dict["source_subquery"] = item["subquery"]
                    merged_chunks.append(chunk_dict)

        merged_chunks.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)

        overlap_factor = 1.4
        dynamic_k_initial = self._calculate_dynamic_k(n_subqueries, overlap_factor)
        n_unique = len(merged_chunks)
        dynamic_k_final = (
            min(dynamic_k_initial, n_unique)
            if n_unique < dynamic_k_initial
            else dynamic_k_initial
        )

        final_chunks_dicts = merged_chunks[:dynamic_k_final]
        final_chunks = []
        for rank, chunk_dict in enumerate(final_chunks_dicts, start=1):
            chunk_dict["final_rank"] = rank
            final_chunks.append(Chunk(**chunk_dict))
        step4_time = time.time() - step4_start

        # Step 5: Generate
        step5_start = time.time()
        generation_result = self.generator.generate(query, final_chunks)
        step5_time = time.time() - step5_start

        # Compile results
        from junrag.models import PipelineConfig

        config = PipelineConfig(
            embedding_model=self.embedding_model_name,
            reranker_model=self.reranker_model_name,
            llm_model=self.llm_model_name,
            decomposition_model=self.decomposition_model,
            retrieval_top_k=self.retrieval_top_k,
            rerank_top_k=self.chunks_per_subquery,
            max_cap=self.max_cap,
            min_floor=self.min_floor,
            chunks_per_subquery=self.chunks_per_subquery,
            max_concurrent_retrievals=self.max_concurrent_retrievals,
            tensor_parallel_size=8,
        )

        usage = UsageInfo(**generation_result.usage.model_dump())
        total_runtime = time.time() - pipeline_start

        result = PipelineResult(
            query=query,
            answer=generation_result.answer,
            pipeline="webui",
            config=config,
            usage=usage,
            sub_queries=sub_queries,
            n_subqueries=n_subqueries,
            dynamic_k_initial=dynamic_k_initial,
            dynamic_k_final=dynamic_k_final,
            n_unique_chunks=n_unique,
            retrieved_per_subquery=all_retrieved,
            reranked_per_subquery=[
                {"subquery": r["subquery"], "count": len(r["chunks"])}
                for r in all_reranked
            ],
            final_chunks=final_chunks,
        )

        # Add timing
        result.timing = {
            "total_runtime": total_runtime,
            "step1_decomposition": step1_time,
            "step2_embedding": step2_time,
            "step3_retrieve_rerank": step3_time,
            "step4_merge_select": step4_time,
            "step5_generation": step5_time,
        }

        return result
