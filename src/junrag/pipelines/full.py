"""
Full Pipeline for JunRAG.
Advanced flow with query decomposition and parallel processing.

Flow: Query → Decompose → Parallel Embed → Parallel Retrieve+Rerank → Dynamic Top-K → Generate

Design:
- Embedding: Parallel processing on assigned GPUs (one per subquery)
- Retrieval + Reranking: Grouped per subquery, executed in parallel across all subqueries
  - Each subquery: retrieve → rerank sequentially on its assigned GPU
  - Different subqueries run in parallel, allowing overlap of operations
- Semaphore limits concurrent Qdrant calls to prevent server overload
- GPU assignment ensures each subquery uses a dedicated GPU for embedding and reranking
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
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


class FullPipeline(BasePipeline):
    """
    Full RAG pipeline with query decomposition and parallel processing.

    Flow:
    1. Decompose complex query into single-hop sub-queries
    2. Embed all sub-queries in parallel (each on assigned GPU)
    3. Retrieve and rerank chunks for each sub-query in parallel
       - Each subquery: retrieve → rerank sequentially on assigned GPU
       - Different subqueries run in parallel, maximizing GPU utilization
    4. Merge and select dynamic top-K chunks
    5. Generate final answer

    GPU Usage:
    - Embedding: Parallel processing, one embedder per subquery on assigned GPU
    - Reranking: Parallel processing, one reranker per subquery on assigned GPU
    - Generation: OpenAI API (no local GPU)

    Parallelism:
    - Embedding: All subqueries embedded in parallel (up to available_gpus workers)
    - Retrieval+Reranking: All subqueries processed in parallel
      - Semaphore limits concurrent Qdrant calls (max_concurrent_retrievals)
      - Each subquery uses its assigned GPU independently
      - Operations overlap: while one subquery retrieves, others can rerank

    Dynamic K formula (Two-stage approach):
    Stage 1: K_initial = min(MAX_CAP, max(MIN_FLOOR, N_sub × chunks_per_subquery × overlap_factor))
    Stage 2: After deduplication, K_final = min(K_initial, n_unique_chunks)

    This accounts for expected overlap between subqueries and ensures we use all
    available unique chunks if fewer than the initial target.
    """

    def __init__(
        self,
        # Full pipeline specific settings
        max_cap: int = 25,
        min_floor: int = 5,
        chunks_per_subquery: int = 10,
        max_concurrent_retrievals: int = 8,
        decomposition_model: str = "gpt-4o",
        # Inherited settings
        **kwargs,
    ):
        """
        Initialize full pipeline.

        Args:
            max_cap: Maximum chunks for final selection (MAX_CAP)
            min_floor: Minimum chunks for final selection (MIN_FLOOR)
            chunks_per_subquery: Chunks per sub-query after reranking (M)
            max_concurrent_retrievals: Max parallel Qdrant retrieval operations
            decomposition_model: Model for query decomposition
            **kwargs: Arguments passed to BasePipeline
        """
        # Don't force tensor_parallel_size=1 - allow configurable GPU assignment
        # tensor_parallel_size now represents available GPUs for per-subquery assignment
        super().__init__(**kwargs)

        self.max_cap = max_cap
        self.min_floor = min_floor
        self.chunks_per_subquery = chunks_per_subquery
        self.max_concurrent_retrievals = max_concurrent_retrievals
        self.decomposition_model = decomposition_model

        # Detect available GPUs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # tensor_parallel_size represents the number of GPUs available for assignment
        # Each subquery will use a single GPU from this pool
        self.available_gpus = min(self.tensor_parallel_size, self.num_gpus)
        if self.available_gpus == 0:
            self.available_gpus = 1  # Fallback to at least 1

        logger.info(f"Full pipeline detected {self.num_gpus} GPUs")
        logger.info(
            f"Using {self.available_gpus} GPUs for per-subquery assignment (tensor_parallel_size={self.tensor_parallel_size})"
        )

        # Semaphore for Qdrant retrieval rate limiting
        self._retrieval_semaphore = threading.Semaphore(max_concurrent_retrievals)

        # Lock for sequential creation of embedders/rerankers to avoid CUDA_VISIBLE_DEVICES conflicts
        self._model_creation_lock = threading.Lock()

        # Per-subquery embedders and rerankers (lazy initialization)
        self._subquery_embedders = {}
        self._subquery_rerankers = {}

        # Lazy initialization
        self._decomposer = None

        # Completion barrier
        self._completion_barrier = None

    @property
    def decomposer(self) -> QueryDecomposer:
        """Lazy load query decomposer."""
        if self._decomposer is None:
            self._decomposer = QueryDecomposer(
                model=self.decomposition_model,
                api_key=self.openai_api_key,
            )
        return self._decomposer

    def _get_gpu_for_subquery(self, subquery_idx: int) -> int:
        """
        Get GPU assignment for a subquery.

        Args:
            subquery_idx: Subquery index (1-based)

        Returns:
            GPU index (0-based)
        """
        # Assign GPUs round-robin: subquery 1 -> GPU 0, subquery 2 -> GPU 1, etc.
        gpu_idx = (subquery_idx - 1) % self.available_gpus
        return gpu_idx

    def _get_embedder_for_subquery(self, subquery_idx: int):
        """
        Get embedder for a specific subquery with GPU assignment.

        Args:
            subquery_idx: Subquery index (1-based)

        Returns:
            EmbeddingModel instance assigned to specific GPU
        """
        if subquery_idx not in self._subquery_embedders:
            with self._model_creation_lock:
                # Double-check after acquiring lock
                if subquery_idx not in self._subquery_embedders:
                    gpu_idx = self._get_gpu_for_subquery(subquery_idx)

                    # Set CUDA_VISIBLE_DEVICES to only see the assigned GPU
                    # This makes vLLM use GPU 0 (which is actually the assigned GPU)
                    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    try:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                        logger.debug(
                            f"Subquery {subquery_idx}: Creating embedder on GPU {gpu_idx}"
                        )

                        # Create embedder with tensor_parallel_size=1 (single GPU per subquery)
                        from junrag.components.embedding import EmbeddingModel

                        self._subquery_embedders[subquery_idx] = EmbeddingModel(
                            model=self.embedding_model_name,
                            tensor_parallel_size=1,  # Single GPU per subquery
                            gpu_memory_utilization=self.gpu_memory_utilization,
                            max_model_len=self.max_model_len,
                        )
                    finally:
                        # Restore original CUDA_VISIBLE_DEVICES
                        if original_cuda_visible is not None:
                            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                        elif "CUDA_VISIBLE_DEVICES" in os.environ:
                            del os.environ["CUDA_VISIBLE_DEVICES"]

        return self._subquery_embedders[subquery_idx]

    def _get_reranker_for_subquery(self, subquery_idx: int):
        """
        Get reranker for a specific subquery with GPU assignment.

        Args:
            subquery_idx: Subquery index (1-based)

        Returns:
            Reranker instance assigned to specific GPU
        """
        if subquery_idx not in self._subquery_rerankers:
            with self._model_creation_lock:
                # Double-check after acquiring lock
                if subquery_idx not in self._subquery_rerankers:
                    gpu_idx = self._get_gpu_for_subquery(subquery_idx)

                    logger.debug(
                        f"Subquery {subquery_idx}: Creating reranker on GPU {gpu_idx}"
                    )

                    # For vLLM reranker, use CUDA_VISIBLE_DEVICES
                    # For transformers reranker, use device="cuda:X"
                    from junrag.components.reranking import Reranker

                    # Check if it's a vLLM reranker (Qwen3)
                    use_vllm = (
                        "qwen" in self.reranker_model_name.lower()
                        and "reranker" in self.reranker_model_name.lower()
                    )

                    if use_vllm:
                        # For vLLM, use CUDA_VISIBLE_DEVICES
                        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                        try:
                            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                            self._subquery_rerankers[subquery_idx] = Reranker(
                                model=self.reranker_model_name,
                                batch_size=self.rerank_batch_size,
                                use_multi_gpu=False,  # Single GPU per subquery
                                tensor_parallel_size=1,  # Single GPU per subquery
                                gpu_memory_utilization=self.gpu_memory_utilization,
                                max_model_len=self.max_model_len,
                            )
                        finally:
                            if original_cuda_visible is not None:
                                os.environ["CUDA_VISIBLE_DEVICES"] = (
                                    original_cuda_visible
                                )
                            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                                del os.environ["CUDA_VISIBLE_DEVICES"]
                    else:
                        # For transformers, use device="cuda:X"
                        device = (
                            f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"
                        )
                        self._subquery_rerankers[subquery_idx] = Reranker(
                            model=self.reranker_model_name,
                            device=device,
                            batch_size=self.rerank_batch_size,
                            use_multi_gpu=False,  # Single GPU per subquery
                            tensor_parallel_size=1,
                            gpu_memory_utilization=self.gpu_memory_utilization,
                            max_model_len=self.max_model_len,
                        )

        return self._subquery_rerankers[subquery_idx]

    def _calculate_dynamic_k(
        self, n_subqueries: int, overlap_factor: float = 1.4
    ) -> int:
        """
        Calculate initial dynamic K for final chunk selection (Stage 1).

        Two-stage approach:
        Stage 1: Calculate target accounting for expected overlap
        Stage 2: After deduplication, adjust if unique chunks < K_initial

        Formula: K_initial = min(MAX_CAP, max(MIN_FLOOR, N_sub × chunks_per_subquery × overlap_factor))

        Args:
            n_subqueries: Number of sub-queries
            overlap_factor: Multiplier to account for expected overlap (default 1.4 = 30% buffer)

        Returns:
            Initial K value (may be adjusted after deduplication)
        """
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
        Run the full pipeline with query decomposition.

        Args:
            query: User query (can be complex multi-hop)
            retrieval_top_k: Override retrieval top_k per sub-query
            rerank_top_k: Override chunks per sub-query after reranking

        Returns:
            Dictionary with answer and pipeline results
        """
        if retrieval_top_k:
            self.retrieval_top_k = retrieval_top_k
        if rerank_top_k:
            self.chunks_per_subquery = rerank_top_k

        # Start timing
        pipeline_start_time = time.time()

        print(f"\n{'='*80}")
        print("FULL PIPELINE (with Query Decomposition)")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Collection: {self.collection_name}")
        print(
            f"Available GPUs: {self.num_gpus} (using {self.available_gpus} for per-subquery assignment)"
        )

        # Step 1: Decompose query
        print(f"\n[Step 1] Decomposing query...")
        step1_start = time.time()
        sub_queries = self.decomposer.decompose(query)
        step1_time = time.time() - step1_start
        n_subqueries = len(sub_queries)
        print(f"  Decomposition completed in {step1_time:.2f} seconds")

        # Validate decomposition results
        if n_subqueries == 0:
            logger.error("Query decomposition returned 0 sub-queries")
            raise ValueError(
                "Query decomposition failed: No sub-queries generated. "
                "The query may be too simple or the decomposition model failed."
            )

        if n_subqueries > 20:
            logger.warning(
                f"Query decomposed into {n_subqueries} sub-queries, which is unusually high. "
                "This may indicate an issue with decomposition or a very complex query."
            )
            print(
                f"\nWARNING: {n_subqueries} sub-queries is unusually high. "
                "Consider simplifying the query or checking decomposition results."
            )

        # Log GPU assignment strategy
        if n_subqueries > self.available_gpus:
            logger.info(
                f"Number of sub-queries ({n_subqueries}) > available GPUs ({self.available_gpus}). "
                "GPUs will be assigned round-robin."
            )
            print(
                f"\nNOTE: {n_subqueries} sub-queries > {self.available_gpus} GPUs. "
                "GPUs will be assigned round-robin (subquery 1->GPU 0, subquery 2->GPU 1, etc.)."
            )
        else:
            print(
                f"\nGPU Assignment: Each of {n_subqueries} sub-queries will use a dedicated GPU "
                f"(from {self.available_gpus} available GPUs)."
            )

        print(f"Decomposed into {n_subqueries} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"  {i}. {sq}")

        # Pre-initialize all embedders and rerankers in parallel to avoid cold starts
        print(f"\n[Pre-initialization] Loading models on assigned GPUs in parallel...")
        preinit_start = time.time()

        def _preinit_models_for_subquery(subquery_idx: int):
            """Pre-initialize embedder and reranker for a subquery."""
            gpu_idx = self._get_gpu_for_subquery(subquery_idx)
            logger.debug(
                f"Pre-initializing models for subquery {subquery_idx} on GPU {gpu_idx}"
            )
            # Initialize both embedder and reranker
            embedder = self._get_embedder_for_subquery(subquery_idx)
            reranker = self._get_reranker_for_subquery(subquery_idx)
            return subquery_idx

        # Pre-initialize all models in parallel
        max_preinit_workers = min(n_subqueries, self.available_gpus)
        with ThreadPoolExecutor(max_workers=max_preinit_workers) as executor:
            futures = {
                executor.submit(_preinit_models_for_subquery, i): i
                for i in range(1, n_subqueries + 1)
            }
            for future in futures:
                try:
                    subquery_idx = future.result()
                    logger.debug(f"Models initialized for subquery {subquery_idx}")
                except Exception as e:
                    subquery_idx = futures[future]
                    logger.error(
                        f"Failed to pre-initialize models for subquery {subquery_idx}: {e}"
                    )
                    raise

        preinit_time = time.time() - preinit_start
        print(f"  All models pre-initialized in {preinit_time:.2f}s")

        # Calculate initial dynamic K (Stage 1: before deduplication)
        overlap_factor = 1.4  # Accounts for ~30% expected overlap
        dynamic_k_initial = self._calculate_dynamic_k(n_subqueries, overlap_factor)
        print(
            f"\nDynamic K (Stage 1 - initial): "
            f"min({self.max_cap}, max({self.min_floor}, "
            f"{n_subqueries} × {self.chunks_per_subquery} × {overlap_factor})) = {dynamic_k_initial}"
        )

        # Step 2: Embed all sub-queries in parallel on assigned GPUs
        print(
            f"\n[Step 2] Embedding {n_subqueries} sub-queries in parallel on assigned GPUs..."
        )

        def _embed_subquery(subquery: str, subquery_idx: int):
            """Embed a single subquery on its assigned GPU."""
            embed_start = time.time()
            gpu_idx = self._get_gpu_for_subquery(subquery_idx)
            logger.debug(f"Subquery {subquery_idx}: Embedding on GPU {gpu_idx}...")
            embedder = self._get_embedder_for_subquery(subquery_idx)
            embedding = embedder.embed_query(subquery)
            embed_time = time.time() - embed_start
            print(
                f"  Subquery {subquery_idx}: Embedded on GPU {gpu_idx} in {embed_time:.2f}s"
            )
            return (subquery_idx, embedding, embed_time)

        # Embed in parallel (one per GPU, up to available_gpus)
        step2_start = time.time()
        query_embeddings_dict = {}
        embedding_times = {}
        max_embed_workers = min(n_subqueries, self.available_gpus)
        with ThreadPoolExecutor(max_workers=max_embed_workers) as executor:
            futures = {
                executor.submit(_embed_subquery, sq, i): i
                for i, sq in enumerate(sub_queries, 1)
            }
            for future in futures:
                try:
                    subquery_idx, embedding, embed_time = future.result()
                    query_embeddings_dict[subquery_idx] = embedding
                    embedding_times[subquery_idx] = embed_time
                except Exception as e:
                    subquery_idx = futures[future]
                    logger.error(f"Embedding failed for subquery {subquery_idx}: {e}")
                    raise

        # Convert to list in order
        query_embeddings = [
            query_embeddings_dict[i] for i in range(1, n_subqueries + 1)
        ]
        step2_time = time.time() - step2_start
        max_embed_time = max(embedding_times.values()) if embedding_times else 0
        print(f"Embedded all {n_subqueries} sub-queries")
        print(
            f"  Total embedding time: {step2_time:.2f}s (longest subquery: {max_embed_time:.2f}s)"
        )

        # Step 3: Parallel retrieval + reranking per subquery
        # Each subquery: retrieve → rerank (on assigned GPU) as a single parallel task
        print(
            f"\n[Step 3] Retrieving and reranking chunks in parallel "
            f"(max {self.max_concurrent_retrievals} concurrent operations)..."
        )

        def _retrieve_and_rerank_subquery(
            subquery: str, query_embedding, subquery_idx: int
        ):
            """
            Retrieve and rerank chunks for a single subquery.
            This function groups retrieval and reranking together so they happen
            sequentially per subquery, but different subqueries can run in parallel.
            """
            subquery_start = time.time()
            gpu_idx = self._get_gpu_for_subquery(subquery_idx)

            # Retrieve chunks (uses semaphore for rate limiting)
            # Semaphore protects the Qdrant call to prevent overwhelming the server
            retrieve_start = time.time()
            retrieved_count = 0
            try:
                with self._retrieval_semaphore:
                    logger.debug(f"Subquery {subquery_idx}: Retrieving from Qdrant...")
                    chunks = retrieve_chunks(
                        query_embedding=query_embedding,
                        collection_name=self.collection_name,
                        url=self.qdrant_url,
                        api_key=self.qdrant_api_key,
                        top_k=self.retrieval_top_k,
                    )
                retrieved_count = len(chunks)
                retrieve_time = time.time() - retrieve_start
                print(
                    f"  [{subquery_idx}] Retrieved {retrieved_count} chunks "
                    f"in {retrieve_time:.2f}s for: {subquery[:50]}..."
                )
            except Exception as e:
                logger.error(f"Retrieval failed for subquery {subquery_idx}: {e}")
                print(f"  ERROR: Subquery {subquery_idx} retrieval failed - {e}")
                chunks = []
                retrieve_time = time.time() - retrieve_start

            # Rerank chunks on assigned GPU (if any chunks retrieved)
            if not chunks:
                subquery_time = time.time() - subquery_start
                return {
                    "subquery": subquery,
                    "subquery_idx": subquery_idx,
                    "chunks": [],
                    "retrieved_count": retrieved_count,
                    "retrieve_time": retrieve_time,
                    "rerank_time": 0.0,
                    "total_time": subquery_time,
                }

            try:
                rerank_start = time.time()
                logger.debug(
                    f"Subquery {subquery_idx}: Reranking {len(chunks)} chunks on GPU {gpu_idx}..."
                )
                reranker = self._get_reranker_for_subquery(subquery_idx)
                reranked = reranker.rerank(
                    subquery, chunks, top_k=self.chunks_per_subquery
                )
                rerank_time = time.time() - rerank_start

                # Convert Chunk models back to dicts for consistency
                reranked_dicts = [
                    chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    for chunk in reranked
                ]

                subquery_time = time.time() - subquery_start
                print(
                    f"  [{subquery_idx}] Reranked to {len(reranked_dicts)} chunks "
                    f"in {rerank_time:.2f}s on GPU {gpu_idx} (total: {subquery_time:.2f}s) "
                    f"for: {subquery[:50]}..."
                )

                return {
                    "subquery": subquery,
                    "subquery_idx": subquery_idx,
                    "chunks": reranked_dicts,
                    "retrieved_count": retrieved_count,
                    "retrieve_time": retrieve_time,
                    "rerank_time": rerank_time,
                    "total_time": subquery_time,
                }
            except Exception as e:
                logger.error(f"Reranking failed for subquery {subquery_idx}: {e}")
                print(f"  ERROR: Subquery {subquery_idx} reranking failed - {e}")
                subquery_time = time.time() - subquery_start
                return {
                    "subquery": subquery,
                    "subquery_idx": subquery_idx,
                    "chunks": [],
                    "retrieved_count": retrieved_count,
                    "retrieve_time": retrieve_time,
                    "rerank_time": 0.0,
                    "total_time": subquery_time,
                }

        # Execute retrieval + reranking in parallel for all subqueries
        step3_start = time.time()
        # Use max of: retrieval semaphore limit, available GPUs, and number of subqueries
        # This ensures we can fully utilize all GPUs while respecting Qdrant rate limits
        all_reranked = []
        all_retrieved = []  # Track retrieved counts for reporting
        max_workers = max(
            min(
                self.max_concurrent_retrievals, n_subqueries
            ),  # At least respect retrieval limit
            min(self.available_gpus, n_subqueries),  # But also utilize all GPUs
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all retrieve+rerank tasks
            futures = {
                executor.submit(_retrieve_and_rerank_subquery, sq, emb, i): (sq, i)
                for i, (sq, emb) in enumerate(zip(sub_queries, query_embeddings), 1)
            }

            # Wait for all operations to complete
            done, _ = wait(futures.keys(), return_when=ALL_COMPLETED)

            # Collect results
            for future in done:
                try:
                    result = future.result()
                    all_reranked.append(result)
                    # Track retrieved count for reporting
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
                    print(f"  ERROR: Subquery {subquery_idx} - {e}")
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
                    all_retrieved.append(
                        {
                            "subquery": subquery,
                            "count": 0,
                        }
                    )

        # Sort by subquery index for consistent ordering
        all_reranked.sort(key=lambda x: x.get("subquery_idx", 0))
        step3_time = time.time() - step3_start

        total_reranked = sum(len(r["chunks"]) for r in all_reranked)
        max_subquery_time = max(
            (r.get("total_time", 0) for r in all_reranked), default=0
        )
        avg_subquery_time = (
            sum((r.get("total_time", 0) for r in all_reranked)) / len(all_reranked)
            if all_reranked
            else 0
        )
        print(f"Total reranked: {total_reranked} chunks")
        print(
            f"  Parallel retrieve+rerank time: {step3_time:.2f}s (longest subquery: {max_subquery_time:.2f}s, avg: {avg_subquery_time:.2f}s)"
        )

        # Step 5: Merge and select dynamic top-K (Two-stage approach)
        print(f"\n[Step 5] Merging and selecting chunks (two-stage dynamic K)...")
        step5_start = time.time()

        # Flatten and deduplicate chunks
        seen_ids = set()
        merged_chunks = []
        for item in all_reranked:
            for chunk in item["chunks"]:
                # Handle both Chunk models and dicts
                if isinstance(chunk, Chunk):
                    chunk_dict = chunk.model_dump()
                else:
                    chunk_dict = chunk.copy()

                chunk_id = chunk_dict.get("chunk_id")
                # Handle None chunk_id (use a fallback key)
                if chunk_id is None:
                    # Fallback: use text hash or composite key
                    text = chunk_dict.get("text", "")
                    if text:
                        import hashlib

                        chunk_id = f"text_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                    else:
                        # Last resort: use rank + subquery
                        chunk_id = (
                            f"rank_{chunk_dict.get('rank', 0)}_{item['subquery'][:20]}"
                        )

                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    # Add source sub-query info
                    chunk_dict["source_subquery"] = item["subquery"]
                    merged_chunks.append(chunk_dict)

        # Sort by reranker score
        merged_chunks.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)

        # Stage 2: Adjust K after deduplication
        n_unique = len(merged_chunks)
        if n_unique < dynamic_k_initial:
            # Fewer unique chunks than initial target - use all available
            dynamic_k_final = n_unique
            print(
                f"  Stage 2 adjustment: {n_unique} unique chunks < initial K ({dynamic_k_initial})"
            )
            print(f"  Using all {n_unique} unique chunks")
        else:
            # Enough unique chunks - use initial target
            dynamic_k_final = dynamic_k_initial
            print(
                f"  Stage 2: {n_unique} unique chunks >= initial K ({dynamic_k_initial})"
            )
            print(f"  Using top {dynamic_k_final} chunks")

        # Select final chunks
        final_chunks_dicts = merged_chunks[:dynamic_k_final]

        # Update ranks and convert to Chunk models
        final_chunks = []
        for rank, chunk_dict in enumerate(final_chunks_dicts, start=1):
            chunk_dict["final_rank"] = rank
            final_chunks.append(Chunk(**chunk_dict))

        step5_time = time.time() - step5_start
        print(
            f"\nFinal selection: {len(final_chunks)} chunks "
            f"(from {n_unique} total unique, initial target was {dynamic_k_initial})"
        )
        print(f"  Merging and selection completed in {step5_time:.2f}s")

        # Check for missing text
        missing_text = sum(1 for c in final_chunks if not c.text)
        if missing_text > 0:
            print(f"Warning: {missing_text}/{len(final_chunks)} chunks have no text.")

        # Step 6: Generate answer
        print(f"\n[Step 6] Generating final answer...")
        step6_start = time.time()
        generation_result = self.generator.generate(query, final_chunks)
        step6_time = time.time() - step6_start
        print(
            f"Answer generated ({generation_result.usage.total_tokens} tokens) in {step6_time:.2f}s"
        )

        # Convert final_chunks to Chunk models if needed
        final_chunks_models = [
            chunk if isinstance(chunk, Chunk) else Chunk(**chunk)
            for chunk in final_chunks
        ]

        # Compile results
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
            tensor_parallel_size=self.tensor_parallel_size,  # Number of GPUs available for assignment
        )

        usage = UsageInfo(**generation_result.usage.model_dump())

        result = PipelineResult(
            query=query,
            answer=generation_result.answer,
            pipeline="full",
            config=config,
            usage=usage,
            sub_queries=sub_queries,
            n_subqueries=n_subqueries,
            dynamic_k_initial=dynamic_k_initial,
            dynamic_k_final=dynamic_k_final,
            n_unique_chunks=n_unique,
            retrieved_per_subquery=[
                {"subquery": r["subquery"], "count": r["count"]} for r in all_retrieved
            ],
            reranked_per_subquery=[
                {"subquery": r["subquery"], "count": len(r["chunks"])}
                for r in all_reranked
            ],
            final_chunks=final_chunks_models,
        )

        # Calculate total runtime
        pipeline_end_time = time.time()
        total_runtime = pipeline_end_time - pipeline_start_time

        # Compile timing information
        timing_info = {
            "total_runtime": total_runtime,
            "preinit_models": preinit_time,
            "step1_decomposition": step1_time,
            "step2_embedding": {
                "total_time": step2_time,
                "longest_subquery": max_embed_time,
                "per_subquery": embedding_times,
            },
            "step3_retrieve_rerank": {
                "total_time": step3_time,
                "longest_subquery": max_subquery_time,
                "avg_subquery": avg_subquery_time,
                "per_subquery": [
                    {
                        "subquery_idx": r["subquery_idx"],
                        "retrieve_time": r.get("retrieve_time", 0),
                        "rerank_time": r.get("rerank_time", 0),
                        "total_time": r.get("total_time", 0),
                    }
                    for r in all_reranked
                ],
            },
            "step5_merge_select": step5_time,
            "step6_generation": step6_time,
        }

        print(f"\n{'='*80}")
        print("FULL PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(
            f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)"
        )
        print(f"\nTiming breakdown:")
        print(f"  Pre-initialization (Model Loading): {preinit_time:.2f}s")
        print(f"  Step 1 (Decomposition): {step1_time:.2f}s")
        print(
            f"  Step 2 (Embedding): {step2_time:.2f}s (longest: {max_embed_time:.2f}s)"
        )
        print(
            f"  Step 3 (Retrieve+Rerank): {step3_time:.2f}s (longest: {max_subquery_time:.2f}s, avg: {avg_subquery_time:.2f}s)"
        )
        print(f"  Step 5 (Merge+Select): {step5_time:.2f}s")
        print(f"  Step 6 (Generation): {step6_time:.2f}s")

        # Add timing to result (using extra="allow" in PipelineResult)
        result.timing = timing_info

        return result
