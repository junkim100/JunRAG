"""
Full Pipeline for JunRAG.
Advanced flow with query decomposition and parallel processing.

Flow: Query → Decompose → Batch Embed → Parallel Retrieve → Sequential Rerank → Dynamic Top-K → Generate

Design:
- Embedding: Batch process all subqueries at once (fast, single model)
- Retrieval: Parallel Qdrant queries (no GPU needed, use semaphore for rate limiting)
- Reranking: Sequential processing to avoid GPU conflicts (single reranker)
- All operations complete before proceeding to the next step (barrier pattern)
"""

import logging
import os
import threading
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
    2. Embed ALL sub-queries in batch (single embedder for efficiency)
    3. Retrieve chunks for each sub-query in parallel (Qdrant, no GPU)
    4. Rerank chunks for each sub-query sequentially (single GPU reranker)
    5. Merge and select dynamic top-K chunks
    6. Generate final answer

    GPU Usage:
    - Embedding: Single model with configurable tensor_parallel_size
    - Reranking: Single model with tensor_parallel_size=1 (to avoid conflicts)
    - Generation: OpenAI API (no local GPU)

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
        # Force tensor_parallel_size=1 for embedder and reranker in full pipeline
        # to ensure single GPU usage and avoid conflicts
        kwargs["tensor_parallel_size"] = 1
        super().__init__(**kwargs)

        self.max_cap = max_cap
        self.min_floor = min_floor
        self.chunks_per_subquery = chunks_per_subquery
        self.max_concurrent_retrievals = max_concurrent_retrievals
        self.decomposition_model = decomposition_model

        # Detect available GPUs (for logging)
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"Full pipeline detected {self.num_gpus} GPUs")
        logger.info("Using tensor_parallel_size=1 for embedder and reranker")

        # Semaphore for Qdrant retrieval rate limiting
        self._retrieval_semaphore = threading.Semaphore(max_concurrent_retrievals)

        # Lock for sequential reranking
        self._rerank_lock = threading.Lock()

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

    def _retrieve_for_subquery(
        self,
        subquery: str,
        query_embedding,
        subquery_idx: int,
    ) -> Dict:
        """
        Retrieve chunks for a single sub-query from Qdrant.

        Uses semaphore for rate limiting. No GPU required.

        Args:
            subquery: The sub-query text
            query_embedding: Pre-computed embedding
            subquery_idx: Index for logging

        Returns:
            Dict with subquery and retrieved chunks
        """
        with self._retrieval_semaphore:
            logger.debug(f"Subquery {subquery_idx}: Retrieving from Qdrant...")
        chunks = retrieve_chunks(
            query_embedding=query_embedding,
            collection_name=self.collection_name,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            top_k=self.retrieval_top_k,
        )
        return {
            "subquery": subquery,
            "subquery_idx": subquery_idx,
            "chunks": chunks,
        }

    def _rerank_for_subquery(
        self,
        subquery: str,
        chunks: List[Dict],
        subquery_idx: int,
    ) -> Dict:
        """
        Rerank chunks for a single sub-query.

        Uses lock to ensure sequential GPU access.

        Args:
            subquery: The sub-query text
            chunks: Retrieved chunks to rerank
            subquery_idx: Index for logging

        Returns:
            Dict with subquery and reranked chunks
        """
        with self._rerank_lock:
            logger.debug(f"Subquery {subquery_idx}: Reranking {len(chunks)} chunks...")
        reranked = self.reranker.rerank(
            subquery, chunks, top_k=self.chunks_per_subquery
        )
        return {
            "subquery": subquery,
            "subquery_idx": subquery_idx,
            "chunks": reranked,
        }

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

        print(f"\n{'='*80}")
        print("FULL PIPELINE (with Query Decomposition)")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Collection: {self.collection_name}")
        print(f"Available GPUs: {self.num_gpus} (using tensor_parallel_size=1)")

        # Step 1: Decompose query
        print(f"\n[Step 1] Decomposing query...")
        sub_queries = self.decomposer.decompose(query)
        n_subqueries = len(sub_queries)

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

        # Warn if more subqueries than GPUs (for future parallel processing scenarios)
        if n_subqueries > self.num_gpus and self.num_gpus > 0:
            logger.warning(
                f"Number of sub-queries ({n_subqueries}) exceeds available GPUs ({self.num_gpus}). "
                "Reranking will be sequential. For optimal parallel processing, "
                "ensure num_subqueries <= num_gpus."
            )
            print(
                f"\nNOTE: {n_subqueries} sub-queries > {self.num_gpus} GPUs. "
                "Reranking will process sequentially (one at a time)."
            )

        print(f"Decomposed into {n_subqueries} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"  {i}. {sq}")

        # Calculate initial dynamic K (Stage 1: before deduplication)
        overlap_factor = 1.4  # Accounts for ~30% expected overlap
        dynamic_k_initial = self._calculate_dynamic_k(n_subqueries, overlap_factor)
        print(
            f"\nDynamic K (Stage 1 - initial): "
            f"min({self.max_cap}, max({self.min_floor}, "
            f"{n_subqueries} × {self.chunks_per_subquery} × {overlap_factor})) = {dynamic_k_initial}"
        )

        # Step 2: Batch embed ALL sub-queries at once (single embedder)
        print(f"\n[Step 2] Batch embedding {n_subqueries} sub-queries...")
        query_embeddings = self.embedder.embed_queries(sub_queries)
        print(f"Embedded all {n_subqueries} sub-queries")

        # Step 3: Parallel retrieval from Qdrant (no GPU, semaphore-controlled)
        print(
            f"\n[Step 3] Retrieving chunks in parallel "
            f"(max {self.max_concurrent_retrievals} concurrent)..."
        )

        all_retrieved = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent_retrievals) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(self._retrieve_for_subquery, sq, emb, i): sq
                for i, (sq, emb) in enumerate(zip(sub_queries, query_embeddings), 1)
            }

            # Wait for ALL retrievals to complete (barrier)
            done, _ = wait(futures.keys(), return_when=ALL_COMPLETED)

            # Collect results
            for future in done:
                try:
                    result = future.result()
                    all_retrieved.append(result)
                    print(
                        f"  [{result['subquery_idx']}] Retrieved {len(result['chunks'])} chunks "
                        f"for: {result['subquery'][:50]}..."
                    )
                except Exception as e:
                    subquery = futures[future]
                    logger.error(f"Retrieval failed for '{subquery}': {e}")
                    print(f"  ERROR: {subquery[:50]}... - {e}")
                    # Add empty result to maintain consistency
                    all_retrieved.append(
                        {
                            "subquery": subquery,
                            "subquery_idx": -1,
                            "chunks": [],
                        }
                    )

        # Sort by subquery index for consistent ordering
        all_retrieved.sort(key=lambda x: x.get("subquery_idx", 0))

        total_retrieved = sum(len(r["chunks"]) for r in all_retrieved)
        print(f"Total retrieved: {total_retrieved} chunks")

        # Step 4: Sequential reranking (single GPU, lock-controlled)
        print(f"\n[Step 4] Reranking chunks sequentially (single GPU)...")

        all_reranked = []
        for item in all_retrieved:
            if not item["chunks"]:
                all_reranked.append(
                    {
                        "subquery": item["subquery"],
                        "subquery_idx": item["subquery_idx"],
                        "chunks": [],
                    }
                )
                continue

            result = self._rerank_for_subquery(
                item["subquery"],
                item["chunks"],
                item["subquery_idx"],
            )
            all_reranked.append(result)
            print(
                f"  [{result['subquery_idx']}] Reranked to {len(result['chunks'])} chunks "
                f"for: {result['subquery'][:50]}..."
            )

        total_reranked = sum(len(r["chunks"]) for r in all_reranked)
        print(f"Total reranked: {total_reranked} chunks")

        # Step 5: Merge and select dynamic top-K (Two-stage approach)
        print(f"\n[Step 5] Merging and selecting chunks (two-stage dynamic K)...")

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

        print(
            f"\nFinal selection: {len(final_chunks)} chunks "
            f"(from {n_unique} total unique, initial target was {dynamic_k_initial})"
        )

        # Check for missing text
        missing_text = sum(1 for c in final_chunks if not c.text)
        if missing_text > 0:
            print(f"Warning: {missing_text}/{len(final_chunks)} chunks have no text.")

        # Step 6: Generate answer
        print(f"\n[Step 6] Generating final answer...")
        generation_result = self.generator.generate(query, final_chunks)
        print(f"Answer generated ({generation_result.usage.total_tokens} tokens)")

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
            tensor_parallel_size=1,  # Always 1 for full pipeline
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
                {"subquery": r["subquery"], "count": len(r["chunks"])}
                for r in all_retrieved
            ],
            reranked_per_subquery=[
                {"subquery": r["subquery"], "count": len(r["chunks"])}
                for r in all_reranked
            ],
            final_chunks=final_chunks_models,
        )

        print(f"\n{'='*80}")
        print("FULL PIPELINE COMPLETE")
        print(f"{'='*80}")

        return result
