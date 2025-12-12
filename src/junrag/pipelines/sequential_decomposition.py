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

import logging
import os
import time
from typing import Dict, List, Optional

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
    - Retriever and reranker are loaded once and reused for all subqueries
    - GPU split: retriever gets floor(gpus/2), reranker gets ceil(gpus/2)
    - For odd tensor_parallel_size (e.g., 7): retriever=3 GPUs, reranker=4 GPUs
    """

    def __init__(
        self,
        # Sequential pipeline specific settings
        decomposition_model: str = "gpt-5-mini-2025-08-07",
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
        self._embedder = None
        self._reranker_model = None

    def _get_retriever_gpu_ids(self) -> str:
        """Get comma-separated GPU IDs for retriever/embedder."""
        if self.retriever_gpus <= 0:
            return "0"
        # Retriever uses first half of GPUs
        gpu_ids = list(range(self.retriever_gpus))
        return ",".join(str(g) for g in gpu_ids)

    def _get_reranker_gpu_ids(self) -> str:
        """Get comma-separated GPU IDs for reranker."""
        if self.reranker_gpus <= 0:
            return "0"
        # Reranker uses second half of GPUs
        start_gpu = self.retriever_gpus
        gpu_ids = list(range(start_gpu, start_gpu + self.reranker_gpus))
        return ",".join(str(g) for g in gpu_ids)

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
        """Lazy load embedding model with dedicated GPU assignment."""
        if self._embedder is None:
            gpu_ids = self._get_retriever_gpu_ids()
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                logger.info(
                    f"Loading embedder on GPUs: {gpu_ids} "
                    f"(tensor_parallel_size={self.retriever_gpus})"
                )

                self._embedder = EmbeddingModel(
                    model=self.embedding_model_name,
                    tensor_parallel_size=self.retriever_gpus,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                )
            finally:
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

        return self._embedder

    @property
    def seq_reranker(self) -> Reranker:
        """Lazy load reranker with dedicated GPU assignment."""
        if self._reranker_model is None:
            gpu_ids = self._get_reranker_gpu_ids()
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                logger.info(
                    f"Loading reranker on GPUs: {gpu_ids} "
                    f"(tensor_parallel_size={self.reranker_gpus})"
                )

                self._reranker_model = Reranker(
                    model=self.reranker_model_name,
                    batch_size=self.rerank_batch_size,
                    use_multi_gpu=self.reranker_gpus > 1,
                    tensor_parallel_size=self.reranker_gpus,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                )
            finally:
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

        return self._reranker_model

    def _format_context(self, chunks: List[Chunk]) -> str:
        """Format chunks into context string for LLM."""
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.text if isinstance(chunk, Chunk) else chunk.get("text", "")
            if text and text.strip():
                context_parts.append(f"[{i}] {text}")

        return "\n\n".join(context_parts)

    def run(
        self,
        query: str,
        retrieval_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
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

        # Step 2: Pre-initialize models
        print(f"\n[Step 2] Loading models...")
        preinit_start = time.time()

        # Load embedder (uses retriever GPUs)
        _ = self.seq_embedder
        print(f"  Embedder loaded on GPUs: {self._get_retriever_gpu_ids()}")

        # Load reranker (uses reranker GPUs)
        _ = self.seq_reranker
        print(f"  Reranker loaded on GPUs: {self._get_reranker_gpu_ids()}")

        preinit_time = time.time() - preinit_start
        print(f"  Models loaded in {preinit_time:.2f}s")

        # Step 3: Process sub-queries sequentially
        print(f"\n[Step 3] Processing sub-queries sequentially...")
        step3_start = time.time()

        # Track chain of answers
        chain_of_answers = []
        all_chunks = []
        subquery_results = []
        current_subqueries = list(sub_queries)  # Copy to track rewrites

        for i, subquery in enumerate(sub_queries):
            subquery_idx = i + 1
            is_last = subquery_idx == n_subqueries

            print(f"\n  --- Sub-query {subquery_idx}/{n_subqueries} ---")

            # If this subquery has [answer] placeholder and we have previous answer, rewrite it
            if "[answer]" in subquery.lower() and chain_of_answers:
                prev_result = chain_of_answers[-1]
                prev_subquery = prev_result["subquery"]
                prev_context = prev_result["context"]

                print(
                    f"  Rewriting placeholder with answer from sub-query {subquery_idx - 1}..."
                )
                rewritten = self.decomposer.rewrite_subquery(
                    previous_subquery=prev_subquery,
                    current_subquery=subquery,
                    context=prev_context,
                )
                print(f"  Original: {subquery}")
                print(f"  Rewritten: {rewritten}")
                subquery = rewritten
                current_subqueries[i] = rewritten

            # Embed the sub-query
            print(f"  Embedding sub-query...")
            embed_start = time.time()
            query_embedding = self.seq_embedder.embed_query(subquery)
            embed_time = time.time() - embed_start
            print(f"  Embedded in {embed_time:.2f}s")

            # Retrieve chunks
            print(f"  Retrieving chunks...")
            retrieve_start = time.time()
            chunks = retrieve_chunks(
                query_embedding=query_embedding,
                collection_name=self.collection_name,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                top_k=self.retrieval_top_k,
            )
            retrieve_time = time.time() - retrieve_start
            print(f"  Retrieved {len(chunks)} chunks in {retrieve_time:.2f}s")

            # Rerank chunks
            if chunks:
                print(f"  Reranking chunks...")
                rerank_start = time.time()
                reranked = self.seq_reranker.rerank(
                    subquery, chunks, top_k=self.rerank_per_subquery
                )
                rerank_time = time.time() - rerank_start
                print(f"  Reranked to {len(reranked)} chunks in {rerank_time:.2f}s")
            else:
                reranked = []
                rerank_time = 0.0

            # Format context from reranked chunks
            context = self._format_context(reranked)

            # Store result
            result = {
                "subquery_idx": subquery_idx,
                "original_subquery": sub_queries[i],
                "subquery": subquery,
                "chunks": reranked,
                "context": context,
                "embed_time": embed_time,
                "retrieve_time": retrieve_time,
                "rerank_time": rerank_time,
            }
            subquery_results.append(result)
            chain_of_answers.append(result)

            # Add chunks to all_chunks (for final result)
            for chunk in reranked:
                if isinstance(chunk, Chunk):
                    chunk_copy = chunk.model_copy()
                    chunk_copy.source_subquery = subquery
                    all_chunks.append(chunk_copy)
                else:
                    chunk_copy = chunk.copy()
                    chunk_copy["source_subquery"] = subquery
                    all_chunks.append(Chunk(**chunk_copy))

        step3_time = time.time() - step3_start
        print(f"\n  Sequential processing completed in {step3_time:.2f}s")

        # Step 4: Generate final answer
        print(f"\n[Step 4] Generating final answer...")
        step4_start = time.time()

        # Build chain summary
        chain_summary_parts = []
        for result in chain_of_answers:
            chain_summary_parts.append(
                f"Sub-query {result['subquery_idx']}: {result['subquery']}"
            )

        chain_summary = "\n".join(chain_summary_parts)

        # Get final subquery and its context
        final_result = chain_of_answers[-1]
        final_subquery = final_result["subquery"]
        final_context = final_result["context"]

        # Use the generator for final answer
        final_chunks = [result["chunks"] for result in chain_of_answers]
        flat_final_chunks = []
        for chunk_list in final_chunks:
            flat_final_chunks.extend(chunk_list)

        # Generate using the base generator
        generation_result = self.generator.generate(query, flat_final_chunks)
        step4_time = time.time() - step4_start
        print(
            f"Answer generated ({generation_result.usage.total_tokens} tokens) in {step4_time:.2f}s"
        )

        # Deduplicate chunks for final result
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            chunk_id = chunk.chunk_id
            if chunk_id is None:
                import hashlib

                chunk_id = f"text_{hashlib.md5(chunk.text.encode()).hexdigest()[:8]}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)

        # Sort by reranker score
        unique_chunks.sort(key=lambda x: x.reranker_score or 0, reverse=True)

        # Update final ranks
        for rank, chunk in enumerate(unique_chunks, 1):
            chunk.final_rank = rank

        # Compile results
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

        result = PipelineResult(
            query=query,
            answer=generation_result.answer,
            pipeline="sequential_decomposition",
            config=config,
            usage=usage,
            sub_queries=current_subqueries,  # Rewritten sub-queries (placeholders filled)
            original_sub_queries=sub_queries,  # Original sub-queries with placeholders
            n_subqueries=n_subqueries,
            n_unique_chunks=len(unique_chunks),
            retrieved_per_subquery=[
                {"subquery": r["subquery"], "count": len(r["chunks"])}
                for r in subquery_results
            ],
            reranked_per_subquery=[
                {"subquery": r["subquery"], "count": len(r["chunks"])}
                for r in subquery_results
            ],
            final_chunks=unique_chunks,
        )

        # Calculate total runtime
        pipeline_end_time = time.time()
        total_runtime = pipeline_end_time - pipeline_start_time

        # Compile timing information
        timing_info = {
            "total_runtime": total_runtime,
            "preinit_models": preinit_time,
            "step1_decomposition": step1_time,
            "step3_sequential_processing": step3_time,
            "step4_generation": step4_time,
            "per_subquery": [
                {
                    "subquery_idx": r["subquery_idx"],
                    "embed_time": r["embed_time"],
                    "retrieve_time": r["retrieve_time"],
                    "rerank_time": r["rerank_time"],
                }
                for r in subquery_results
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
        print(f"  Step 3 (Sequential Processing): {step3_time:.2f}s")
        print(f"  Step 4 (Generation): {step4_time:.2f}s")

        # Add timing to result
        result.timing = timing_info

        return result
