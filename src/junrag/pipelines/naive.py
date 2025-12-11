"""
Naive Pipeline for JunRAG.
Simple linear flow: Query → Retrieve → Rerank → Generate
"""

from typing import Optional

from junrag.pipelines.base import BasePipeline
from junrag.components.retrieval import retrieve_chunks
from junrag.models import (
    Chunk,
    PipelineConfig,
    PipelineResult,
    UsageInfo,
)


class NaivePipeline(BasePipeline):
    """
    Naive RAG pipeline with simple linear flow.

    Flow: Query → Embed → Retrieve → Rerank → Generate
    """

    def run(
        self,
        query: str,
        retrieval_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run the naive pipeline.

        Args:
            query: User query
            retrieval_top_k: Override retrieval top_k
            rerank_top_k: Override rerank top_k

        Returns:
            PipelineResult with answer and pipeline results
        """
        retrieval_top_k = retrieval_top_k or self.retrieval_top_k
        rerank_top_k = rerank_top_k or self.rerank_top_k

        print(f"\n{'='*80}")
        print("NAIVE PIPELINE")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Collection: {self.collection_name}")

        # Step 1: Embed query
        print(f"\n[Step 1] Embedding query...")
        query_embedding = self.embedder.embed_query(query)
        print(f"Embedding shape: {query_embedding.shape}")

        # Step 2: Retrieve
        print(f"\n[Step 2] Retrieving top {retrieval_top_k} chunks...")
        retrieved_chunks = retrieve_chunks(
            query_embedding=query_embedding,
            collection_name=self.collection_name,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            top_k=retrieval_top_k,
        )
        print(f"Retrieved {len(retrieved_chunks)} chunks")

        # Check for missing text
        missing_text = sum(
            1
            for c in retrieved_chunks
            if not (
                c.text if isinstance(c, Chunk) else c.get("text", "")
            )
        )
        if missing_text > 0:
            print(
                f"Warning: {missing_text}/{len(retrieved_chunks)} chunks have no text."
            )

        # Step 3: Rerank
        print(f"\n[Step 3] Reranking to top {rerank_top_k}...")
        reranked_chunks = self.reranker.rerank(
            query, retrieved_chunks, top_k=rerank_top_k
        )
        print(f"Reranked to {len(reranked_chunks)} chunks")

        # Step 4: Generate
        print(f"\n[Step 4] Generating answer...")
        generation_result = self.generator.generate(query, reranked_chunks)
        print(f"Answer generated ({generation_result.usage.total_tokens} tokens)")

        # Convert chunks to Chunk models if needed
        retrieved_chunks_models = [
            chunk if isinstance(chunk, Chunk) else Chunk(**chunk)
            for chunk in retrieved_chunks
        ]
        reranked_chunks_models = [
            chunk if isinstance(chunk, Chunk) else Chunk(**chunk)
            for chunk in reranked_chunks
        ]

        # Compile results
        config = PipelineConfig(
            embedding_model=self.embedding_model_name,
            reranker_model=self.reranker_model_name,
            llm_model=self.llm_model_name,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
        )

        usage = UsageInfo(**generation_result.usage.model_dump())

        result = PipelineResult(
            query=query,
            answer=generation_result.answer,
            pipeline="naive",
            config=config,
            usage=usage,
            retrieved_chunks=retrieved_chunks_models,
            reranked_chunks=reranked_chunks_models,
        )

        print(f"\n{'='*80}")
        print("NAIVE PIPELINE COMPLETE")
        print(f"{'='*80}")

        return result
