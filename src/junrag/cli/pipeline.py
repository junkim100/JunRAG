#!/usr/bin/env python3
"""
CLI for running JunRAG pipelines.

Usage:
    junrag naive --query "Simple question" --metadata_path metadata.json
    junrag full --query "Complex multi-hop question" --metadata_path metadata.json
"""

# IMPORTANT: Set multiprocessing start method to 'spawn' BEFORE any other imports
# This is required for vLLM tensor parallelism to work correctly with CUDA
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set or in subprocess, ignore
    pass

import json
import logging
import sys
import traceback

import fire
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def naive(
    query: str,
    # Qdrant settings
    metadata_path: str = None,
    collection_name: str = None,
    qdrant_url: str = None,
    qdrant_api_key: str = None,
    # Model settings
    embedding_model: str = "jinaai/jina-embeddings-v3",
    reranker_model: str = "Qwen/Qwen3-Reranker-4B",
    llm_model: str = "gpt-5.1-2025-11-13",
    # Embedding settings
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    # Retrieval settings
    retrieval_top_k: int = 20,
    rerank_top_k: int = 5,
    rerank_batch_size: int = 64,
    use_multi_gpu: bool = True,
    # Generation settings
    temperature: float = 0.1,
    max_tokens: int = 16384,
    reasoning_effort: str = "medium",
    openai_api_key: str = None,
    # Output
    output_json: str = None,
    env_file: str = None,
    # Debug
    debug: bool = False,
):
    """
    Run the Naive Pipeline: Query → Retrieve → Rerank → Generate

    Args:
        query: User query (required)
        metadata_path: Path to Qdrant metadata JSON file
        collection_name: Qdrant collection name
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key
        embedding_model: Embedding model name
        reranker_model: Reranker model name
        llm_model: LLM model for generation
        tensor_parallel_size: GPUs for embedding/reranking
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Max sequence length
        retrieval_top_k: Chunks to retrieve
        rerank_top_k: Chunks after reranking
        rerank_batch_size: Reranking batch size
        use_multi_gpu: Use multi-GPU for reranking
        temperature: LLM temperature (ignored for reasoning models)
        max_tokens: Max response tokens
        reasoning_effort: Reasoning effort for reasoning models (low, medium, high)
        openai_api_key: OpenAI API key
        output_json: Save results to JSON
        env_file: Path to .env file
        debug: Enable debug logging
    """
    # Configure debug logging
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("junrag").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Load environment variables
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Validate required parameters
    if not query or not query.strip():
        logger.error("Query is required")
        print("ERROR: --query is required and cannot be empty")
        sys.exit(1)

    if not metadata_path and not collection_name:
        logger.error("Either metadata_path or collection_name is required")
        print("ERROR: Either --metadata_path or --collection_name is required")
        sys.exit(1)

    try:
        from junrag.pipelines import NaivePipeline

        logger.info(f"Initializing Naive Pipeline...")
        logger.info(f"  Embedding model: {embedding_model}")
        logger.info(f"  Reranker model: {reranker_model}")
        logger.info(f"  LLM model: {llm_model}")

        pipeline = NaivePipeline(
            metadata_path=metadata_path,
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            llm_model=llm_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
            rerank_batch_size=rerank_batch_size,
            use_multi_gpu=use_multi_gpu,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            openai_api_key=openai_api_key,
        )

        result = pipeline.run(query)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nERROR: {e}")
        print("\nCheck your configuration:")
        print("  - Ensure API keys are set (OPENAI_API_KEY, QDRANT_API_KEY)")
        print("  - Verify model names are correct")
        print("  - Check collection exists in Qdrant")
        sys.exit(1)

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        print(f"\nERROR: {e}")
        print("\nCheck your connections:")
        print("  - Is Qdrant server running?")
        print("  - Is the URL correct?")
        sys.exit(1)

    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        print(f"\nERROR: GPU out of memory")
        print("\nTry:")
        print(
            "  - Reduce --gpu_memory_utilization (current: {})".format(
                gpu_memory_utilization
            )
        )
        print("  - Reduce --max_model_len (current: {})".format(max_model_len))
        print(
            "  - Reduce --tensor_parallel_size (current: {})".format(
                tensor_parallel_size
            )
        )
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Pipeline failed: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nERROR: Pipeline failed: {e}")
        print("\nFor more details, check the logs or run with DEBUG logging.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("ANSWER")
    print(f"{'='*80}")

    # Convert Pydantic model to dict for display
    result_dict = result.model_dump(mode="json")

    # Display in order: config, usage, retrieved_chunks, pipeline, query, answer
    print("\n[CONFIG]")
    print(json.dumps(result_dict.get("config", {}), indent=2))

    print("\n[USAGE]")
    print(json.dumps(result_dict.get("usage", {}), indent=2))

    print("\n[RETRIEVED_CHUNKS]")
    chunks_key = "retrieved_chunks" if "retrieved_chunks" in result_dict else "final_chunks"
    chunks = result_dict.get(chunks_key, [])
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print("First chunk preview:")
        print(json.dumps(chunks[0] if isinstance(chunks[0], dict) else str(chunks[0]), indent=2))

    print(f"\n[PIPELINE]")
    print(result_dict.get("pipeline", "unknown"))

    print(f"\n[QUERY]")
    print(result_dict.get("query", ""))

    print(f"\n[ANSWER]")
    print(result_dict.get("answer", ""))

    if output_json:
        try:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_json}")
        except Exception as e:
            logger.warning(f"Failed to save output JSON: {e}")
            print(f"\nWarning: Failed to save results to {output_json}: {e}")

    return result_dict


def full(
    query: str,
    # Qdrant settings
    metadata_path: str = None,
    collection_name: str = None,
    qdrant_url: str = None,
    qdrant_api_key: str = None,
    # Model settings
    embedding_model: str = "jinaai/jina-embeddings-v3",
    reranker_model: str = "Qwen/Qwen3-Reranker-4B",
    llm_model: str = "gpt-5.1-2025-11-13",
    decomposition_model: str = "gpt-4o",
    # Embedding settings (tensor_parallel_size forced to 1 for full pipeline)
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    # Retrieval settings
    retrieval_top_k: int = 50,
    chunks_per_subquery: int = 10,
    rerank_batch_size: int = 64,
    # Full pipeline settings
    max_cap: int = 25,
    min_floor: int = 5,
    max_concurrent_retrievals: int = 8,
    # Generation settings
    temperature: float = 0.1,
    max_tokens: int = 16384,
    reasoning_effort: str = "medium",
    openai_api_key: str = None,
    # Output
    output_json: str = None,
    env_file: str = None,
    # Debug
    debug: bool = False,
):
    """
    Run the Full Pipeline: Query → Decompose → Batch Embed → Parallel Retrieve → Sequential Rerank → Dynamic Top-K → Generate

    Design:
    - Embedding: Batch process all subqueries (tensor_parallel_size=1 for stability)
    - Retrieval: Parallel Qdrant queries (no GPU, semaphore-controlled)
    - Reranking: Sequential to avoid GPU conflicts (single GPU reranker)

    Dynamic K formula: K_final = min(MAX_CAP, max(MIN_FLOOR, N_sub × chunks_per_subquery))

    Args:
        query: User query (can be complex multi-hop)
        metadata_path: Path to Qdrant metadata JSON file
        collection_name: Qdrant collection name
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key
        embedding_model: Embedding model name
        reranker_model: Reranker model name
        llm_model: LLM model for generation
        decomposition_model: Model for query decomposition
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Max sequence length
        retrieval_top_k: Chunks to retrieve per sub-query
        chunks_per_subquery: Chunks per sub-query after reranking (M)
        rerank_batch_size: Reranking batch size
        max_cap: Maximum chunks for final selection (MAX_CAP)
        min_floor: Minimum chunks for final selection (MIN_FLOOR)
        max_concurrent_retrievals: Max parallel Qdrant retrievals
        temperature: LLM temperature (ignored for reasoning models)
        max_tokens: Max response tokens
        reasoning_effort: Reasoning effort for reasoning models (low, medium, high)
        openai_api_key: OpenAI API key
        output_json: Save results to JSON
        env_file: Path to .env file
        debug: Enable debug logging

    Note: tensor_parallel_size is forced to 1 for stability. Reranking is sequential.
    """
    # Configure debug logging
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("junrag").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Load environment variables
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Validate required parameters
    if not query or not query.strip():
        logger.error("Query is required")
        print("ERROR: --query is required and cannot be empty")
        sys.exit(1)

    if not metadata_path and not collection_name:
        logger.error("Either metadata_path or collection_name is required")
        print("ERROR: Either --metadata_path or --collection_name is required")
        sys.exit(1)

    try:
        from junrag.pipelines import FullPipeline

        logger.info(f"Initializing Full Pipeline...")
        logger.info(f"  Embedding model: {embedding_model}")
        logger.info(f"  Reranker model: {reranker_model}")
        logger.info(f"  LLM model: {llm_model}")
        logger.info(f"  Decomposition model: {decomposition_model}")

        pipeline = FullPipeline(
            metadata_path=metadata_path,
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            llm_model=llm_model,
            decomposition_model=decomposition_model,
            # tensor_parallel_size is forced to 1 in FullPipeline
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=chunks_per_subquery,
            rerank_batch_size=rerank_batch_size,
            max_cap=max_cap,
            min_floor=min_floor,
            chunks_per_subquery=chunks_per_subquery,
            max_concurrent_retrievals=max_concurrent_retrievals,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            openai_api_key=openai_api_key,
        )

        result = pipeline.run(query)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nERROR: {e}")
        print("\nCheck your configuration:")
        print("  - Ensure API keys are set (OPENAI_API_KEY, QDRANT_API_KEY)")
        print("  - Verify model names are correct")
        print("  - Check collection exists in Qdrant")
        sys.exit(1)

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        print(f"\nERROR: {e}")
        print("\nCheck your connections:")
        print("  - Is Qdrant server running?")
        print("  - Is the URL correct?")
        sys.exit(1)

    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        print(f"\nERROR: GPU out of memory")
        print("\nTry:")
        print(
            "  - Reduce --gpu_memory_utilization (current: {})".format(
                gpu_memory_utilization
            )
        )
        print("  - Reduce --max_model_len (current: {})".format(max_model_len))
        print(
            "  - Reduce --tensor_parallel_size (current: {})".format(
                tensor_parallel_size
            )
        )
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Pipeline failed: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nERROR: Pipeline failed: {e}")
        print("\nFor more details, check the logs or run with DEBUG logging.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("ANSWER")
    print(f"{'='*80}")

    # Convert Pydantic model to dict for display
    result_dict = result.model_dump(mode="json")

    # Display in order: config, usage, retrieved_chunks, pipeline, query, answer
    print("\n[CONFIG]")
    print(json.dumps(result_dict.get("config", {}), indent=2))

    print("\n[USAGE]")
    print(json.dumps(result_dict.get("usage", {}), indent=2))

    print("\n[RETRIEVED_CHUNKS]")
    chunks_key = "retrieved_chunks" if "retrieved_chunks" in result_dict else "final_chunks"
    chunks = result_dict.get(chunks_key, [])
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print("First chunk preview:")
        print(json.dumps(chunks[0] if isinstance(chunks[0], dict) else str(chunks[0]), indent=2))

    print(f"\n[PIPELINE]")
    print(result_dict.get("pipeline", "unknown"))

    print(f"\n[QUERY]")
    print(result_dict.get("query", ""))

    print(f"\n[ANSWER]")
    print(result_dict.get("answer", ""))

    if output_json:
        try:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_json}")
        except Exception as e:
            logger.warning(f"Failed to save output JSON: {e}")
            print(f"\nWarning: Failed to save results to {output_json}: {e}")

    return result_dict


def main():
    """Main entry point."""
    fire.Fire({"naive": naive, "full": full})


if __name__ == "__main__":
    main()
