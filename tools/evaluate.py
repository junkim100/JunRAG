"""
This script is modified from the original script:
https://github.com/codelion/optillm/blob/main/scripts/eval_frames_benchmark.py
"""

import json
import os
import inspect
from typing import Any, Dict, List, Optional, Type

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import argparse

load_dotenv()

HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")
UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY")
UPSTAGE_BASE_URL = os.environ.get("UPSTAGE_BASE_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # get from dotenv


eval_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url=UPSTAGE_BASE_URL)


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re

    # Insert underscore before uppercase letters (except the first one)
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters that follow lowercase or digits
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def discover_pipelines() -> Dict[str, Type]:
    """
    Dynamically discover available pipeline classes from junrag.pipelines.

    Returns:
        Dictionary mapping pipeline name (lowercase, without 'Pipeline' suffix) to class
    """
    try:
        import junrag.pipelines as pipelines_module

        pipeline_classes = {}
        for name in dir(pipelines_module):
            obj = getattr(pipelines_module, name)
            # Check if it's a class, ends with 'Pipeline', and is not the base class
            if (
                inspect.isclass(obj)
                and name.endswith("Pipeline")
                and name != "BasePipeline"
                and hasattr(obj, "run")  # Ensure it has a run method
            ):
                # Convert "NaivePipeline" -> "naive", "ParallelPipeline" -> "parallel",
                # "SequentialDecompositionPipeline" -> "sequential_decomposition"
                base_name = name[:-8]  # Remove "Pipeline" suffix
                pipeline_name = camel_to_snake(base_name)
                pipeline_classes[pipeline_name] = obj

        return pipeline_classes
    except Exception as e:
        # Fallback to known pipelines if discovery fails
        print(f"Warning: Could not discover pipelines dynamically: {e}")
        from junrag.pipelines import (
            NaivePipeline,
            ParallelPipeline,
            SequentialDecompositionPipeline,
        )

        return {
            "naive": NaivePipeline,
            "parallel": ParallelPipeline,
            "sequential_decomposition": SequentialDecompositionPipeline,
        }


def get_pipeline_class(pipeline_type: str) -> Type:
    """
    Get pipeline class by name.

    Args:
        pipeline_type: Pipeline name (e.g., 'naive', 'parallel')

    Returns:
        Pipeline class
    """
    pipelines = discover_pipelines()
    if pipeline_type not in pipelines:
        available = ", ".join(sorted(pipelines.keys()))
        raise ValueError(
            f"Unknown pipeline type '{pipeline_type}'. "
            f"Available pipelines: {available}"
        )
    return pipelines[pipeline_type]


def instantiate_pipeline(
    pipeline_class: Type,
    pipeline_type: str,
    metadata_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    embedding_model: str = "jinaai/jina-embeddings-v3",
    reranker_model: str = "Qwen/Qwen3-Reranker-4B",
    llm_model: str = "gpt-5.1-2025-11-13",
    decomposition_model: str = "gpt-4o",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    retrieval_top_k: int = 20,
    rerank_top_k: int = 5,
    chunks_per_subquery: int = 10,
    rerank_batch_size: int = 64,
    use_multi_gpu: bool = True,
    max_cap: int = 25,
    min_floor: int = 5,
    max_concurrent_retrievals: int = 8,
    temperature: float = 0.1,
    max_tokens: int = 16384,
    reasoning_effort: str = "medium",
    openai_api_key: Optional[str] = None,
):
    """
    Dynamically instantiate a pipeline, only passing parameters it accepts.

    Args:
        pipeline_class: Pipeline class to instantiate
        pipeline_type: Pipeline type name (for special handling if needed)
        **kwargs: All possible pipeline parameters

    Returns:
        Instantiated pipeline
    """
    # Get the __init__ signature to see what parameters the pipeline accepts
    sig = inspect.signature(pipeline_class.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    # Check if the pipeline accepts **kwargs (for parameters passed to BasePipeline)
    # This is important because SequentialDecompositionPipeline uses **kwargs to pass to BasePipeline
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )

    # BasePipeline parameters that should be included if pipeline accepts **kwargs
    # These are parameters that BasePipeline.__init__ accepts
    base_pipeline_params = {
        "metadata_path",
        "collection_name",
        "qdrant_url",
        "qdrant_api_key",
        "embedding_model",
        "reranker_model",
        "llm_model",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "retrieval_top_k",
        "rerank_top_k",
        "rerank_batch_size",
        "use_multi_gpu",
        "temperature",
        "max_tokens",
        "reasoning_effort",
        "openai_api_key",
        "env_file",
    }

    # Build kwargs dict with only valid parameters
    kwargs = {}
    all_params = {
        "metadata_path": metadata_path,
        "collection_name": collection_name,
        "qdrant_url": qdrant_url,
        "qdrant_api_key": qdrant_api_key,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "llm_model": llm_model,
        "decomposition_model": decomposition_model,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "retrieval_top_k": retrieval_top_k,
        "rerank_top_k": rerank_top_k,
        "chunks_per_subquery": chunks_per_subquery,
        "rerank_batch_size": rerank_batch_size,
        "use_multi_gpu": use_multi_gpu,
        "max_cap": max_cap,
        "min_floor": min_floor,
        "max_concurrent_retrievals": max_concurrent_retrievals,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "openai_api_key": openai_api_key,
    }

    # Special handling for pipeline-specific parameters
    # ParallelPipeline uses chunks_per_subquery for rerank_top_k
    if pipeline_type == "parallel":
        all_params["rerank_top_k"] = chunks_per_subquery
    # SequentialDecompositionPipeline uses rerank_per_subquery instead of chunks_per_subquery
    elif pipeline_type == "sequential_decomposition":
        all_params["rerank_per_subquery"] = chunks_per_subquery

    # Only include parameters that the pipeline accepts
    # For collection_name, qdrant_url, qdrant_api_key: if metadata_path is provided,
    # always pass them (even if None) so pipeline can read from metadata
    # Otherwise, skip None values to use defaults
    has_metadata_path = bool(all_params.get("metadata_path"))

    for param_name, param_value in all_params.items():
        # Include if it's an explicit parameter OR if pipeline accepts **kwargs and it's a BasePipeline parameter
        is_valid = param_name in valid_params
        is_base_param = accepts_kwargs and param_name in base_pipeline_params

        if is_valid or is_base_param:
            # Skip None values for collection_name, qdrant_url, qdrant_api_key
            # UNLESS metadata_path is provided (in which case pipeline should read from it)
            if (
                param_name in ("collection_name", "qdrant_url", "qdrant_api_key")
                and param_value is None
                and not has_metadata_path
            ):
                continue
            kwargs[param_name] = param_value

    return pipeline_class(**kwargs)


def load_existing_results(filename: str) -> Dict[str, Any]:
    """Load results file, which may contain metadata and results array."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            # Handle both old format (list) and new format (dict with metadata)
            if isinstance(data, list):
                return {"metadata": None, "results": data}
            return data
    except FileNotFoundError:
        return {"metadata": None, "results": []}


def save_result(filename: str, result: Dict, metadata: Optional[Dict] = None):
    """Save result to file, with optional metadata."""
    data = load_existing_results(filename)

    # Update metadata if provided
    if metadata is not None:
        data["metadata"] = metadata

    # Add result to results array
    data["results"].append(result)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def get_last_processed_index(data: Dict) -> int:
    """Get the last processed index from results data."""
    results = data.get("results", [])
    if not results:
        return -1
    return max(int(r.get("index", -1)) for r in results)


def generate_llm_prompt(prompt: str, wiki_links: List[str]) -> str:
    return f"Here are the relevant Wikipedia articles:\n{wiki_links}\n\nBased on all the information, answer the query. \n\nQuery: {prompt}\n\n"


def get_pipeline_response(query: str, pipeline) -> tuple:
    """Get response from a pipeline.

    Returns:
        Tuple of (answer, n_chains, used_internal_knowledge) where:
        - n_chains is None for non-sequential_decomposition pipelines
        - used_internal_knowledge is None for non-sequential_decomposition pipelines
    """
    # For evaluation, skip model cleanup so models persist across items
    # This significantly speeds up evaluation by loading models only once
    # Check if pipeline.run supports cleanup_models parameter
    sig = inspect.signature(pipeline.run)
    if "cleanup_models" in sig.parameters:
        result = pipeline.run(query, cleanup_models=False)
    else:
        # Pipeline doesn't support cleanup_models parameter (e.g., naive, parallel)
        result = pipeline.run(query)
    n_chains = getattr(result, "n_chains", None)
    used_internal_knowledge = getattr(result, "used_internal_knowledge", None)
    return result.answer, n_chains, used_internal_knowledge


def evaluate_response(
    question: str, llm_response: str, ground_truth: str
) -> Dict[str, str]:
    evaluation_prompt = f"""===Task===
I need your help in evaluating an answer provided by an LLM against a ground
truth answer. Your task is to determine if the ground truth answer is present in the LLM's
response. Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers.
Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the
"Ground Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Question: {question}
- Predicted Answer: {llm_response}
- Ground Truth Answer: {ground_truth}
===Output Format===
Provide your final evaluation in the following format:
"Explanation:" (How you made the decision?)
"Decision:" ("TRUE" or "FALSE" )
Please proceed with the evaluation."""

    evaluation_response = eval_client.chat.completions.create(
        model="solar-pro2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": evaluation_prompt},
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.3,
    )

    evaluation_text = evaluation_response.choices[0].message.content.strip()

    # Extract the decision and explanation
    lines = evaluation_text.split("\n")
    decision = "FALSE"
    explanation = ""
    for line in lines:
        if line.startswith("Decision:"):
            decision = line.split(":")[1].strip().upper()
        elif line.startswith("Explanation:"):
            explanation = line.split(":", 1)[1].strip()

    return {"decision": decision, "explanation": explanation}


def get_qdrant_info_from_metadata(metadata_path):
    """Read qdrant_url and collection_name from the metadata JSON file."""
    qdrant_url = None
    collection_name = None
    if metadata_path:
        try:
            if not os.path.exists(metadata_path):
                print(f"Warning: Metadata file does not exist: {metadata_path}")
                return qdrant_url, collection_name
            with open(metadata_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
                qdrant_url = meta.get("qdrant_url", None)
                collection_name = meta.get("collection_name", None)
                if collection_name:
                    print(f"Read collection_name from metadata: {collection_name}")
                else:
                    print(
                        f"Warning: collection_name not found in metadata file: {metadata_path}"
                    )
                    print(f"Metadata keys: {list(meta.keys())}")
        except Exception as e:
            print(f"Warning: Could not read metadata JSON: {e}")
            import traceback

            traceback.print_exc()
    return qdrant_url, collection_name


def main(
    pipeline_type: str,
    metadata_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    embedding_model: str = "jinaai/jina-embeddings-v3",
    reranker_model: str = "Qwen/Qwen3-Reranker-4B",
    llm_model: str = "gpt-5.1-2025-11-13",
    decomposition_model: str = "gpt-4o",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    retrieval_top_k: int = 20,
    rerank_top_k: int = 5,
    chunks_per_subquery: int = 10,
    rerank_batch_size: int = 64,
    use_multi_gpu: bool = True,
    max_cap: int = 25,
    min_floor: int = 5,
    max_concurrent_retrievals: int = 8,
    temperature: float = 0.1,
    max_tokens: int = 16384,
    reasoning_effort: str = "medium",
    openai_api_key: Optional[str] = None,
):
    """Main evaluation function using pipelines."""
    # qdrant_api_key always comes from dotenv
    global QDRANT_API_KEY

    # Read qdrant_url and collection_name from metadata_path if provided
    file_qdrant_url, file_collection_name = get_qdrant_info_from_metadata(metadata_path)

    # Always prefer values from metadata file if available
    # If metadata_path is provided but we couldn't read from file, BasePipeline will read from metadata_path
    if metadata_path:
        # When metadata_path is provided, use file values if successfully read
        # If file values are None (reading failed), BasePipeline will read from metadata_path
        if file_collection_name:
            # Successfully read from metadata file - use it directly
            real_collection_name = file_collection_name
            print(f"✓ Using collection_name from metadata file: {real_collection_name}")
        else:
            # Couldn't read from file - pass None so BasePipeline reads from metadata_path
            real_collection_name = None
            print(
                "⚠ Note: collection_name not found when reading metadata file, BasePipeline will read from metadata_path"
            )

        if file_qdrant_url:
            real_qdrant_url = file_qdrant_url
            print(f"✓ Using qdrant_url from metadata file: {real_qdrant_url}")
        else:
            real_qdrant_url = None
    else:
        # When no metadata_path, use explicit parameters
        real_collection_name = collection_name
        real_qdrant_url = qdrant_url
        if real_collection_name:
            print(f"Using collection_name from parameter: {real_collection_name}")

    # Now func args/CLI parameters for qdrant_api_key are ignored; always use dotenv
    real_qdrant_api_key = QDRANT_API_KEY

    if not metadata_path and not real_collection_name:
        raise ValueError("Either metadata_path or collection_name is required")

    pipeline_class = get_pipeline_class(pipeline_type)

    print(f"Initializing {pipeline_type} pipeline...")
    pipeline = instantiate_pipeline(
        pipeline_class=pipeline_class,
        pipeline_type=pipeline_type,
        metadata_path=metadata_path,
        collection_name=real_collection_name,
        qdrant_url=real_qdrant_url,
        qdrant_api_key=real_qdrant_api_key,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        llm_model=llm_model,
        decomposition_model=decomposition_model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        chunks_per_subquery=chunks_per_subquery,
        rerank_batch_size=rerank_batch_size,
        use_multi_gpu=use_multi_gpu,
        max_cap=max_cap,
        min_floor=min_floor,
        max_concurrent_retrievals=max_concurrent_retrievals,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        openai_api_key=openai_api_key,
    )

    # Verify collection_name is set after pipeline initialization
    if not pipeline.collection_name:
        raise ValueError(
            f"collection_name is required but was not set. "
            f"metadata_path={metadata_path}, "
            f"real_collection_name={real_collection_name}, "
            f"pipeline.collection_name={pipeline.collection_name}"
        )

    print(f"Pipeline initialized successfully")
    print(f"Using collection_name: {pipeline.collection_name}")

    # Load the dataset
    dataset = load_dataset(
        "google/frames-benchmark", split="test", token=HF_ACCESS_TOKEN
    )

    # Take first 100 items for this assignment
    dataset = dataset.take(100)

    # Ensure the results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Build filename with key parameters
    safe_llm_model = llm_model.replace("/", "_")
    if pipeline_type == "naive":
        filename = os.path.join(
            results_dir,
            f"evaluation_results_{pipeline_type}_{safe_llm_model}_ret{retrieval_top_k}_rerank{rerank_top_k}_batch{rerank_batch_size}.json",
        )
    else:
        filename = os.path.join(
            results_dir,
            f"evaluation_results_{pipeline_type}_{safe_llm_model}_ret{retrieval_top_k}_chunks{chunks_per_subquery}_batch{rerank_batch_size}.json",
        )

    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)

    # Prepare metadata with all configuration parameters
    metadata = {
        "pipeline_type": pipeline_type,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "llm_model": llm_model,
        "decomposition_model": decomposition_model,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "retrieval_top_k": retrieval_top_k,
        "rerank_top_k": rerank_top_k if pipeline_type == "naive" else None,
        "chunks_per_subquery": (
            chunks_per_subquery if pipeline_type != "naive" else None
        ),
        "rerank_batch_size": rerank_batch_size,
        "use_multi_gpu": use_multi_gpu,
        "max_cap": max_cap if pipeline_type == "parallel" else None,
        "min_floor": min_floor if pipeline_type == "parallel" else None,
        "max_concurrent_retrievals": (
            max_concurrent_retrievals if pipeline_type == "parallel" else None
        ),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "metadata_path": metadata_path,
        "collection_name": (
            pipeline.collection_name if hasattr(pipeline, "collection_name") else None
        ),
    }

    # Save metadata if this is a new file or metadata has changed
    if existing_results.get("metadata") != metadata:
        # Update metadata in file
        data = load_existing_results(filename)
        data["metadata"] = metadata
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    try:
        for item in tqdm(dataset, desc="Processing samples"):
            index = int(item["Unnamed: 0"])
            if index <= last_processed_index:
                continue

            # Use pipeline to get response (pass query directly, not the generated prompt)
            llm_response, n_chains, used_internal_knowledge = get_pipeline_response(
                item["Prompt"], pipeline
            )
            evaluation = evaluate_response(item["Prompt"], llm_response, item["Answer"])

            result = {
                "index": index,
                "prompt": item["Prompt"],
                "ground_truth": item["Answer"],
                "llm_response": llm_response,
                "evaluation_decision": evaluation["decision"],
                "evaluation_explanation": evaluation["explanation"],
                "reasoning_type": item["reasoning_types"],
            }

            # Add n_chains and used_internal_knowledge for sequential_decomposition pipeline
            if pipeline_type == "sequential_decomposition":
                if n_chains is not None:
                    result["n_chains"] = n_chains
                if used_internal_knowledge is not None:
                    result["used_internal_knowledge"] = used_internal_knowledge

            save_result(filename, result, metadata=metadata)

        # Calculate and print summary statistics
        data = load_existing_results(filename)
        results = data.get("results", [])
        total_samples = len(results)
        correct_answers = sum(1 for r in results if r["evaluation_decision"] == "TRUE")
        accuracy = correct_answers / total_samples

        print(f"Pipeline: {pipeline_type}")
        print(f"LLM Model: {llm_model}")
        print(f"Total samples: {total_samples}")
        print(f"Correct answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2%}")

        # Display number of chains and internal knowledge usage for sequential_decomposition pipeline
        if pipeline_type == "sequential_decomposition":
            chains_data = [r.get("n_chains") for r in results if "n_chains" in r]
            if chains_data:
                avg_chains = sum(chains_data) / len(chains_data)
                min_chains = min(chains_data)
                max_chains = max(chains_data)
                print(
                    f"Number of chains - Average: {avg_chains:.2f}, Min: {min_chains}, Max: {max_chains}"
                )

            # Display internal knowledge usage statistics
            internal_knowledge_data = [
                r.get("used_internal_knowledge")
                for r in results
                if "used_internal_knowledge" in r
            ]
            if internal_knowledge_data:
                total_with_internal_knowledge = sum(
                    1 for x in internal_knowledge_data if x is True
                )
                pct_with_internal_knowledge = (
                    total_with_internal_knowledge / len(internal_knowledge_data) * 100
                )
                print(
                    f"Internal knowledge usage: {total_with_internal_knowledge}/{len(internal_knowledge_data)} "
                    f"({pct_with_internal_knowledge:.1f}%) samples used internal knowledge at least once"
                )

        # Print accuracy by reasoning type
        reasoning_types = set(r["reasoning_type"] for r in results)
        for rt in reasoning_types:
            rt_samples = [r for r in results if r["reasoning_type"] == rt]
            rt_correct = sum(
                1 for r in rt_samples if r["evaluation_decision"] == "TRUE"
            )
            rt_accuracy = rt_correct / len(rt_samples)
            print(f"Accuracy for {rt}: {rt_accuracy:.2%}")
    finally:
        # Clean up pipeline models to avoid vLLM worker warnings on exit
        print("\nCleaning up pipeline models...")
        try:
            # Clean up embedder if it exists (BasePipeline)
            if (
                hasattr(pipeline, "_embedding_model")
                and pipeline._embedding_model is not None
            ):
                if hasattr(pipeline._embedding_model, "llm"):
                    try:
                        del pipeline._embedding_model.llm
                    except:
                        pass
                pipeline._embedding_model = None

            # Clean up reranker if it exists (BasePipeline)
            if hasattr(pipeline, "_reranker") and pipeline._reranker is not None:
                if hasattr(pipeline._reranker, "model"):
                    try:
                        del pipeline._reranker.model
                    except:
                        pass
                pipeline._reranker = None

            # For sequential_decomposition, clean up seq_embedder and seq_reranker
            if hasattr(pipeline, "_embedders") and getattr(pipeline, "_embedders"):
                for embedder in pipeline._embedders.values():
                    if hasattr(embedder, "llm"):
                        try:
                            del embedder.llm
                        except:
                            pass
                try:
                    pipeline._embedders.clear()
                except:
                    pipeline._embedders = {}

            if hasattr(pipeline, "_embedder") and pipeline._embedder is not None:
                if hasattr(pipeline._embedder, "llm"):
                    try:
                        del pipeline._embedder.llm
                    except:
                        pass
                pipeline._embedder = None

            if (
                hasattr(pipeline, "_reranker_model")
                and pipeline._reranker_model is not None
            ):
                if hasattr(pipeline._reranker_model, "model"):
                    try:
                        del pipeline._reranker_model.model
                    except:
                        pass
                pipeline._reranker_model = None

            # For parallel pipeline, clean up per-subquery models
            if hasattr(pipeline, "_subquery_embedders"):
                for embedder in pipeline._subquery_embedders.values():
                    if hasattr(embedder, "llm"):
                        try:
                            del embedder.llm
                        except:
                            pass
                pipeline._subquery_embedders.clear()

            if hasattr(pipeline, "_subquery_rerankers"):
                for reranker in pipeline._subquery_rerankers.values():
                    if hasattr(reranker, "model"):
                        try:
                            del reranker.model
                        except:
                            pass
                pipeline._subquery_rerankers.clear()

            print("✓ Models cleaned up successfully")

            # Small delay to allow vLLM workers to shut down cleanly
            import time

            time.sleep(1)
        except Exception as e:
            print(f"Warning: Error during model cleanup: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline performance on google/frames-benchmark"
    )
    # Dynamically discover available pipelines for choices
    try:
        available_pipelines = list(discover_pipelines().keys())
        pipeline_choices = sorted(available_pipelines)
        pipeline_help = f"Pipeline type. Available: {', '.join(pipeline_choices)}"
    except Exception:
        # Fallback if discovery fails
        pipeline_choices = ["naive", "parallel", "sequential_decomposition"]
        pipeline_help = (
            "Pipeline type: 'naive', 'parallel', or 'sequential_decomposition'"
        )

    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=pipeline_choices,
        help=pipeline_help,
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Path to Qdrant metadata JSON file",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=None,
        help="Qdrant collection name (overridden by metadata file if present)",
    )
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default=None,
        help="Qdrant server URL (overridden by metadata file if present)",
    )
    # We keep qdrant_api_key arg for compatibility, but always use QDRANT_API_KEY from dotenv
    parser.add_argument(
        "--qdrant_api_key",
        type=str,
        default=None,
        help="Qdrant API key (should be set via .env as QDRANT_API_KEY)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="jinaai/jina-embeddings-v3",
        help="Embedding model name",
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="Qwen/Qwen3-Reranker-4B",
        help="Reranker model name",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-5.1-2025-11-13",
        help="LLM model for generation",
    )
    parser.add_argument(
        "--decomposition_model",
        type=str,
        default="gpt-4o",
        help="Decomposition model (for parallel/sequential_decomposition pipeline)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for embedding/reranking",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Max sequence length",
    )
    parser.add_argument(
        "--retrieval_top_k",
        type=int,
        default=20,
        help="Chunks to retrieve",
    )
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=10,
        help="Chunks after reranking (for naive pipeline)",
    )
    parser.add_argument(
        "--chunks_per_subquery",
        type=int,
        default=10,
        help="Chunks per subquery after reranking (for parallel/sequential_decomposition pipeline)",
    )
    parser.add_argument(
        "--rerank_batch_size",
        type=int,
        default=64,
        help="Reranking batch size",
    )
    parser.add_argument(
        "--use_multi_gpu",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        nargs="?",
        const=True,
        default=True,
        help="Use multi-GPU for reranking (default: True, use --use_multi_gpu false to disable)",
    )
    parser.add_argument(
        "--max_cap",
        type=int,
        default=25,
        help="Maximum chunks for final selection (for parallel pipeline)",
    )
    parser.add_argument(
        "--min_floor",
        type=int,
        default=5,
        help="Minimum chunks for final selection (for parallel pipeline)",
    )
    parser.add_argument(
        "--max_concurrent_retrievals",
        type=int,
        default=8,
        help="Max parallel Qdrant retrievals (for parallel pipeline)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (ignored for reasoning models)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Max response tokens",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort for reasoning models",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key",
    )
    args = parser.parse_args()

    main(
        pipeline_type=args.pipeline,
        metadata_path=args.metadata_path,
        collection_name=args.collection_name,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=QDRANT_API_KEY,  # always use from dotenv
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        llm_model=args.llm_model,
        decomposition_model=args.decomposition_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        retrieval_top_k=args.retrieval_top_k,
        rerank_top_k=args.rerank_top_k,
        chunks_per_subquery=args.chunks_per_subquery,
        rerank_batch_size=args.rerank_batch_size,
        use_multi_gpu=args.use_multi_gpu,
        max_cap=args.max_cap,
        min_floor=args.min_floor,
        max_concurrent_retrievals=args.max_concurrent_retrievals,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        openai_api_key=args.openai_api_key,
    )
