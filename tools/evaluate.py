"""
This script is modified from the original script:
https://github.com/codelion/optillm/blob/main/scripts/eval_frames_benchmark.py
"""

import json
import os
import inspect
from typing import Dict, List, Optional, Type

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import argparse

load_dotenv()

HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")
UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY")
UPSTAGE_BASE_URL = os.environ.get("UPSTAGE_BASE_URL")

eval_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url=UPSTAGE_BASE_URL)


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
                # Convert "NaivePipeline" -> "naive", "ParallelPipeline" -> "parallel"
                pipeline_name = name[:-8].lower()  # Remove "Pipeline" suffix
                pipeline_classes[pipeline_name] = obj

        return pipeline_classes
    except Exception as e:
        # Fallback to known pipelines if discovery fails
        print(f"Warning: Could not discover pipelines dynamically: {e}")
        from junrag.pipelines import NaivePipeline, ParallelPipeline

        return {
            "naive": NaivePipeline,
            "parallel": ParallelPipeline,
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

    # Only include parameters that the pipeline accepts
    for param_name, param_value in all_params.items():
        if param_name in valid_params:
            kwargs[param_name] = param_value

    return pipeline_class(**kwargs)


def load_existing_results(filename: str) -> List[Dict]:
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_result(filename: str, result: Dict):
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)


def get_last_processed_index(results: List[Dict]) -> int:
    if not results:
        return -1
    return max(int(r.get("index", -1)) for r in results)


def generate_llm_prompt(prompt: str, wiki_links: List[str]) -> str:
    return f"Here are the relevant Wikipedia articles:\n{wiki_links}\n\nBased on all the information, answer the query. \n\nQuery: {prompt}\n\n"


def get_pipeline_response(query: str, pipeline) -> str:
    """Get response from a pipeline."""
    result = pipeline.run(query)
    return result.answer


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
    # Validate required parameters
    if not metadata_path and not collection_name:
        raise ValueError("Either metadata_path or collection_name is required")

    # Get pipeline class dynamically
    pipeline_class = get_pipeline_class(pipeline_type)

    # Initialize pipeline
    print(f"Initializing {pipeline_type} pipeline...")
    pipeline = instantiate_pipeline(
        pipeline_class=pipeline_class,
        pipeline_type=pipeline_type,
        metadata_path=metadata_path,
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
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

    print(f"Pipeline initialized successfully")

    # Load the dataset
    dataset = load_dataset(
        "google/frames-benchmark", split="test", token=HF_ACCESS_TOKEN
    )

    # Take first 100 items for this assignment
    dataset = dataset.take(100)

    filename = f"evaluation_results_{pipeline_type}_{llm_model.replace('/', '_')}.json"
    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)

    for item in tqdm(dataset, desc="Processing samples"):
        index = int(item["Unnamed: 0"])
        if index <= last_processed_index:
            continue

        # Use pipeline to get response (pass query directly, not the generated prompt)
        llm_response = get_pipeline_response(item["Prompt"], pipeline)
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

        save_result(filename, result)

    # Calculate and print summary statistics
    results = load_existing_results(filename)
    total_samples = len(results)
    correct_answers = sum(1 for r in results if r["evaluation_decision"] == "TRUE")
    accuracy = correct_answers / total_samples

    print(f"Pipeline: {pipeline_type}")
    print(f"LLM Model: {llm_model}")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")

    # Print accuracy by reasoning type
    reasoning_types = set(r["reasoning_type"] for r in results)
    for rt in reasoning_types:
        rt_samples = [r for r in results if r["reasoning_type"] == rt]
        rt_correct = sum(1 for r in rt_samples if r["evaluation_decision"] == "TRUE")
        rt_accuracy = rt_correct / len(rt_samples)
        print(f"Accuracy for {rt}: {rt_accuracy:.2%}")


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
        pipeline_choices = ["naive", "parallel"]
        pipeline_help = "Pipeline type: 'naive' or 'parallel'"

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
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default=None,
        help="Qdrant server URL",
    )
    parser.add_argument(
        "--qdrant_api_key",
        type=str,
        default=None,
        help="Qdrant API key",
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
        help="Decomposition model (for parallel pipeline)",
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
        default=5,
        help="Chunks after reranking (for naive pipeline)",
    )
    parser.add_argument(
        "--chunks_per_subquery",
        type=int,
        default=10,
        help="Chunks per subquery after reranking (for parallel pipeline)",
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
        qdrant_api_key=args.qdrant_api_key,
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
