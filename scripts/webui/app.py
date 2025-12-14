"""
Gradio Web UI for JunRAG.
Provides a ChatGPT/Gemini-like interface with settings sidebar.
Supports multiple pipeline modes: Naive, Parallel, and Sequential Decomposition.
"""

import json
import os
import sys
from pathlib import Path

# Load environment variables first
from dotenv import load_dotenv

load_dotenv()

import gradio as gr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from junrag.pipelines import (
    NaivePipeline,
    ParallelPipeline,
    SequentialDecompositionPipeline,
)
from junrag.components.retrieval import load_metadata

# Metadata directory
METADATA_DIR = Path(__file__).parent.parent.parent / "data" / "metadata"

# Global pipeline instance
pipeline = None

# Reasoning models that support reasoning_effort
REASONING_MODELS = {
    "gpt-5.1-2025-11-13",
    "o3",
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-preview",
}

# Model options
EMBEDDING_MODELS = [
    "jinaai/jina-embeddings-v3",
]

RERANKER_MODELS = [
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-8B",
    "jinaai/jina-reranker-v3",
]

LLM_MODELS = [
    "gpt-5.1-2025-11-13 (low)",
    "gpt-5.1-2025-11-13 (medium)",
    "gpt-5.1-2025-11-13 (high)",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "o3 (low)",
    "o3 (medium)",
    "o3 (high)",
    "o3-mini (low)",
    "o3-mini (medium)",
    "o3-mini (high)",
    "o1 (low)",
    "o1 (medium)",
    "o1 (high)",
    "o1-mini (low)",
    "o1-mini (medium)",
    "o1-mini (high)",
    "o1-preview (low)",
    "o1-preview (medium)",
    "o1-preview (high)",
]

DECOMPOSITION_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
]


def parse_model_with_effort(model_str: str):
    """Parse model string with reasoning effort, e.g., 'gpt-5.1-2025-11-13 (medium)' -> ('gpt-5.1-2025-11-13', 'medium')"""
    """Parse model string with reasoning effort, e.g., 'gpt-5.1-2025-11-13 (medium)' -> ('gpt-5.1-2025-11-13', 'medium')"""
    if " (" in model_str and model_str.endswith(")"):
        parts = model_str.rsplit(" (", 1)
        model = parts[0]
        effort = parts[1].rstrip(")")
        return model, effort
    return model_str, "medium"  # Default effort


def load_metadata_files():
    """Load available metadata files from the metadata directory."""
    metadata_files = {}
    if METADATA_DIR.exists():
        for file in METADATA_DIR.glob("*.json"):
            try:
                with open(file, "r") as f:
                    metadata = json.load(f)
                    metadata_files[file.name] = {
                        "path": str(file),
                        "collection_name": metadata.get("collection_name", ""),
                        "dataset": metadata.get("dataset", ""),
                        "embedding_type": metadata.get("embedding_type", ""),
                    }
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return metadata_files


def initialize_pipeline(
    pipeline_mode: str,
    embedding_model: str,
    reranker_model: str,
    llm_model: str,
    decomposition_model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    chunks_per_subquery: int = 10,
    max_cap: int = 25,
    min_floor: int = 5,
):
    """Initialize the pipeline with pre-loaded models."""
    global pipeline
    import traceback

    try:
        # Deinitialize existing pipeline if any
        if pipeline is not None:
            try:
                if hasattr(pipeline, "_cleanup_all_models"):
                    pipeline._cleanup_all_models()
                elif hasattr(pipeline, "_cleanup_rerankers"):
                    pipeline._cleanup_rerankers()
                pipeline = None
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")

        # Parse LLM model and reasoning effort
        llm_model_name, reasoning_effort = parse_model_with_effort(llm_model)

        # Create pipeline instance based on mode
        common_kwargs = {
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
            "llm_model": llm_model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": 8192,
        }

        if pipeline_mode == "naive":
            pipeline = NaivePipeline(**common_kwargs)
        elif pipeline_mode == "parallel":
            pipeline = ParallelPipeline(
                decomposition_model=decomposition_model,
                chunks_per_subquery=chunks_per_subquery,
                max_cap=max_cap,
                min_floor=min_floor,
                **common_kwargs,
            )
        elif pipeline_mode == "sequential_decomposition":
            pipeline = SequentialDecompositionPipeline(
                decomposition_model=decomposition_model,
                rerank_per_subquery=chunks_per_subquery,
                **common_kwargs,
            )
        else:
            return f"❌ Unknown pipeline mode: {pipeline_mode}"

        # Set reasoning effort if supported
        if hasattr(pipeline, "generator") and hasattr(
            pipeline.generator, "reasoning_effort"
        ):
            pipeline.generator.reasoning_effort = reasoning_effort

        return "✅ Pipeline initialized successfully! Models will be loaded on first query."

    except Exception as e:
        pipeline = None
        error_msg = str(e)
        import traceback

        tb = traceback.format_exc()
        tb_lines = tb.split("\n")
        if len(tb_lines) > 10:
            tb = "\n".join(tb_lines[-10:])
        return f"❌ Error initializing pipeline: {error_msg}\n\n{tb}"


def deinitialize_pipeline():
    """Deinitialize and clean up the pipeline."""
    global pipeline

    if pipeline is None:
        return "ℹ️ Pipeline is not initialized."

    try:
        if hasattr(pipeline, "_cleanup_all_models"):
            pipeline._cleanup_all_models()
        elif hasattr(pipeline, "_cleanup_rerankers"):
            pipeline._cleanup_rerankers()
        pipeline = None
        return "✅ Pipeline deinitialized and models unloaded."
    except Exception as e:
        return f"❌ Error deinitializing pipeline: {str(e)}"


def get_init_status():
    """Get current initialization status."""
    global pipeline
    if pipeline is not None:
        return "✅ Pipeline initialized (models load on first query)"
    return "Not initialized"


def format_chain_output(chain):
    """Format chain information for display."""
    chain_id = chain.get("chain_id", 1)
    n_steps = chain.get("n_steps", 0)
    steps = chain.get("steps", [])
    chain_answer = chain.get("chain_answer", "")

    output = f"### Chain {chain_id} ({n_steps} steps)\n\n"

    for step in steps:
        step_idx = step.get("step_idx", 0)
        original_subquery = step.get("original_subquery", "")
        rewritten_subquery = step.get("rewritten_subquery", "")
        answer = step.get("answer", "")
        used_internal_knowledge = step.get("used_internal_knowledge", False)

        output += f"**Step {step_idx}:**\n"
        output += f"- **Subquery:** {original_subquery}\n"
        if rewritten_subquery != original_subquery:
            output += f"- **Rewritten:** {rewritten_subquery}\n"
        output += f"- **Answer:** {answer}\n"
        if used_internal_knowledge:
            output += f"- ⚠️ *Used internal knowledge*\n"
        output += "\n"

    output += f"**Chain Answer:** {chain_answer}\n\n"
    return output


def process_query(
    query: str,
    vectordb: str,
    reasoning_effort: str,
    retrieval_top_k: int,
    rerank_per_subquery: int,
    chunks_per_subquery: int,
    max_cap: int,
    min_floor: int,
):
    """Process a query through the pipeline."""
    global pipeline

    if pipeline is None:
        return "❌ Pipeline not initialized. Please initialize first.", "", ""

    if not query or not query.strip():
        return "❌ Please enter a query.", "", ""

    try:
        # Get metadata path from vectordb selection
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        vectordb_options = {
            "late chunking": str(
                PROJECT_ROOT / "data" / "metadata" / "test_late_metadata.json"
            ),
            "contextual retrieval": str(
                PROJECT_ROOT / "data" / "metadata" / "test_contextual_metadata.json"
            ),
        }

        if vectordb not in vectordb_options:
            return f"❌ VectorDB option not found: {vectordb}", "", ""

        metadata_path = vectordb_options[vectordb]
        metadata = load_metadata(metadata_path)

        # Update pipeline settings
        pipeline.retrieval_top_k = retrieval_top_k
        pipeline.collection_name = metadata.get("collection_name")
        pipeline.qdrant_url = metadata.get("qdrant_url")
        pipeline.qdrant_api_key = metadata.get("qdrant_api_key")

        # Update pipeline-specific settings
        if isinstance(pipeline, ParallelPipeline):
            pipeline.chunks_per_subquery = chunks_per_subquery
            pipeline.max_cap = max_cap
            pipeline.min_floor = min_floor
        elif isinstance(pipeline, SequentialDecompositionPipeline):
            pipeline.rerank_per_subquery = rerank_per_subquery

        # Update generator reasoning effort (if supported)
        if hasattr(pipeline, "generator") and hasattr(
            pipeline.generator, "reasoning_effort"
        ):
            pipeline.generator.reasoning_effort = reasoning_effort

        # Run pipeline (models load automatically on first query)
        # For sequential decomposition: keep models loaded (cleanup_models=False) to enable
        # smart reranker conflict detection - only conflicting rerankers are unloaded when
        # n_chains changes between queries, keeping non-conflicting models loaded for efficiency
        cleanup_models = False  # Keep models loaded for faster subsequent queries
        if isinstance(pipeline, SequentialDecompositionPipeline):
            result = pipeline.run(query, cleanup_models=cleanup_models)
        else:
            result = pipeline.run(query)

        # Format response
        answer = result.answer

        # Format chain information (only for sequential decomposition)
        chains_info = ""
        if isinstance(pipeline, SequentialDecompositionPipeline):
            if hasattr(result, "chains") and result.chains:
                chains_info = "## Decomposition Chains\n\n"
                for chain in result.chains:
                    chains_info += format_chain_output(chain)

        # Format timing information
        timing_info = ""
        if hasattr(result, "timing") and result.timing:
            timing = result.timing
            timing_info = "## Timing Information\n\n"
            timing_info += (
                f"- **Total Runtime:** {timing.get('total_runtime', 0):.2f}s\n"
            )
            if "preinit_models" in timing:
                timing_info += (
                    f"- **Model Loading:** {timing.get('preinit_models', 0):.2f}s\n"
                )
            if "step1_decomposition" in timing:
                timing_info += f"- **Decomposition:** {timing.get('step1_decomposition', 0):.2f}s\n"
            if "step3_chain_processing" in timing:
                timing_info += f"- **Chain Processing:** {timing.get('step3_chain_processing', 0):.2f}s\n"
            if "step4_generation" in timing:
                timing_info += (
                    f"- **Generation:** {timing.get('step4_generation', 0):.2f}s\n"
                )

        return answer, chains_info, timing_info

    except Exception as e:
        import traceback

        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return "", "", error_msg


def create_ui():
    """Create the Gradio interface."""
    # Define vectordb options with specific file paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    vectordb_options = {
        "late chunking": str(
            PROJECT_ROOT / "data" / "metadata" / "test_late_metadata.json"
        ),
        "contextual retrieval": str(
            PROJECT_ROOT / "data" / "metadata" / "test_contextual_metadata.json"
        ),
    }

    with gr.Blocks(title="JunRAG") as app:
        gr.Markdown(
            """
            # 🔍 JunRAG - Multi-Pipeline RAG System
            Ask complex multi-hop questions and get comprehensive answers with decomposition chains.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=600,
                    avatar_images=(None, "🤖"),
                )
                with gr.Row():
                    query_input = gr.Textbox(
                        label="",
                        placeholder="Ask a complex multi-hop question...",
                        lines=2,
                        scale=4,
                        container=False,
                    )
                    submit_btn = gr.Button(
                        "Send", variant="primary", scale=1, size="lg"
                    )

                # Chain information display
                chains_display = gr.Markdown(
                    label="Decomposition Chains",
                    value="",
                    visible=True,
                )

                # Timing information display
                timing_display = gr.Markdown(
                    label="Timing Information",
                    value="",
                    visible=True,
                )

            with gr.Column(scale=1):
                # Settings sidebar
                gr.Markdown("### ⚙️ Settings")

                # Pipeline mode selection
                gr.Markdown("#### Pipeline Mode")
                pipeline_mode = gr.Dropdown(
                    label="Pipeline Mode",
                    choices=["naive", "parallel", "sequential_decomposition"],
                    value="sequential_decomposition",
                    info="Select the pipeline type to use",
                )

                # Model initialization
                gr.Markdown("#### Model Configuration")
                embedding_model = gr.Dropdown(
                    label="Embedding Model",
                    choices=EMBEDDING_MODELS,
                    value=EMBEDDING_MODELS[0],
                )
                reranker_model = gr.Dropdown(
                    label="Reranker Model",
                    choices=RERANKER_MODELS,
                    value=RERANKER_MODELS[0],
                )
                llm_model = gr.Dropdown(
                    label="LLM Model",
                    choices=LLM_MODELS,
                    value="gpt-5.1-2025-11-13 (medium)",
                    info="Reasoning models show effort level in parentheses",
                )
                decomposition_model = gr.Dropdown(
                    label="Decomposition Model",
                    choices=DECOMPOSITION_MODELS,
                    value="gpt-4o",
                )
                tensor_parallel_size = gr.Slider(
                    label="Tensor Parallel Size (GPUs)",
                    minimum=1,
                    maximum=8,
                    value=8,
                    step=1,
                )
                gpu_memory_utilization = gr.Slider(
                    label="GPU Memory Utilization",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                )
                init_btn = gr.Button("Initialize Pipeline", variant="secondary")
                deinit_btn = gr.Button("Deinitialize Pipeline", variant="stop")
                init_status = gr.Textbox(
                    label="Initialization Status",
                    interactive=False,
                    value=get_init_status(),
                    lines=3,
                    max_lines=5,
                )

                gr.Markdown("#### Query Settings")
                vectordb = gr.Dropdown(
                    label="vectordb",
                    choices=list(vectordb_options.keys()),
                    value=(
                        list(vectordb_options.keys())[0] if vectordb_options else None
                    ),
                )
                reasoning_effort = gr.Dropdown(
                    label="Reasoning Effort",
                    choices=["low", "medium", "high"],
                    value="medium",
                    info="Only used if LLM model supports reasoning",
                )
                retrieval_top_k = gr.Slider(
                    label="Retrieval Top K",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=5,
                )
                rerank_per_subquery = gr.Slider(
                    label="Rerank per Subquery",
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    visible=True,
                )
                chunks_per_subquery = gr.Slider(
                    label="Chunks per Subquery (Parallel)",
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    visible=False,
                )
                max_cap = gr.Slider(
                    label="Max Cap (Parallel)",
                    minimum=10,
                    maximum=50,
                    value=25,
                    step=5,
                    visible=False,
                )
                min_floor = gr.Slider(
                    label="Min Floor (Parallel)",
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    visible=False,
                )

        # Show/hide parallel-specific settings based on pipeline mode
        def update_ui_for_pipeline_mode(mode):
            """Update UI visibility based on pipeline mode."""
            if mode == "parallel":
                return (
                    gr.update(visible=False),  # rerank_per_subquery
                    gr.update(visible=True),  # chunks_per_subquery
                    gr.update(visible=True),  # max_cap
                    gr.update(visible=True),  # min_floor
                )
            elif mode == "sequential_decomposition":
                return (
                    gr.update(visible=True),  # rerank_per_subquery
                    gr.update(visible=False),  # chunks_per_subquery
                    gr.update(visible=False),  # max_cap
                    gr.update(visible=False),  # min_floor
                )
            else:  # naive
                return (
                    gr.update(visible=False),  # rerank_per_subquery
                    gr.update(visible=False),  # chunks_per_subquery
                    gr.update(visible=False),  # max_cap
                    gr.update(visible=False),  # min_floor
                )

        pipeline_mode.change(
            update_ui_for_pipeline_mode,
            inputs=[pipeline_mode],
            outputs=[rerank_per_subquery, chunks_per_subquery, max_cap, min_floor],
        )

        # Event handlers
        def init_with_status(*args):
            """Initialize pipeline and show loading status."""
            yield "⏳ Initializing pipeline..."

            result = initialize_pipeline(*args)
            yield result

        init_btn.click(
            init_with_status,
            inputs=[
                pipeline_mode,
                embedding_model,
                reranker_model,
                llm_model,
                decomposition_model,
                tensor_parallel_size,
                gpu_memory_utilization,
                chunks_per_subquery,
                max_cap,
                min_floor,
            ],
            outputs=init_status,
        )

        deinit_btn.click(
            deinitialize_pipeline,
            inputs=[],
            outputs=init_status,
        )

        def respond(
            message,
            history,
            vectordb_val,
            reasoning_effort_val,
            retrieval_top_k_val,
            rerank_per_subquery_val,
            chunks_per_subquery_val,
            max_cap_val,
            min_floor_val,
        ):
            """Handle chat response with loading indicator."""
            if not message or not message.strip():
                return history, "", "", ""

            # Immediately add user message to history
            history.append({"role": "user", "content": message})
            # Add loading indicator as assistant response
            history.append({"role": "assistant", "content": "..."})
            yield history, "", "", ""

            # Process query
            answer, chains_info, timing_info = process_query(
                message,
                vectordb_val,
                reasoning_effort_val,
                retrieval_top_k_val,
                rerank_per_subquery_val,
                chunks_per_subquery_val,
                max_cap_val,
                min_floor_val,
            )

            # Update assistant response with actual answer
            if answer:
                history[-1] = {"role": "assistant", "content": answer}
            else:
                history[-1] = {
                    "role": "assistant",
                    "content": "Error processing query.",
                }

            yield history, chains_info, timing_info, ""

        submit_btn.click(
            respond,
            inputs=[
                query_input,
                chatbot,
                vectordb,
                reasoning_effort,
                retrieval_top_k,
                rerank_per_subquery,
                chunks_per_subquery,
                max_cap,
                min_floor,
            ],
            outputs=[chatbot, chains_display, timing_display, query_input],
        )

        query_input.submit(
            respond,
            inputs=[
                query_input,
                chatbot,
                vectordb,
                reasoning_effort,
                retrieval_top_k,
                rerank_per_subquery,
                chunks_per_subquery,
                max_cap,
                min_floor,
            ],
            outputs=[chatbot, chains_display, timing_display, query_input],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(),
    )
