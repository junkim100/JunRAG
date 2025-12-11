"""
Gradio Web UI for JunRAG.
Provides a ChatGPT/Gemini-like interface with settings sidebar.
"""

import json
import os
import sys
from pathlib import Path

# Load environment variables first
from dotenv import load_dotenv

load_dotenv()

import gradio as gr

# Add custom CSS for timing tooltip
TIMING_CSS = """
.assistant-message {
    position: relative;
}
.assistant-message:hover::after {
    content: attr(data-timing);
    position: absolute;
    bottom: 100%;
    left: 0;
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    white-space: pre-line;
    font-size: 12px;
    z-index: 1000;
    margin-bottom: 5px;
    max-width: 300px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
"""

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from junrag.pipelines import WebUIPipeline
from junrag.components.retrieval import load_metadata

# Metadata directory
METADATA_DIR = Path(__file__).parent.parent.parent / "data" / "metadata"

# Global pipeline instance
pipeline: WebUIPipeline = None


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
    embedding_model: str,
    reranker_model: str,
    llm_model: str,
    gpu_memory_utilization: float,
):
    """Initialize the pipeline with pre-loaded models."""
    global pipeline
    import traceback

    try:
        # Deinitialize existing pipeline if any
        if pipeline is not None:
            try:
                # Clean up models
                if hasattr(pipeline, "_embedders"):
                    for embedder in pipeline._embedders.values():
                        if hasattr(embedder, "llm"):
                            del embedder.llm
                    pipeline._embedders.clear()
                if hasattr(pipeline, "_rerankers"):
                    for reranker in pipeline._rerankers.values():
                        if hasattr(reranker, "llm"):
                            del reranker.llm
                    pipeline._rerankers.clear()
                pipeline = None
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")

        # Create pipeline instance (this is fast)
        pipeline = WebUIPipeline(
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            llm_model=llm_model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,  # Fixed default value
            rerank_batch_size=64,
        )

        # Load models (this can take a long time)
        # This will log progress, but we should catch and report any errors
        try:
            pipeline.load_models()
            return "✅ Pipeline initialized and models loaded successfully!"
        except Exception as load_error:
            # Clean up on load failure
            pipeline = None
            error_msg = str(load_error)
            # Include traceback for debugging, but keep it concise
            import traceback

            tb = traceback.format_exc()
            # Only include last few lines of traceback to avoid overwhelming the UI
            tb_lines = tb.split("\n")
            if len(tb_lines) > 10:
                tb = "\n".join(tb_lines[-10:])
            return f"❌ Error loading models: {error_msg}\n\n{tb}"

    except Exception as e:
        # Error during pipeline creation
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
        # Clean up models
        if hasattr(pipeline, "_embedders"):
            for embedder in pipeline._embedders.values():
                if hasattr(embedder, "llm"):
                    try:
                        del embedder.llm
                    except:
                        pass
            pipeline._embedders.clear()
        if hasattr(pipeline, "_rerankers"):
            for reranker in pipeline._rerankers.values():
                if hasattr(reranker, "llm"):
                    try:
                        del reranker.llm
                    except:
                        pass
            pipeline._rerankers.clear()

        pipeline = None
        return "✅ Pipeline deinitialized and models unloaded."
    except Exception as e:
        return f"❌ Error deinitializing pipeline: {str(e)}"


def get_init_status():
    """Get current initialization status."""
    global pipeline
    if (
        pipeline is not None
        and hasattr(pipeline, "_models_loaded")
        and pipeline._models_loaded
    ):
        return "✅ Pipeline initialized and models loaded successfully!"
    return "Not initialized"


def format_timing_tooltip(timing: dict) -> str:
    """Format timing information as a tooltip string."""
    if not timing:
        return ""
    tooltip = "⏱️ Timing Statistics:\n"
    tooltip += f"• Total: {timing.get('total_runtime', 0):.2f}s\n"
    tooltip += f"• Decomposition: {timing.get('step1_decomposition', 0):.2f}s\n"
    tooltip += f"• Embedding: {timing.get('step2_embedding', 0):.2f}s\n"
    tooltip += f"• Retrieve+Rerank: {timing.get('step3_retrieve_rerank', 0):.2f}s\n"
    tooltip += f"• Merge+Select: {timing.get('step4_merge_select', 0):.2f}s\n"
    tooltip += f"• Generation: {timing.get('step5_generation', 0):.2f}s"
    return tooltip


def process_query(
    query: str,
    vectordb: str,
    reasoning_effort: str,
    retrieval_top_k: int,
    chunks_per_subquery: int,
    max_cap: int,
    min_floor: int,
):
    """Process a query through the pipeline."""
    global pipeline

    if pipeline is None:
        return "❌ Pipeline not initialized. Please initialize first.", "", {}

    if not query or not query.strip():
        return "❌ Please enter a query.", "", {}

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
            return f"❌ VectorDB option not found: {vectordb}", "", {}

        metadata_path = vectordb_options[vectordb]
        metadata = load_metadata(metadata_path)

        # Update pipeline settings
        pipeline.retrieval_top_k = retrieval_top_k
        pipeline.chunks_per_subquery = chunks_per_subquery
        pipeline.max_cap = max_cap
        pipeline.min_floor = min_floor
        pipeline.collection_name = metadata.get("collection_name")
        pipeline.qdrant_url = metadata.get("qdrant_url")
        pipeline.qdrant_api_key = metadata.get("qdrant_api_key")
        # Update generator reasoning effort (if supported)
        if hasattr(pipeline.generator, "reasoning_effort"):
            pipeline.generator.reasoning_effort = reasoning_effort

        # Run pipeline
        result = pipeline.run(query)

        # Format response
        answer = result.answer
        timing = result.timing if hasattr(result, "timing") else {}

        return answer, "✅ Query processed successfully!", timing

    except Exception as e:
        import traceback

        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return "", error_msg, {}


def create_ui():
    """Create the Gradio interface."""
    # Define vectordb options with specific file paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    vectordb_options = {
        "late chunking": str(
            PROJECT_ROOT / "data" / "metadata" / "test_late_metadata.json"
        ),
        "contextual retrieval": str(
            PROJECT_ROOT / "data" / "metadata" / "val_contextual_metadata.json"
        ),
    }

    with gr.Blocks(title="JunRAG", css=TIMING_CSS) as app:
        gr.Markdown(
            """
            # 🔍 JunRAG - Advanced RAG Pipeline
            Ask complex multi-hop questions and get comprehensive answers.
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

            with gr.Column(scale=1):
                # Settings sidebar
                gr.Markdown("### ⚙️ Settings")

                # Model initialization
                gr.Markdown("#### Model Configuration")
                embedding_model = gr.Textbox(
                    label="Embedding Model",
                    value="jinaai/jina-embeddings-v3",
                )
                reranker_model = gr.Textbox(
                    label="Reranker Model",
                    value="Qwen/Qwen3-Reranker-4B",
                )
                llm_model = gr.Textbox(
                    label="LLM Model",
                    value="gpt-5.1-2025-11-13",
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
                    lines=3,  # Make it taller so the text fits
                    max_lines=5,  # Optionally, allow for more lines if needed
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
                )
                retrieval_top_k = gr.Slider(
                    label="Retrieval Top K",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=5,
                )
                chunks_per_subquery = gr.Slider(
                    label="Chunks per Subquery",
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                )
                max_cap = gr.Slider(
                    label="Max Cap",
                    minimum=10,
                    maximum=50,
                    value=25,
                    step=5,
                )
                min_floor = gr.Slider(
                    label="Min Floor",
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                )

        # Event handlers
        def init_with_status(*args):
            """Initialize pipeline and show loading status."""
            # Show loading status immediately
            yield "⏳ Initializing pipeline and loading models... This may take several minutes."

            # Run initialization (this is a blocking call)
            result = initialize_pipeline(*args)

            # Return final status
            yield result

        init_btn.click(
            init_with_status,
            inputs=[
                embedding_model,
                reranker_model,
                llm_model,
                gpu_memory_utilization,
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
            chunks_per_subquery_val,
            max_cap_val,
            min_floor_val,
        ):
            """Handle chat response with loading indicator."""
            if not message or not message.strip():
                return history, ""

            # Immediately add user message to history
            history.append({"role": "user", "content": message})
            # Add loading indicator as assistant response
            history.append({"role": "assistant", "content": "..."})
            yield history, ""

            # Process query
            answer, status, timing_info = process_query(
                message,
                vectordb_val,
                reasoning_effort_val,
                retrieval_top_k_val,
                chunks_per_subquery_val,
                max_cap_val,
                min_floor_val,
            )

            # Update assistant response with actual answer
            # Add timing info as tooltip on hover
            if answer:
                if timing_info:
                    tooltip = format_timing_tooltip(timing_info)
                    # Add a small timing indicator that shows tooltip on hover
                    # We'll append a small icon/indicator to the answer
                    answer_with_timing = f"{answer}\n\n<span title='{tooltip.replace(chr(10), '&#10;')}' style='cursor: help; opacity: 0.6; font-size: 0.8em;'>⏱️ Hover for timing stats</span>"
                    history[-1] = {"role": "assistant", "content": answer_with_timing}
                else:
                    history[-1] = {"role": "assistant", "content": answer}
            elif status:
                history[-1] = {"role": "assistant", "content": status}

            yield history, ""

        submit_btn.click(
            respond,
            inputs=[
                query_input,
                chatbot,
                vectordb,
                reasoning_effort,
                retrieval_top_k,
                chunks_per_subquery,
                max_cap,
                min_floor,
            ],
            outputs=[chatbot, query_input],
        )

        query_input.submit(
            respond,
            inputs=[
                query_input,
                chatbot,
                vectordb,
                reasoning_effort,
                retrieval_top_k,
                chunks_per_subquery,
                max_cap,
                min_floor,
            ],
            outputs=[chatbot, query_input],
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
