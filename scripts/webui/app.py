"""
Gradio Web UI for JunRAG.
Provides a ChatGPT/Gemini-like interface with settings sidebar.
"""

import json
import os
import sys
from pathlib import Path

import gradio as gr

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
    max_model_len: int,
):
    """Initialize the pipeline with pre-loaded models."""
    global pipeline

    try:
        pipeline = WebUIPipeline(
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            llm_model=llm_model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            rerank_batch_size=64,
        )
        pipeline.load_models()
        return "✅ Pipeline initialized and models loaded successfully!"
    except Exception as e:
        return f"❌ Error initializing pipeline: {str(e)}"


def process_query(
    query: str,
    metadata_file: str,
    retrieval_top_k: int,
    chunks_per_subquery: int,
    max_cap: int,
    min_floor: int,
    reasoning_effort: str,
):
    """Process a query through the pipeline."""
    global pipeline

    if pipeline is None:
        return "❌ Pipeline not initialized. Please initialize first.", ""

    if not query or not query.strip():
        return "❌ Please enter a query.", ""

    try:
        # Load metadata
        metadata_files = load_metadata_files()
        if metadata_file not in metadata_files:
            return f"❌ Metadata file not found: {metadata_file}", ""

        metadata_path = metadata_files[metadata_file]["path"]
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
        timing_str = f"\n\n**Timing:**\n"
        timing_str += f"- Total: {timing.get('total_runtime', 0):.2f}s\n"
        timing_str += f"- Decomposition: {timing.get('step1_decomposition', 0):.2f}s\n"
        timing_str += f"- Embedding: {timing.get('step2_embedding', 0):.2f}s\n"
        timing_str += (
            f"- Retrieve+Rerank: {timing.get('step3_retrieve_rerank', 0):.2f}s\n"
        )
        timing_str += f"- Merge+Select: {timing.get('step4_merge_select', 0):.2f}s\n"
        timing_str += f"- Generation: {timing.get('step5_generation', 0):.2f}s\n"

        return answer + timing_str, "✅ Query processed successfully!"

    except Exception as e:
        import traceback

        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return "", error_msg


def create_ui():
    """Create the Gradio interface."""
    metadata_files = load_metadata_files()
    metadata_options = (
        list(metadata_files.keys()) if metadata_files else ["No metadata files found"]
    )

    with gr.Blocks(title="JunRAG", theme=gr.themes.Soft()) as app:
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
                    show_copy_button=True,
                    avatar_images=(None, "🤖"),
                    bubble_full_width=False,
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
                max_model_len = gr.Number(
                    label="Max Model Length",
                    value=8192,
                    precision=0,
                )
                init_btn = gr.Button("Initialize Pipeline", variant="secondary")
                init_status = gr.Textbox(
                    label="Initialization Status", interactive=False
                )

                gr.Markdown("#### Query Settings")
                metadata_file = gr.Dropdown(
                    label="Metadata File",
                    choices=metadata_options,
                    value=metadata_options[0] if metadata_options else None,
                )
                retrieval_top_k = gr.Slider(
                    label="Retrieval Top K",
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=10,
                )
                chunks_per_subquery = gr.Slider(
                    label="Chunks per Subquery",
                    minimum=5,
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
                reasoning_effort = gr.Dropdown(
                    label="Reasoning Effort",
                    choices=["low", "medium", "high"],
                    value="medium",
                )

        # Event handlers
        init_btn.click(
            initialize_pipeline,
            inputs=[
                embedding_model,
                reranker_model,
                llm_model,
                gpu_memory_utilization,
                max_model_len,
            ],
            outputs=init_status,
        )

        def respond(message, history):
            """Handle chat response."""
            if not message or not message.strip():
                return history, ""

            answer, status = process_query(
                message,
                metadata_file.value,
                retrieval_top_k.value,
                chunks_per_subquery.value,
                max_cap.value,
                min_floor.value,
                reasoning_effort.value,
            )

            if answer:
                history.append((message, answer))
            elif status:
                history.append((message, status))

            return history, ""

        submit_btn.click(
            respond,
            inputs=[query_input, chatbot],
            outputs=[chatbot, query_input],
        )

        query_input.submit(
            respond,
            inputs=[query_input, chatbot],
            outputs=[chatbot, query_input],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

