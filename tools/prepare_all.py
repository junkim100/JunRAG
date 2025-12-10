#!/usr/bin/env python3
"""
Master script to prepare all 4 Qdrant collections.
Runs the entire pipeline: download → fetch → embed → upload.

Usage:
    uv run tools/prepare_all.py --qdrant_url https://your-qdrant.cloud.qdrant.io
    uv run tools/prepare_all.py --qdrant_url https://... --tensor_parallel_size 8
    uv run tools/prepare_all.py --qdrant_url https://... --qdrant_api_key your_key
    uv run tools/prepare_all.py --skip_download --skip_fetch --skip_embed  # Resume from upload

Output:
    data/metadata/test_late_metadata.json
    data/metadata/test_contextual_metadata.json
    data/metadata/val_late_metadata.json
    data/metadata/val_contextual_metadata.json
"""

import os
import subprocess
import sys

import fire
from dotenv import load_dotenv


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with exit code {result.returncode}")
        return False

    print(f"\n✓ {description} completed successfully")
    return True


def main(
    qdrant_url: str = None,
    qdrant_api_key: str = None,
    dataset: str = "all",
    embedding_type: str = "all",
    tensor_parallel_size: int = 1,
    max_workers: int = 20,
    recreate: bool = True,
    skip_download: bool = False,
    skip_fetch: bool = False,
    skip_embed: bool = False,
    skip_upload: bool = False,
    env_file: str = None,
):
    """
    Prepare all Qdrant collections for JunRAG.

    Args:
        qdrant_url: Qdrant server URL (or set QDRANT_URL env var)
        qdrant_api_key: Qdrant API key (or set QDRANT_API_KEY env var)
        dataset: Dataset to process: "test", "val", or "all" (default: "all")
        embedding_type: Embedding type: "late", "contextual", or "all" (default: "all")
        tensor_parallel_size: Number of GPUs for embedding
        max_workers: Number of parallel workers for Wikipedia fetching
        recreate: Recreate Qdrant collections if they exist
        skip_download: Skip dataset download step
        skip_fetch: Skip Wikipedia fetch step
        skip_embed: Skip embedding step
        skip_upload: Skip Qdrant upload step
        env_file: Path to .env file
    """
    # Load environment variables
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Get qdrant_url from env if not provided
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
    if not qdrant_url:
        print(
            "ERROR: qdrant_url is required. Provide --qdrant_url or set QDRANT_URL in .env file"
        )
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable

    datasets = ["test", "val"] if dataset == "all" else [dataset]
    embedding_types = (
        ["late", "contextual"] if embedding_type == "all" else [embedding_type]
    )

    print("\n" + "=" * 60)
    print("JUNRAG COLLECTION PREPARATION")
    print("=" * 60)
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Datasets: {datasets}")
    print(f"Embedding types: {embedding_types}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Collections to create: {len(datasets) * len(embedding_types)}")

    # Step 1: Download dataset
    if not skip_download:
        cmd = [python, os.path.join(script_dir, "dataset", "download_dataset.py")]
        if not run_command(cmd, "Download FRAMES dataset"):
            return

    # Step 2: Fetch Wikipedia content
    if not skip_fetch:
        cmd = [
            python,
            os.path.join(script_dir, "preprocessing", "fetch_wikipedia.py"),
            "--dataset",
            dataset,
            "--max_workers",
            str(max_workers),
        ]
        if not run_command(cmd, "Fetch Wikipedia content"):
            return

    # Step 3 & 4: Embed and upload for each combination
    for ds in datasets:
        for emb_type in embedding_types:
            collection_name = f"{ds}_{emb_type}"

            # Step 3: Create embeddings
            if not skip_embed:
                if emb_type == "late":
                    embed_script = os.path.join(
                        script_dir, "embedding", "embed_late.py"
                    )
                else:
                    embed_script = os.path.join(
                        script_dir, "embedding", "embed_contextual.py"
                    )

                cmd = [
                    python,
                    embed_script,
                    "--dataset",
                    ds,
                    "--tensor_parallel_size",
                    str(tensor_parallel_size),
                ]
                if not run_command(cmd, f"Create {emb_type} embeddings for {ds}"):
                    continue

            # Step 4: Upload to Qdrant
            if not skip_upload:
                cmd = [
                    python,
                    os.path.join(script_dir, "qdrant", "upload_to_qdrant.py"),
                    "--qdrant_url",
                    qdrant_url,
                    "--dataset",
                    ds,
                    "--embedding_type",
                    emb_type,
                ]
                if qdrant_api_key:
                    cmd.extend(["--qdrant_api_key", qdrant_api_key])
                if recreate:
                    cmd.append("--recreate")

                if not run_command(cmd, f"Upload {collection_name} to Qdrant"):
                    continue

            print(f"\n✓ Collection '{collection_name}' ready!")

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print("\nCreated collections:")
    for ds in datasets:
        for emb_type in embedding_types:
            metadata_path = os.path.join(
                script_dir, "..", "data", "metadata", f"{ds}_{emb_type}_metadata.json"
            )
            if os.path.exists(metadata_path):
                print(f"  ✓ {ds}_{emb_type}")
            else:
                print(f"  ✗ {ds}_{emb_type} (metadata not found)")

    print("\nMetadata files saved to: data/metadata/")
    print("\nNext steps:")
    print("  # Run naive pipeline:")
    print(
        "  junrag naive --query 'Your question' --metadata_path data/metadata/test_late_metadata.json"
    )
    print("\n  # Run full pipeline:")
    print(
        "  junrag full --query 'Complex question' --metadata_path data/metadata/test_late_metadata.json"
    )


if __name__ == "__main__":
    fire.Fire(main)
