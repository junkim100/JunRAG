#!/usr/bin/env python3
"""
Late chunking embedding tool using vLLM.
Embeds documents at token level then pools into chunks.

Usage:
    uv run tools/embedding/embed_late.py --dataset test
    uv run tools/embedding/embed_late.py --dataset val --tensor_parallel_size 8

Output:
    data/embeddings/test_late.json
    data/embeddings/val_late.json
"""

import hashlib
import json
import os
from typing import Dict, List

import fire
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from vllm import LLM


def get_segment_boundaries(
    text: str, segment_size: int, overlap: int, tokenizer
) -> List[Dict]:
    """Get segment boundaries for late chunking."""
    if hasattr(tokenizer, "encode") and hasattr(tokenizer.encode(text), "ids"):
        token_ids = tokenizer.encode(text).ids
    else:
        token_ids = tokenizer.encode(text, add_special_tokens=False)

    boundaries = []
    start_idx = 0

    while start_idx < len(token_ids):
        end_idx = min(start_idx + segment_size, len(token_ids))
        boundaries.append(
            {
                "start_token_idx": start_idx,
                "end_token_idx": end_idx,
                "token_length": end_idx - start_idx,
            }
        )
        if end_idx >= len(token_ids):
            break
        start_idx += segment_size - overlap

    return boundaries


def pool_tokens(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mean pool token embeddings."""
    return np.mean(embeddings[start:end], axis=0)


def main(
    dataset: str = None,
    input_file: str = None,
    output_file: str = None,
    model: str = "jinaai/jina-embeddings-v3",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    chunk_size: int = 1024,
    segment_size: int = 8192,
    overlap: int = 1024,
    text_key: str = "text",
    env_file: str = None,
):
    """
    Embed documents using late chunking.

    Args:
        dataset: Dataset name ("test" or "val") - auto-sets input/output paths
        input_file: JSON file with documents (overrides dataset)
        output_file: Output JSON file for embeddings (overrides dataset)
        model: Embedding model name
        tensor_parallel_size: Number of GPUs
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Maximum sequence length
        chunk_size: Tokens per chunk (default: 1024)
        segment_size: Tokens per segment (default: 8192, max_model_len)
        overlap: Overlap between segments (default: 1024 to preserve context)
        text_key: Key for text content
        env_file: Path to .env file
    """
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    if dataset and not input_file:
        # Read markdown files directly from data/markdown/
        input_file = os.path.join(project_root, "data", "markdown")
    if dataset and not output_file:
        output_file = os.path.join(
            project_root, "data", "embeddings", f"{dataset}_late.json"
        )

    if not output_file:
        print("Error: Provide --dataset or --output_file")
        return

    # Load markdown files
    if os.path.isdir(input_file):
        # input_file is a directory - load markdown files
        markdown_dir = input_file

        # Load URL mapping if it exists
        mapping_file = os.path.join(markdown_dir, f"{dataset}_url_mapping.json")
        url_mapping = {}
        if os.path.exists(mapping_file):
            with open(mapping_file, "r", encoding="utf-8") as f:
                url_mapping = json.load(f)
            print(f"Loaded URL mapping: {len(url_mapping)} entries")

        # Build reverse mapping: canonical_filename -> list of (item_id, article_idx)
        canonical_to_references = {}
        for key, canonical_filename in url_mapping.items():
            if canonical_filename not in canonical_to_references:
                canonical_to_references[canonical_filename] = []
            parts = key.split("_", 1)
            if len(parts) == 2:
                item_id, article_idx = parts[0], (
                    int(parts[1]) if parts[1].isdigit() else 0
                )
                canonical_to_references[canonical_filename].append(
                    (item_id, article_idx)
                )

        # Get canonical files
        if dataset:
            canonical_files = [
                f
                for f in os.listdir(markdown_dir)
                if f.endswith(".md") and f.startswith(f"{dataset}_canonical_")
            ]
        else:
            canonical_files = [
                f
                for f in os.listdir(markdown_dir)
                if f.endswith(".md") and "_canonical_" in f
            ]

        canonical_files.sort()
        print(
            f"Found {len(canonical_files)} canonical markdown files in {markdown_dir}"
        )

        documents = []
        for canonical_filename in canonical_files:
            filepath = os.path.join(markdown_dir, canonical_filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title from first heading
            title = canonical_filename
            lines = content.split("\n")
            for line in lines[:10]:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
                elif line.startswith("## "):
                    title = line[3:].strip()
                    break

            # Get all (item_id, article_idx) references for this canonical file
            references = canonical_to_references.get(canonical_filename, [])

            # Create only ONE document entry per canonical file (deduplication)
            # Store all references in metadata for tracking
            if references:
                # Use the first reference as primary, but store all references
                primary_item_id, primary_article_idx = references[0]
                metadata = {
                    "dataset": dataset or "unknown",
                    "item_id": primary_item_id,  # Primary reference
                    "article_idx": primary_article_idx,  # Primary reference
                    "markdown_file": canonical_filename,
                    "all_references": references,  # All (item_id, article_idx) pairs that reference this file
                }
            else:
                # No mapping found, treat as standalone file (backward compatibility)
                metadata = {
                    "dataset": dataset or "unknown",
                    "item_id": "unknown",
                    "article_idx": 0,
                    "markdown_file": canonical_filename,
                    "all_references": [],
                }

            documents.append(
                {
                    "text": content,
                    "title": title,
                    **metadata,
                }
            )
    else:
        # Legacy: input_file is a JSON file (for backward compatibility)
        print(f"Loading documents from JSON: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            documents = json.load(f)
        if not isinstance(documents, list):
            documents = [documents]

    print(f"Loaded {len(documents)} documents")

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # Initialize vLLM
    print(f"Loading model: {model}")
    print(f"  Tensor parallel: {tensor_parallel_size} GPUs")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        runner="pooling",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    all_chunks = []

    for doc_idx, doc in enumerate(tqdm(documents, desc="Processing documents")):
        text = doc.get(text_key, doc.get("text", ""))
        if not text.strip():
            continue

        metadata = {
            k: v
            for k, v in doc.items()
            if k != text_key and k != "text" and k != "embedding"
        }

        # Tokenize
        if hasattr(tokenizer, "encode"):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            token_ids = tokenizer.encode(text).ids

        # Get segment boundaries
        boundaries = get_segment_boundaries(text, segment_size, overlap, tokenizer)

        for seg_idx, boundary in enumerate(boundaries):
            seg_start = boundary["start_token_idx"]
            seg_end = min(boundary["end_token_idx"], len(token_ids))

            if seg_start >= seg_end:
                continue

            # Decode segment
            segment_token_ids = token_ids[seg_start:seg_end]

            # Truncate if needed to account for prefix
            prefix_text = ""
            if "jina-embeddings-v3" in model.lower():
                prefix_text = "Represent the document for retrieval: "
                # Tokenize prefix to get its length
                if hasattr(tokenizer, "encode"):
                    prefix_token_ids = tokenizer.encode(
                        prefix_text, add_special_tokens=False
                    )
                else:
                    prefix_token_ids = tokenizer.encode(prefix_text).ids
                prefix_len = len(prefix_token_ids)

                # Truncate segment to leave room for prefix
                max_segment_len = (
                    max_model_len - prefix_len - 10
                )  # 10 token safety margin
                if len(segment_token_ids) > max_segment_len:
                    segment_token_ids = segment_token_ids[:max_segment_len]
                    print(
                        f"Warning: Truncated segment {seg_idx} from {len(token_ids[seg_start:seg_end])} "
                        f"to {len(segment_token_ids)} tokens to fit max_model_len={max_model_len}"
                    )

            segment_text = tokenizer.decode(segment_token_ids, skip_special_tokens=True)

            # Add prefix for Jina v3
            if prefix_text:
                segment_text_prefixed = prefix_text + segment_text
            else:
                segment_text_prefixed = segment_text

            # Get token embeddings
            try:
                outputs = llm.encode(
                    [segment_text_prefixed], pooling_task="token_embed"
                )
                output_data = outputs[0].outputs.data
                if hasattr(output_data, "cpu"):
                    output_data = output_data.cpu()
                token_embeddings = np.asarray(output_data)
            except Exception as e:
                print(f"Warning: Failed to embed segment {seg_idx}: {e}")
                continue

            actual_tokens = len(token_embeddings)

            # Split into chunks
            chunk_start = 0

            while chunk_start < actual_tokens:
                chunk_end = min(chunk_start + chunk_size, actual_tokens)

                # Pool embeddings
                chunk_embedding = pool_tokens(token_embeddings, chunk_start, chunk_end)

                # Decode chunk text
                chunk_global_start = seg_start + chunk_start
                chunk_global_end = seg_start + chunk_end
                chunk_token_ids = token_ids[chunk_global_start:chunk_global_end]
                chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)

                # Generate stable chunk_id from content + metadata
                stable_key = f"{metadata.get('markdown_file', '')}_{len(all_chunks)}_{chunk_global_start}_{chunk_global_end}_{chunk_text[:100]}"
                chunk_id = int(hashlib.md5(stable_key.encode()).hexdigest()[:8], 16)

                chunk_data = {
                    "doc_id": doc_idx,
                    "chunk_index": len(all_chunks),
                    "chunk_id": chunk_id,  # Stable ID for deduplication
                    "segment_index": seg_idx,
                    "start_token_idx": chunk_global_start,
                    "end_token_idx": chunk_global_end,
                    "token_length": chunk_end - chunk_start,
                    "chunk_text": chunk_text,
                    "embedding": chunk_embedding.tolist(),
                    **metadata,
                }
                all_chunks.append(chunk_data)

                chunk_start = chunk_end

    # Save
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_chunks)} chunks to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
