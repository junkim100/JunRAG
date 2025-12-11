#!/usr/bin/env python3
"""
Contextual embedding tool using vLLM and GPT-4o batch API.

Process:
1. Split documents into 1024-token chunks (no overlap)
2. Use GPT-4o batch API to generate succinct context for each chunk
3. Prepend context to original chunk
4. Embed the combined chunks using vLLM

Usage:
    uv run tools/embedding/embed_contextual.py --dataset test --tensor_parallel_size 8
    uv run tools/embedding/embed_contextual.py --dataset val --tensor_parallel_size 8

Output:
    data/embeddings/test_contextual.json
    data/embeddings/val_contextual.json
    (additionally, saves raw batch API results as data/embeddings/[dataset]_context_batchapi.json)
"""

import hashlib
import json
import os
import tempfile
import time
from typing import Dict, List

import fire
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from vllm import LLM


def get_tokenizer(model: str):
    """Load tokenizer for the embedding model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int = 1024) -> List[str]:
    """Split text into chunks of chunk_size tokens (no overlap)."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start_idx = 0

    while start_idx < len(token_ids):
        end_idx = min(start_idx + chunk_size, len(token_ids))
        chunk_ids = token_ids[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
        start_idx = end_idx

    return chunks


def create_batch_messages(document_text: str, chunks: List[str]) -> List[List[Dict]]:
    """Create batch messages for GPT-4o to generate context for each chunk."""
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that generates succinct context to improve document chunk retrieval.",
    }

    template = (
        "<document>\n\n"
        "{document}\n\n"
        "</document>\n\n"
        "Here is the chunk we want to situate within the whole document:\n\n"
        "<chunk>\n\n"
        "{chunk}\n\n"
        "</chunk>\n\n"
        "Please give a short succinct context to situate this chunk within the overall document "
        "for the purposes of improving search retrieval of the chunk. "
        "Answer only with the succinct context and nothing else."
    )

    messages_list = []
    # Truncate document to 32k chars for context
    doc_truncated = document_text[:32000]

    for chunk in chunks:
        user_content = template.format(document=doc_truncated, chunk=chunk)
        messages_list.append(
            [system_message, {"role": "user", "content": user_content}]
        )

    return messages_list


def submit_batch_request(
    client: OpenAI,
    all_messages: List[List[Dict]],
    model: str = "gpt-4o-2024-08-06",
    max_completion_tokens: int = 300,
) -> str:
    """Submit batch request to OpenAI and return batch_id."""
    requests_list = []
    for idx, messages in enumerate(all_messages):
        requests_list.append(
            {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_completion_tokens,
                    "temperature": 0.1,
                },
            }
        )

    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for req in requests_list:
            f.write(json.dumps(req) + "\n")
        input_file_path = f.name

    # Upload file
    with open(input_file_path, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    os.unlink(input_file_path)

    # Create batch
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    return batch.id


def wait_for_batch(
    client: OpenAI, batch_id: str, poll_interval: int = 60, return_lines: bool = False
):
    """Wait for batch to complete and return results. Optionally return the raw lines as well."""
    print(f"Waiting for batch {batch_id}...")

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status

        if status in ("completed", "failed", "expired", "cancelled"):
            break

        print(f"  Status: {status}")
        time.sleep(poll_interval)

    if status != "completed":
        raise RuntimeError(f"Batch {batch_id} ended with status: {status}")

    # Download results
    file_response = client.files.content(batch.output_file_id)

    results = {}
    batch_lines = []
    for line in file_response.text.strip().split("\n"):
        if not line.strip():
            continue
        data = json.loads(line)
        custom_id = data.get("custom_id")

        # Save raw result line for later reference
        batch_lines.append(data)

        # Check for errors
        if data.get("error"):
            results[custom_id] = ""
            continue

        response = data.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])

        if choices:
            content = choices[0].get("message", {}).get("content", "")
            results[custom_id] = content.strip()
        else:
            results[custom_id] = ""

    if return_lines:
        return results, batch_lines
    return results


def main(
    dataset: str = None,
    input_file: str = None,
    output_file: str = None,
    embedding_model: str = "jinaai/jina-embeddings-v3",
    context_model: str = "gpt-4o-2024-08-06",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    chunk_size: int = 1024,
    text_key: str = "text",
    batch_size: int = 32,
    env_file: str = None,
):
    """
    Create contextual embeddings using GPT-4o for context generation and vLLM for embedding.

    Args:
        dataset: Dataset name ("test" or "val") - auto-sets input/output paths
        input_file: JSON file with documents (overrides dataset)
        output_file: Output JSON file for embeddings (overrides dataset)
        embedding_model: Embedding model name
        context_model: Model for generating context (default: gpt-4o-2024-08-06)
        tensor_parallel_size: Number of GPUs for embedding
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Maximum sequence length for embedding
        chunk_size: Tokens per chunk (default: 1024)
        text_key: Key for text content in documents
        batch_size: Embedding batch size
        env_file: Path to .env file
    """
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    if dataset and not input_file:
        # Read markdown files directly from data/markdown/
        input_file = os.path.join(project_root, "data", "markdown")
    if dataset and not output_file:
        output_file = os.path.join(
            project_root, "data", "embeddings", f"{dataset}_contextual.json"
        )

    # Determine batch API result save path
    if dataset:
        batchapi_file = os.path.join(
            project_root, "data", "embeddings", f"{dataset}_context_batchapi.json"
        )
    elif output_file:
        # Save in the same directory as output_file
        outdir, basename = os.path.split(output_file)
        batchapi_file = os.path.join(outdir, "context_batchapi.json")
    else:
        batchapi_file = None

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
    print(f"Loading tokenizer: {embedding_model}")
    tokenizer = get_tokenizer(embedding_model)

    # Step 1: Chunk all documents
    print(f"\nStep 1: Chunking documents ({chunk_size} tokens per chunk)...")
    all_chunks_data = []  # List of (doc_idx, chunk_idx, chunk_text, metadata)
    all_messages = []  # Messages for batch API

    for doc_idx, doc in enumerate(tqdm(documents, desc="Chunking")):
        doc_text = doc.get(text_key, doc.get("text", ""))
        if not doc_text.strip():
            continue

        metadata = {
            k: v
            for k, v in doc.items()
            if k != text_key and k != "text" and k != "embedding"
        }
        chunks = chunk_text_by_tokens(doc_text, tokenizer, chunk_size)

        # Create batch messages for each chunk
        messages = create_batch_messages(doc_text, chunks)

        for chunk_idx, (chunk, msg) in enumerate(zip(chunks, messages)):
            all_chunks_data.append(
                {
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk,
                    "metadata": metadata,
                }
            )
            all_messages.append(msg)

    print(f"Created {len(all_chunks_data)} chunks from {len(documents)} documents")

    if not all_chunks_data:
        print("No chunks created")
        return

    # Step 2: Generate context using GPT-4o batch API
    print(f"\nStep 2: Generating context with {context_model} batch API...")
    print(f"Submitting {len(all_messages)} requests...")

    batch_id = submit_batch_request(client, all_messages, context_model)
    print(f"Batch ID: {batch_id}")

    results, batch_lines = wait_for_batch(client, batch_id, return_lines=True)
    print(f"Received {len(results)} results")

    # Save batch API raw results for reference
    if batchapi_file:
        os.makedirs(os.path.dirname(batchapi_file) or ".", exist_ok=True)
        with open(batchapi_file, "w", encoding="utf-8") as f:
            json.dump(batch_lines, f, indent=2, ensure_ascii=False)
        print(f"Saved raw batch API results to {batchapi_file}")

    # Step 3: Combine context with chunks
    print("\nStep 3: Combining context with chunks...")
    combined_chunks = []

    for idx, chunk_data in enumerate(all_chunks_data):
        context = results.get(str(idx), "")
        original_chunk = chunk_data["chunk_text"]

        # Combine: context + original chunk
        if context:
            combined_text = f"{context}\n\n{original_chunk}"
        else:
            combined_text = original_chunk

        combined_chunks.append(
            {
                **chunk_data["metadata"],
                "doc_id": chunk_data["doc_idx"],
                "chunk_index": idx,
                "chunk_text": combined_text,
                "original_chunk": original_chunk,
                "context": context,
            }
        )

    # Step 4: Embed combined chunks using vLLM
    print(f"\nStep 4: Embedding {len(combined_chunks)} chunks with vLLM...")
    print(f"  Model: {embedding_model}")
    print(f"  Tensor parallel: {tensor_parallel_size} GPUs")

    llm = LLM(
        model=embedding_model,
        trust_remote_code=True,
        runner="pooling",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    # Prepare texts with prefix for Jina v3
    texts = []
    for chunk in combined_chunks:
        text = chunk["chunk_text"]
        if "jina-embeddings-v3" in embedding_model.lower():
            text = "Represent the document for retrieval: " + text
        texts.append(text)

    # Truncate texts that are too long
    max_tokens = max_model_len - 100
    truncated_count = 0
    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            truncated_count += 1
            texts[i] = tokenizer.decode(
                token_ids[:max_tokens], skip_special_tokens=True
            )

    if truncated_count > 0:
        print(f"Warning: Truncated {truncated_count} chunks to fit max_model_len")

    # Embed in batches
    embedded_chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        batch_chunks = combined_chunks[i : i + batch_size]

        outputs = llm.encode(batch_texts, pooling_task="embed")

        for chunk, output in zip(batch_chunks, outputs):
            output_data = output.outputs.data
            if hasattr(output_data, "cpu"):
                output_data = output_data.cpu()
            embedding = np.asarray(output_data).flatten()

            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding.tolist()

            # Generate stable chunk_id if not present
            if "chunk_id" not in embedded_chunk:
                stable_key = f"{embedded_chunk.get('markdown_file', '')}_{embedded_chunk.get('chunk_index', len(embedded_chunks))}_{embedded_chunk.get('chunk_text', '')[:100]}"
                chunk_id = int(hashlib.md5(stable_key.encode()).hexdigest()[:8], 16)
                embedded_chunk["chunk_id"] = chunk_id

            embedded_chunks.append(embedded_chunk)

    # Save
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(embedded_chunks)} embedded chunks to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
