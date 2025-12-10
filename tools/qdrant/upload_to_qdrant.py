#!/usr/bin/env python3
"""
Upload embeddings to Qdrant.

Usage:
    uv run tools/qdrant/upload_to_qdrant.py --qdrant_url https://... --dataset test --embedding_type late
    uv run tools/qdrant/upload_to_qdrant.py --qdrant_url https://... --qdrant_api_key ... --dataset val --embedding_type contextual

Output:
    data/metadata/test_late_metadata.json
    data/metadata/test_contextual_metadata.json
    data/metadata/val_late_metadata.json
    data/metadata/val_contextual_metadata.json
"""

import json
import os

import fire
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm


def main(
    qdrant_url: str = None,
    dataset: str = None,
    embedding_type: str = None,
    embedded_file: str = None,
    collection_name: str = None,
    metadata_output: str = None,
    qdrant_api_key: str = None,
    vector_size: int = 1024,
    batch_size: int = 100,
    recreate: bool = False,
    env_file: str = None,
):
    """
    Upload embedded chunks to Qdrant.

    Args:
        qdrant_url: Qdrant server URL (or set QDRANT_URL env var)
        dataset: Dataset name ("test" or "val") - auto-sets paths
        embedding_type: Embedding type ("late" or "contextual") - auto-sets paths
        embedded_file: JSON file with embedded chunks (overrides dataset)
        collection_name: Qdrant collection name (auto-generated if using dataset)
        metadata_output: Save metadata to this file (auto-generated if using dataset)
        qdrant_api_key: Qdrant API key (or set QDRANT_API_KEY env var)
        vector_size: Embedding dimension
        batch_size: Upload batch size
        recreate: Recreate collection if exists
        env_file: Path to .env file
    """
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Get qdrant_url from env if not provided
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
    if not qdrant_url:
        print("ERROR: qdrant_url is required. Provide --qdrant_url or set QDRANT_URL in .env file")
        return

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Auto-configure paths based on dataset and embedding_type
    if dataset and embedding_type:
        if not embedded_file:
            embedded_file = os.path.join(
                project_root, "data", "embeddings", f"{dataset}_{embedding_type}.json"
            )
        if not collection_name:
            collection_name = f"{dataset}_{embedding_type}"
        if not metadata_output:
            metadata_output = os.path.join(
                project_root,
                "data",
                "metadata",
                f"{dataset}_{embedding_type}_metadata.json",
            )

    if not embedded_file or not collection_name:
        print(
            "Error: Provide --dataset and --embedding_type, or --embedded_file and --collection_name"
        )
        return

    # Get API key from env if not provided
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

    print(f"Loading embeddings from: {embedded_file}")
    with open(embedded_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # Auto-detect vector size from first embedding
    if chunks:
        first_embedding = chunks[0].get("embedding", [])
        if first_embedding:
            detected_vector_size = len(first_embedding)
            if vector_size != detected_vector_size:
                print(
                    f"Auto-detected vector size: {detected_vector_size} (overriding {vector_size})"
                )
                vector_size = detected_vector_size
        else:
            print(
                f"Warning: First chunk has no embedding, using default vector_size={vector_size}"
            )
    else:
        print("Error: No chunks to upload")
        return

    # Connect to Qdrant
    print(f"Connecting to Qdrant at: {qdrant_url}")
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)

    # Check/create collection
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        if recreate:
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            print(f"Using existing collection: {collection_name}")

    if collection_name not in collections or recreate:
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    # Prepare points with stable IDs
    import hashlib

    points = []
    for idx, chunk in enumerate(chunks):
        embedding = chunk.get("embedding", [])
        if not embedding:
            continue

        payload = {k: v for k, v in chunk.items() if k != "embedding"}

        # Generate stable chunk_id if not present
        if "chunk_id" not in payload:
            # Create stable ID from content + metadata
            stable_key = f"{payload.get('markdown_file', '')}_{payload.get('chunk_index', idx)}_{payload.get('start_token_idx', '')}_{payload.get('end_token_idx', '')}"
            chunk_id = int(hashlib.md5(stable_key.encode()).hexdigest()[:8], 16)
            payload["chunk_id"] = chunk_id
        else:
            chunk_id = payload["chunk_id"]

        # Use chunk_id as point ID (stable across re-uploads)
        point = PointStruct(
            id=chunk_id,
            vector=embedding,
            payload=payload,
        )
        points.append(point)

    # Upload
    print(f"Uploading {len(points)} points to collection '{collection_name}'...")
    for i in tqdm(range(0, len(points), batch_size), desc="Uploading"):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    # Get collection info
    info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' now has {info.points_count} points")

    # Save metadata
    metadata = {
        "collection_name": collection_name,
        "qdrant_url": qdrant_url,
        "points_count": info.points_count,
        "vector_size": vector_size,
        "dataset": dataset,
        "embedding_type": embedding_type,
        "source_file": os.path.basename(embedded_file),
    }

    if metadata_output:
        os.makedirs(os.path.dirname(metadata_output) or ".", exist_ok=True)
        with open(metadata_output, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_output}")


if __name__ == "__main__":
    fire.Fire(main)
