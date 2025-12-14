"""
Retrieval component for JunRAG.
Handles vector similarity search from Qdrant.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Union

import numpy as np

from junrag.models import Chunk, ChunkMetadata, RetrievalConfig

# Configure logging
logger = logging.getLogger(__name__)


def get_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> "QdrantClient":
    """
    Get Qdrant client instance.

    Args:
        url: Qdrant server URL
        api_key: Qdrant API key
        timeout: Request timeout in seconds (default: 60.0)

    Returns:
        QdrantClient instance

    Raises:
        ImportError: If qdrant-client is not installed
        ConnectionError: If connection to Qdrant fails
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        logger.error("qdrant-client not installed")
        raise ImportError(
            "qdrant-client is required. Install with: pip install qdrant-client"
        ) from e

    url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = api_key or os.getenv("QDRANT_API_KEY")
    timeout = timeout or float(os.getenv("QDRANT_TIMEOUT", "60.0"))

    logger.debug(f"Connecting to Qdrant at {url} (timeout: {timeout}s)")

    try:
        if api_key:
            client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        else:
            client = QdrantClient(url=url, timeout=timeout)

        # Test connection (optional - comment out if causing issues)
        # client.get_collections()

        return client
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            logger.error(f"Failed to connect to Qdrant at {url}: {e}")
            raise ConnectionError(
                f"Cannot connect to Qdrant at {url}. Is the server running?"
            ) from e
        elif "unauthorized" in error_msg or "authentication" in error_msg:
            logger.error(f"Qdrant authentication failed: {e}")
            raise ValueError("Qdrant authentication failed. Check your API key.") from e
        else:
            logger.error(f"Failed to create Qdrant client: {e}")
            raise


def load_metadata(metadata_path: str) -> Dict:
    """
    Load collection metadata from JSON file.

    Args:
        metadata_path: Path to metadata JSON file

    Returns:
        Metadata dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid JSON
    """
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.debug(f"Loaded metadata from {metadata_path}")
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file: {e}")
        raise ValueError(f"Invalid JSON in metadata file: {metadata_path}") from e
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise


def retrieve_chunks(
    query_embedding: np.ndarray,
    collection_name: Optional[str] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    top_k: int = 10,
    score_threshold: Optional[float] = None,
    filter_conditions: Optional[Dict] = None,
    config: Optional[RetrievalConfig] = None,
    timeout: Optional[float] = None,
    retry_count: int = 2,
) -> List[Chunk]:
    """
    Retrieve chunks from Qdrant collection.

    Args:
        query_embedding: Query embedding vector
        collection_name: Name of the Qdrant collection
        url: Qdrant server URL
        api_key: Qdrant API key
        top_k: Number of results to return
        score_threshold: Minimum score threshold
        filter_conditions: Filter conditions dict
        config: Optional RetrievalConfig instance
        timeout: Request timeout in seconds (default: 60.0)
        retry_count: Number of retries on timeout/connection errors (default: 2)

    Returns:
        List of retrieved chunks with scores and payloads

    Raises:
        ValueError: If inputs are invalid
        ConnectionError: If connection to Qdrant fails
        RuntimeError: If retrieval fails after retries
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    # Handle Pydantic config or individual parameters
    if config is not None:
        collection_name = config.collection_name
        url = config.url or url
        api_key = config.api_key or api_key
        top_k = config.top_k
        score_threshold = config.score_threshold
        filter_conditions = config.filter_conditions or filter_conditions

    # Validate inputs
    if query_embedding is None:
        raise ValueError("query_embedding cannot be None")

    if not isinstance(query_embedding, np.ndarray):
        try:
            query_embedding = np.asarray(query_embedding)
        except Exception as e:
            raise ValueError(f"Invalid query_embedding: {e}") from e

    if query_embedding.size == 0:
        raise ValueError("query_embedding cannot be empty")

    if not collection_name or not collection_name.strip():
        raise ValueError("collection_name is required")

    if top_k <= 0:
        logger.warning(f"Invalid top_k={top_k}. Using 10.")
        top_k = 10

    # Get client with error handling
    try:
        client = get_qdrant_client(url, api_key)
    except Exception as e:
        logger.error(f"Failed to get Qdrant client: {e}")
        raise

    # Build filter
    query_filter = None
    if filter_conditions:
        try:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            query_filter = Filter(must=must_conditions)
        except Exception as e:
            logger.warning(
                f"Failed to build filter conditions: {e}. Proceeding without filter."
            )
            query_filter = None

    # Query
    try:
        query_vector = query_embedding.flatten().tolist()
    except Exception as e:
        logger.error(f"Failed to convert embedding to list: {e}")
        raise ValueError(f"Invalid embedding format: {e}") from e

    # Retry logic for timeout and connection errors
    last_exception = None
    for attempt in range(retry_count + 1):
        try:
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    score_threshold=score_threshold,
                )
                results = response.points
            else:
                results = client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                )
            # Success - break out of retry loop
            break
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            error_type = type(e).__name__

            # Check for timeout errors
            is_timeout = (
                "timeout" in error_msg
                or "timed out" in error_msg
                or "ResponseHandlingException" in error_type
            )

            # Check for connection errors
            is_connection_error = (
                "connection" in error_msg
                or "refused" in error_msg
                or "ConnectionError" in error_type
            )

            # Non-retryable errors
            if "not found" in error_msg or "doesn't exist" in error_msg:
                logger.error(f"Collection '{collection_name}' not found in Qdrant")
                raise ValueError(
                    f"Collection '{collection_name}' not found. "
                    "Check collection name or create the collection first."
                ) from e
            elif "dimension" in error_msg:
                logger.error(f"Embedding dimension mismatch: {e}")
                raise ValueError(
                    f"Embedding dimension mismatch. Query embedding has {len(query_vector)} dimensions. "
                    "Check that embedding model matches collection."
                ) from e

            # Retryable errors (timeout, connection)
            if (is_timeout or is_connection_error) and attempt < retry_count:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s...
                logger.warning(
                    f"Qdrant query failed (attempt {attempt + 1}/{retry_count + 1}): {error_type}: {e}"
                )
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
                # Recreate client for connection errors
                if is_connection_error:
                    try:
                        client = get_qdrant_client(url, api_key, timeout=timeout)
                    except Exception as client_error:
                        logger.warning(f"Failed to recreate client: {client_error}")
                continue

            # If we get here, it's either not retryable or we've exhausted retries
            if is_timeout:
                logger.error(
                    f"Qdrant query timed out after {retry_count + 1} attempts: {e}"
                )
                raise RuntimeError(
                    f"Qdrant query timed out after {retry_count + 1} attempts. "
                    "The server may be overloaded or the query is too complex. "
                    f"Original error: {e}"
                ) from e
            elif is_connection_error:
                logger.error(
                    f"Lost connection to Qdrant after {retry_count + 1} attempts: {e}"
                )
                raise ConnectionError(
                    f"Lost connection to Qdrant after {retry_count + 1} attempts: {e}"
                ) from e
            else:
                # Other errors - don't retry
                logger.error(f"Qdrant query failed: {e}")
                raise RuntimeError(f"Retrieval failed: {e}") from e

    # If we exhausted retries without success
    if last_exception is not None:
        raise RuntimeError(
            f"Retrieval failed after {retry_count + 1} attempts: {last_exception}"
        ) from last_exception

    if not results:
        logger.warning(
            f"No results found in collection '{collection_name}'. "
            f"Query embedding dim: {len(query_vector)}, top_k: {top_k}"
        )
        return []

    # Format results
    formatted_results = []
    missing_text_count = 0

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}

        # Extract text from various possible fields
        text = (
            payload.get("chunk_text")
            or payload.get("text")
            or payload.get("content")
            or ""
        )

        if not text:
            missing_text_count += 1

        metadata = ChunkMetadata(
            dataset=payload.get("dataset"),
            item_id=payload.get("item_id"),
            source=payload.get("source"),
            markdown_file=payload.get("markdown_file"),
        )

        chunk = Chunk(
            rank=rank,
            chunk_id=str(result.id) if result.id is not None else None,
            retrieval_score=(float(result.score) if result.score is not None else 0.0),
            text=text,
            metadata=metadata,
            payload=payload,
        )

        formatted_results.append(chunk)

    if missing_text_count > 0:
        logger.warning(
            f"{missing_text_count}/{len(formatted_results)} retrieved chunks have no text. "
            "Check payload structure (expected: chunk_text, text, or content)."
        )

    logger.debug(f"Retrieved {len(formatted_results)} chunks from '{collection_name}'")
    return formatted_results
