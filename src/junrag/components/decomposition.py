"""
Query Decomposition component for JunRAG.
Uses GPT-4o to decompose multi-hop queries into single-hop sub-queries.
"""

import json
import logging
import os
from typing import List, Optional, Union

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
)

from junrag.models import DecompositionConfig

# Configure logging
logger = logging.getLogger(__name__)


DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition expert. Your task is to break down complex multi-hop questions into a list of simple, independent search queries.

**CRITICAL RULES:**
1. **NO GUESSING:** Never hallucinate or guess specific names, dates, or entities that are not explicitly in the prompt (e.g., do not guess "La La Land" if the prompt only says "the movie").
2. **INDEPENDENCE:** Each sub-query must be answerable entirely on its own. Do NOT refer to "step 1" or "the previous answer."
3. **DESCRIPTIVE REDUNDANCY:** Because you cannot reference previous answers, you must repeat the descriptive conditions in later queries.
   - *Wrong:* "Who directed it?" (Depends on previous step)
   - *Right:* "Who directed the movie that won Best Picture in 1989?" (Self-contained)
4. **COVERAGE:** The sub-queries must cover all facts required to derive the final answer.

**EXAMPLES:**

Input: "Who was older, the guitar player for the Dugites from 1982-1983 or the lead singer of The Sports?"
Output: {{"sub_queries": ["Who was the guitar player for the Dugites from 1982-1983?", "When was the guitar player for the Dugites from 1982-1983 born?", "Who was the lead singer of The Sports?", "When was the lead singer of The Sports born?"]}}

Input: "In the first movie that Emma Stone won an Academy Award for Best Actress in, did her costar win an Academy Award for Best Actor?"
Output: {{"sub_queries": ["What is the first movie that Emma Stone won an Academy Award for Best Actress for?", "Who was Emma Stone's costar in the first movie she won an Academy Award for Best Actress for?", "Did the costar in the first movie Emma Stone won an Academy Award for Best Actress for win an Academy Award for Best Actor?"]}}

Input: "As of August 4, 2024, what is the first initial and surname of the cricketer who became the top-rated test batsman in the 2020s, is the fastest player of their country to 6 1000 run milestones in tests, and became their country's all-time leading run scorer in tests in the same year?"
Output: {{"sub_queries": ["Which cricketer became the top-rated test batsman in the 2020s?", "Which cricketer is the fastest player of their country to reach 6 1000 run milestones in tests?", "Which cricketer became their country's all-time leading run scorer in tests in the 2020s?"]}}

You must return a JSON object with a "sub_queries" key containing an array of strings."""


class QueryDecomposer:
    """Decomposes complex multi-hop queries into single-hop sub-queries."""

    def __init__(
        self,
        model: Union[str, DecompositionConfig] = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        reasoning_effort: str = "medium",
    ):
        """
        Initialize query decomposer.

        Args:
            model: OpenAI model name or DecompositionConfig instance
            api_key: OpenAI API key
            temperature: Generation temperature (low for consistency, ignored for reasoning models)
            max_tokens: Maximum tokens in response
            reasoning_effort: Reasoning effort for reasoning models (low, medium, high) - default: low
        """
        # Handle Pydantic config or individual parameters
        if isinstance(model, DecompositionConfig):
            config = model
            self.model = config.model
            api_key = config.api_key or api_key
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
            # Get reasoning_effort from config if available, otherwise use default
            self.reasoning_effort = getattr(config, "reasoning_effort", "medium")
        else:
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.reasoning_effort = reasoning_effort

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required for query decomposition. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Check if this is a non-reasoning model (default is reasoning model)
        # Only gpt-4 models are explicitly non-reasoning
        self.is_reasoning_model = "gpt-4" not in self.model.lower()

        try:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"QueryDecomposer initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def decompose(self, query: str, retry_count: int = 2) -> List[str]:
        """
        Decompose a complex query into single-hop sub-queries.

        Args:
            query: The complex multi-hop query
            retry_count: Number of retries on failure

        Returns:
            List of single-hop sub-queries (returns original query on failure)
        """
        # Validate input
        if not query or not isinstance(query, str):
            logger.warning("Empty or invalid query provided to decomposer")
            return [query] if query else []

        query = query.strip()
        if not query:
            logger.warning("Query is empty after stripping whitespace")
            return []

        # Use system/user messages and request JSON object format
        messages = [
            {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Decompose this query:\n{query}"},
        ]

        last_error = None
        for attempt in range(retry_count + 1):
            try:
                # Build API call parameters based on model type
                if self.is_reasoning_model:
                    # Reasoning models use max_completion_tokens (OpenAI SDK)
                    # Try with response_format first, fallback if not supported
                    max_completion_tokens = min(self.max_tokens * (2**attempt), 8192)
                    reasoning_effort = self.reasoning_effort if attempt == 0 else "low"
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_completion_tokens=max_completion_tokens,
                            reasoning_effort=reasoning_effort,
                            response_format={"type": "json_object"},
                        )
                    except (TypeError, ValueError) as e:
                        # If response_format is not supported, try without it
                        if "response_format" in str(e) or "json_object" in str(e):
                            logger.debug(
                                f"response_format not supported, retrying without it"
                            )
                            try:
                                response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=messages,
                                    max_completion_tokens=max_completion_tokens,
                                    reasoning_effort=reasoning_effort,
                                )
                            except TypeError as e2:
                                # Fallback to max_tokens if max_completion_tokens not supported
                                if "max_completion_tokens" in str(e2):
                                    logger.debug(
                                        f"max_completion_tokens not supported, trying max_tokens"
                                    )
                                    response = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        max_tokens=max_completion_tokens,
                                        reasoning_effort=reasoning_effort,
                                    )
                                else:
                                    raise
                        elif "reasoning_effort" in str(e):
                            logger.debug(
                                f"reasoning_effort not supported, retrying without it"
                            )
                            try:
                                response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=messages,
                                    max_completion_tokens=max_completion_tokens,
                                    response_format={"type": "json_object"},
                                )
                            except (TypeError, ValueError) as e2:
                                if "response_format" in str(e2) or "json_object" in str(
                                    e2
                                ):
                                    response = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        max_completion_tokens=max_completion_tokens,
                                    )
                                elif "max_completion_tokens" in str(e2):
                                    response = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        max_tokens=max_completion_tokens,
                                    )
                                else:
                                    raise
                        else:
                            raise
                    except TypeError as e:
                        # Handle case where reasoning_effort is not supported
                        if "reasoning_effort" in str(e):
                            logger.debug(
                                f"reasoning_effort not supported, retrying without it"
                            )
                            try:
                                response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=messages,
                                    max_completion_tokens=max_completion_tokens,
                                    response_format={"type": "json_object"},
                                )
                            except TypeError as e2:
                                # Fallback to max_tokens if max_completion_tokens not supported
                                if "max_completion_tokens" in str(e2):
                                    logger.debug(
                                        f"max_completion_tokens not supported, trying max_tokens"
                                    )
                                    response = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        max_tokens=max_completion_tokens,
                                        response_format={"type": "json_object"},
                                    )
                                else:
                                    raise
                        # Fallback to max_tokens if max_completion_tokens not supported
                        elif "max_completion_tokens" in str(e):
                            logger.debug(
                                f"max_completion_tokens not supported, trying max_tokens"
                            )
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                max_tokens=max_completion_tokens,
                                reasoning_effort=reasoning_effort,
                                response_format={"type": "json_object"},
                            )
                        else:
                            raise
                else:
                    # Standard models (gpt-4o, etc.) use max_tokens
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            response_format={"type": "json_object"},
                        )
                    except (TypeError, ValueError) as e:
                        # If response_format is not supported, try without it
                        if "response_format" in str(e) or "json_object" in str(e):
                            logger.debug(
                                f"response_format not supported, retrying without it"
                            )
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                            )
                        else:
                            raise

                if not response.choices:
                    logger.warning("Decomposition API returned empty choices")
                    return [query]

                choice = response.choices[0]
                content = choice.message.content
                if content is None:
                    logger.warning("Decomposition API returned None content")
                    return [query]

                if not content or not content.strip():
                    logger.warning(
                        f"Decomposition API returned empty content. Response: {response}"
                    )
                    if (
                        self.is_reasoning_model
                        and choice.finish_reason == "length"
                        and attempt < retry_count
                    ):
                        logger.warning(
                            "Empty output with finish_reason=length. Retrying with larger "
                            "max_completion_tokens and lower reasoning_effort..."
                        )
                        last_error = ValueError("Empty output (finish_reason=length)")
                        continue
                    return [query]

                content = content.strip()
                logger.debug(
                    f"Received content from API (first 500 chars): {content[:500]}"
                )

                # Parse JSON object
                try:
                    result = json.loads(content)
                    # Extract sub_queries from JSON object
                    if isinstance(result, dict) and "sub_queries" in result:
                        sub_queries = result["sub_queries"]
                    elif isinstance(result, list):
                        # Fallback: if it's already a list, use it directly
                        sub_queries = result
                    else:
                        logger.warning(
                            f"Decomposition returned unexpected format: {type(result)}"
                        )
                        logger.debug(f"Content: {content[:500]}")
                        return [query]
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse decomposition JSON: {e}")
                    logger.debug(f"Raw content (first 500 chars): {content[:500]}")
                    logger.debug(f"Raw content (full): {content}")

                    # If output was truncated, retry with a larger token budget.
                    if choice.finish_reason == "length" and attempt < retry_count:
                        logger.warning(
                            "JSON parse failed with finish_reason=length. Retrying with a larger "
                            "max_completion_tokens..."
                        )
                        last_error = e
                        continue
                    # Try to extract JSON from markdown code blocks
                    if "```json" in content:
                        start = content.find("```json") + 7
                        end = content.find("```", start)
                        if end != -1:
                            try:
                                result = json.loads(content[start:end].strip())
                                sub_queries = (
                                    result.get("sub_queries", result)
                                    if isinstance(result, dict)
                                    else result
                                )
                            except json.JSONDecodeError:
                                return [query]
                        else:
                            return [query]
                    elif "```" in content:
                        # Try to extract from markdown code blocks
                        lines = content.split("\n")
                        json_lines = [
                            line for line in lines if not line.startswith("```")
                        ]
                        try:
                            result = json.loads("\n".join(json_lines))
                            sub_queries = (
                                result.get("sub_queries", result)
                                if isinstance(result, dict)
                                else result
                            )
                        except json.JSONDecodeError:
                            return [query]
                    else:
                        return [query]

                if not isinstance(sub_queries, list):
                    logger.warning(
                        f"Decomposition returned non-list: {type(sub_queries)}"
                    )
                    return [query]

                # Filter out empty strings and validate
                sub_queries = [
                    q.strip() for q in sub_queries if isinstance(q, str) and q.strip()
                ]

                if not sub_queries:
                    logger.debug("No valid sub-queries after filtering")
                    return [query]

                # Validate count (prompt says max 4, but model might return more)
                max_expected = 4
                max_reasonable = 10  # Hard limit to prevent excessive processing

                if len(sub_queries) > max_reasonable:
                    logger.error(
                        f"Decomposition returned {len(sub_queries)} sub-queries, "
                        f"exceeding reasonable maximum of {max_reasonable}. "
                        "Truncating to prevent excessive processing."
                    )
                    sub_queries = sub_queries[:max_reasonable]
                    logger.info(f"Truncated to {max_reasonable} sub-queries")
                elif len(sub_queries) > max_expected:
                    logger.warning(
                        f"Decomposition returned {len(sub_queries)} sub-queries, "
                        f"exceeding expected maximum of {max_expected}. "
                        "This may indicate an overly complex query. Processing will continue, "
                        "but consider simplifying the query for better results."
                    )

                logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
                return sub_queries

            except AuthenticationError as e:
                logger.error(f"OpenAI authentication failed: {e}")
                raise ValueError(
                    "OpenAI API authentication failed. Check your API key."
                ) from e

            except RateLimitError as e:
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{retry_count + 1}): {e}"
                )
                last_error = e
                if attempt < retry_count:
                    import time

                    wait_time = 2**attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                # Fall back to original query
                logger.warning("Rate limit exceeded. Returning original query.")
                return [query]

            except APIConnectionError as e:
                logger.warning(f"API connection error (attempt {attempt + 1}): {e}")
                last_error = e
                if attempt < retry_count:
                    continue
                logger.warning("Connection failed. Returning original query.")
                return [query]

            except APIError as e:
                logger.warning(f"OpenAI API error: {e}")
                if "model" in str(e).lower() and "not found" in str(e).lower():
                    logger.error(f"Model '{self.model}' not found. Check model name.")
                    raise ValueError(f"Model '{self.model}' not found") from e
                last_error = e
                if attempt < retry_count:
                    continue
                return [query]

            except Exception as e:
                logger.warning(f"Query decomposition failed: {type(e).__name__}: {e}")
                last_error = e
                if attempt < retry_count:
                    continue
                # Return original query if decomposition fails
                return [query]

        # Should not reach here, but just in case
        logger.warning(
            f"Decomposition failed after all retries. Last error: {last_error}"
        )
        return [query]

    def decompose_batch(self, queries: List[str]) -> List[List[str]]:
        """
        Decompose multiple queries.

        Args:
            queries: List of queries to decompose

        Returns:
            List of lists of sub-queries
        """
        if not queries:
            logger.warning("Empty queries list provided to decompose_batch")
            return []

        results = []
        failed_count = 0

        for i, q in enumerate(queries):
            try:
                sub_queries = self.decompose(q)
                results.append(sub_queries)
            except Exception as e:
                logger.warning(f"Failed to decompose query {i}: {e}")
                results.append([q])  # Fall back to original query
                failed_count += 1

        if failed_count > 0:
            logger.warning(
                f"Decomposition failed for {failed_count}/{len(queries)} queries"
            )

        return results


# Singleton for convenience
_decomposer: Optional[QueryDecomposer] = None


def get_decomposer(
    config: Optional[DecompositionConfig] = None, **kwargs
) -> QueryDecomposer:
    """Get or create decomposer singleton."""
    global _decomposer
    if _decomposer is None:
        if config is not None:
            _decomposer = QueryDecomposer(config)
        else:
            _decomposer = QueryDecomposer(**kwargs)
    return _decomposer


def decompose_query(
    query: str,
    decomposer: Optional[QueryDecomposer] = None,
    config: Optional[DecompositionConfig] = None,
    **kwargs,
) -> List[str]:
    """
    Convenience function to decompose a query.

    Args:
        query: The query to decompose
        decomposer: Optional pre-initialized decomposer
        config: Optional DecompositionConfig instance
        **kwargs: Arguments for QueryDecomposer if creating new

    Returns:
        List of sub-queries
    """
    if decomposer is None:
        if config is not None:
            decomposer = get_decomposer(config=config)
        else:
            decomposer = get_decomposer(**kwargs)
    return decomposer.decompose(query)
