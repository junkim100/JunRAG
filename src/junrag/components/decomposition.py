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


DECOMPOSITION_PROMPT = """
You are a query decomposition expert. Your task is to break down complex multi-hop questions into a list of simple, independent search queries.

**CRITICAL RULES:**
1. **NO GUESSING:** Never hallucinate or guess specific names, dates, or entities that are not explicitly in the prompt (e.g., do not guess "La La Land" if the prompt only says "the movie").
2. **INDEPENDENCE:** Each sub-query must be answerable entirely on its own. Do NOT refer to "step 1" or "the previous answer."
3. **DESCRIPTIVE REDUNDANCY:** Because you cannot reference previous answers, you must repeat the descriptive conditions in later queries.
   - *Wrong:* "Who directed it?" (Depends on previous step)
   - *Right:* "Who directed the movie that won Best Picture in 1989?" (Self-contained)
4. **COVERAGE:** The sub-queries must cover all facts required to derive the final answer.
5. **MAX SUB-QUERIES:** There should be no more than 4 sub-queries.

**EXAMPLES:**

Input: "Who was older, the guitar player for the Dugites from 1982-1983 or the lead singer of The Sports?"
Output: [
  "Who was the guitar player for the Dugites from 1982-1983?",
  "When was the guitar player for the Dugites from 1982-1983 born?",
  "Who was the lead singer of The Sports?",
  "When was the lead singer of The Sports born?"
]

Input: "In the first movie that Emma Stone won an Academy Award for Best Actress in, did her costar win an Academy Award for Best Actor?"
Output: [
  "What is the first movie that Emma Stone won an Academy Award for Best Actress for?",
  "Who was Emma Stone's costar in the first movie she won an Academy Award for Best Actress for?",
  "Did the costar in the first movie Emma Stone won an Academy Award for Best Actress for win an Academy Award for Best Actor?"
]

Input: "As of August 4, 2024, what is the first initial and surname of the cricketer who became the top-rated test batsman in the 2020s, is the fastest player of their country to 6 1000 run milestones in tests, and became their country's all-time leading run scorer in tests in the same year?"
Output: [
  "Which cricketer became the top-rated test batsman in the 2020s?",
  "Which cricketer is the fastest player of their country to reach 6 1000 run milestones in tests?",
  "Which cricketer became their country's all-time leading run scorer in tests in the 2020s?"
]

Now decompose this query:
{query}

Return ONLY the JSON array:"""


class QueryDecomposer:
    """Decomposes complex multi-hop queries into single-hop sub-queries."""

    def __init__(
        self,
        model: Union[str, DecompositionConfig] = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        """
        Initialize query decomposer.

        Args:
            model: OpenAI model name or DecompositionConfig instance
            api_key: OpenAI API key
            temperature: Generation temperature (low for consistency)
            max_tokens: Maximum tokens in response
        """
        # Handle Pydantic config or individual parameters
        if isinstance(model, DecompositionConfig):
            config = model
            self.model = config.model
            api_key = config.api_key or api_key
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
        else:
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required for query decomposition. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

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

        prompt = DECOMPOSITION_PROMPT.format(query=query)

        last_error = None
        for attempt in range(retry_count + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if not response.choices:
                    logger.warning("Decomposition API returned empty choices")
                    return [query]

                content = response.choices[0].message.content
                if content is None:
                    logger.warning("Decomposition API returned None content")
                    return [query]

                content = content.strip()

                # Parse JSON array
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(
                        line for line in lines if not line.startswith("```")
                    )

                try:
                    sub_queries = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse decomposition JSON: {e}")
                    logger.debug(f"Raw content: {content[:200]}...")
                    # Try to extract from malformed response
                    if "[" in content and "]" in content:
                        try:
                            start = content.index("[")
                            end = content.rindex("]") + 1
                            sub_queries = json.loads(content[start:end])
                        except (json.JSONDecodeError, ValueError):
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
