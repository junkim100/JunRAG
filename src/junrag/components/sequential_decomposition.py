"""
Sequential Query Decomposition component for JunRAG.
Uses GPT-4o to decompose multi-hop queries into sequential sub-queries with placeholders.

Unlike standard decomposition where queries are independent, sequential decomposition
creates a chain where each subquery (except the first) contains a [answer] placeholder
that references the answer from the previous subquery.

Example:
    Query: "Who was older, the guitar player for the Dugites from 1982-1983 or the lead singer of The Sports?"

    Sequential subqueries:
    1. "Who was the guitar player for the Dugites from 1982-1983?"
    2. "When was [answer] born?"  # [answer] = answer to query 1
    3. "Who was the lead singer of The Sports?"
    4. "When was [answer] born?"  # [answer] = answer to query 3

    Note: Only immediate previous answer is referenced, not answers from earlier.
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


SEQUENTIAL_DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition expert. Your task is to break down complex multi-hop questions into a sequence of simple search queries where later queries depend on earlier answers.

**CRITICAL RULES:**
1. **NO GUESSING:** Never hallucinate or guess specific names, dates, or entities that are not explicitly in the prompt.
2. **PLACEHOLDER SYNTAX:** Use exactly `[answer]` (lowercase, with brackets) as a placeholder for the previous query's answer.
3. **FIRST QUERY:** The first sub-query must NOT contain any placeholder. It must be fully self-contained.
4. **SINGLE DEPENDENCY:** Each subsequent sub-query may contain `[answer]` to refer ONLY to the immediately previous sub-query's answer. Do NOT reference answers from earlier queries.
5. **SEQUENTIAL LOGIC:** Design queries so that answering them in order leads to the final answer.
6. **COVERAGE:** The sub-queries must cover all facts required to derive the final answer.

**EXAMPLES:**

Input: "Who was older, the guitar player for the Dugites from 1982-1983 or the lead singer of The Sports?"
Output: {{"sub_queries": ["Who was the guitar player for the Dugites from 1982-1983?", "When was [answer] born?", "Who was the lead singer of The Sports?", "When was [answer] born?"]}}

Input: "In the first movie that Emma Stone won an Academy Award for Best Actress in, did her costar win an Academy Award for Best Actor?"
Output: {{"sub_queries": ["What is the first movie that Emma Stone won an Academy Award for Best Actress for?", "Who was Emma Stone's costar in [answer]?", "Did [answer] win an Academy Award for Best Actor?"]}}

Input: "What is the capital of the country where the author of '1984' was born?"
Output: {{"sub_queries": ["Who is the author of '1984'?", "In which country was [answer] born?", "What is the capital of [answer]?"]}}

Input: "As of August 4, 2024, what is the first initial and surname of the cricketer who became the top-rated test batsman in the 2020s, is the fastest player of their country to 6 1000 run milestones in tests, and became their country's all-time leading run scorer in tests in the same year?"
Output: {{"sub_queries": ["Which cricketer became the top-rated test batsman in the 2020s and is the fastest player of their country to 6 1000 run milestones in tests?", "Is [answer] also their country's all-time leading run scorer in tests?", "What is the first initial and surname of [answer]?"]}}

You must return a JSON object with a "sub_queries" key containing an array of strings."""


# Prompt for replacing [answer] placeholder with actual answer from retrieval
REWRITE_SUBQUERY_PROMPT = """You are given a sub-query that contains the placeholder [answer].
Based on the context provided from a previous retrieval, replace [answer] with the actual answer.

Previous sub-query: {previous_subquery}

Context from retrieval for the previous sub-query:
{context}

Current sub-query with placeholder: {current_subquery}

Your task:
1. Identify what [answer] should be based on the context
2. Return the current sub-query with [answer] replaced by the actual answer

Return ONLY the rewritten sub-query (no explanation, no quotes around it):"""


# Prompt for generating final answer
FINAL_ANSWER_PROMPT = """You are an expert AI assistant that answers questions strictly based on the provided context.

<original_query>
{original_query}
</original_query>

<final_subquery>
{final_subquery}
</final_subquery>

<context>
{context}
</context>

<chain_of_answers>
Here is the chain of sub-queries and their answers that led to this point:
{chain_summary}
</chain_of_answers>

Based on the context and the chain of answers above, provide the final answer to the original query.

IMPORTANT:
1. Use ONLY the information from the context and chain of answers
2. Be concise and direct
3. Single keyword or short phrase answers are preferred when appropriate

Answer:"""


class SequentialQueryDecomposer:
    """Decomposes complex multi-hop queries into sequential sub-queries with placeholders."""

    def __init__(
        self,
        model: Union[str, DecompositionConfig] = "gpt-5-mini-2025-08-07",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        reasoning_effort: str = "medium",
    ):
        """
        Initialize sequential query decomposer.

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
            logger.info(f"SequentialQueryDecomposer initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def decompose(self, query: str, retry_count: int = 2) -> List[str]:
        """
        Decompose a complex query into sequential sub-queries with placeholders.

        Args:
            query: The complex multi-hop query
            retry_count: Number of retries on failure

        Returns:
            List of sequential sub-queries (first has no placeholder, others may have [answer])
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
            {"role": "system", "content": SEQUENTIAL_DECOMPOSITION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Decompose this query into sequential sub-queries:\n{query}",
            },
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
                                        "max_completion_tokens not supported, trying max_tokens"
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
                    # For reasoning models, empty output + finish_reason=length often means
                    # the model used the entire token budget for internal reasoning.
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
                                logger.debug(
                                    "Failed to parse JSON from markdown code block"
                                )
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
                            logger.debug("Failed to parse JSON from markdown")
                            return [query]
                    # Try to find JSON object in the content
                    elif "{" in content and "sub_queries" in content:
                        try:
                            # Try to find the JSON object
                            start = content.find("{")
                            end = content.rfind("}") + 1
                            if start != -1 and end > start:
                                result = json.loads(content[start:end])
                                sub_queries = result.get("sub_queries", [])
                            else:
                                return [query]
                        except json.JSONDecodeError:
                            return [query]
                    else:
                        logger.warning(
                            f"Could not extract JSON from response. Content: {content[:200]}"
                        )
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

                # Validate: first query should not have placeholder
                if "[answer]" in sub_queries[0].lower():
                    logger.warning(
                        "First sub-query contains [answer] placeholder. "
                        "This violates sequential decomposition rules."
                    )
                    # Try to fix by removing the placeholder
                    sub_queries[0] = (
                        sub_queries[0]
                        .replace("[answer]", "")
                        .replace("[Answer]", "")
                        .strip()
                    )

                logger.info(
                    f"Decomposed query into {len(sub_queries)} sequential sub-queries"
                )
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
                return [query]

        logger.warning(
            f"Decomposition failed after all retries. Last error: {last_error}"
        )
        return [query]

    def rewrite_subquery(
        self,
        previous_subquery: str,
        current_subquery: str,
        context: str,
        retry_count: int = 2,
    ) -> str:
        """
        Rewrite a sub-query by replacing [answer] placeholder with actual answer.

        Args:
            previous_subquery: The sub-query that was just answered
            current_subquery: The sub-query containing [answer] placeholder
            context: Context from retrieval for the previous sub-query
            retry_count: Number of retries on failure

        Returns:
            Rewritten sub-query with [answer] replaced
        """
        if "[answer]" not in current_subquery.lower():
            logger.debug("No [answer] placeholder found, returning original subquery")
            return current_subquery

        prompt = REWRITE_SUBQUERY_PROMPT.format(
            previous_subquery=previous_subquery,
            context=context,
            current_subquery=current_subquery,
        )

        last_error = None
        for attempt in range(retry_count + 1):
            try:
                # Build API call parameters based on model type
                if self.is_reasoning_model:
                    # Reasoning models use max_completion_tokens (OpenAI SDK)
                    max_completion_tokens = min(200 * (2**attempt), 2048)
                    reasoning_effort = self.reasoning_effort if attempt == 0 else "low"
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=max_completion_tokens,
                            reasoning_effort=reasoning_effort,
                        )
                    except TypeError as e:
                        # Handle case where reasoning_effort is not supported
                        if "reasoning_effort" in str(e):
                            try:
                                response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=[{"role": "user", "content": prompt}],
                                    max_completion_tokens=max_completion_tokens,
                                )
                            except TypeError as e2:
                                if "max_completion_tokens" in str(e2):
                                    response = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=max_completion_tokens,
                                    )
                                else:
                                    raise
                        # Fallback to max_tokens if max_completion_tokens not supported
                        elif "max_completion_tokens" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_completion_tokens,
                                reasoning_effort=reasoning_effort,
                            )
                        else:
                            raise
                else:
                    # Standard models (gpt-4o, etc.) use max_tokens
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=200,
                    )

                if not response.choices:
                    logger.warning("Rewrite API returned empty choices")
                    return current_subquery

                choice = response.choices[0]
                content = choice.message.content
                if content is None:
                    logger.warning("Rewrite API returned None content")
                    last_error = ValueError("Empty rewrite output (None)")
                    if (
                        self.is_reasoning_model
                        and choice.finish_reason == "length"
                        and attempt < retry_count
                    ):
                        continue
                    return current_subquery

                if not content.strip():
                    logger.warning("Rewrite API returned empty content")
                    last_error = ValueError("Empty rewrite output")
                    if (
                        self.is_reasoning_model
                        and choice.finish_reason == "length"
                        and attempt < retry_count
                    ):
                        continue
                    return current_subquery

                rewritten = content.strip().strip('"').strip("'")
                logger.debug(f"Rewrote subquery: '{current_subquery}' -> '{rewritten}'")
                return rewritten

            except Exception as e:
                logger.warning(f"Subquery rewrite failed (attempt {attempt + 1}): {e}")
                last_error = e
                if attempt < retry_count:
                    continue
                logger.warning(f"Rewrite failed after retries: {last_error}")
                return current_subquery

        return current_subquery

    def generate_final_answer(
        self,
        original_query: str,
        final_subquery: str,
        context: str,
        chain_summary: str,
        retry_count: int = 2,
    ) -> str:
        """
        Generate the final answer using the accumulated context and chain of answers.

        Args:
            original_query: The original user query
            final_subquery: The final sub-query in the chain
            context: Context from retrieval for the final sub-query
            chain_summary: Summary of all previous sub-queries and their answers
            retry_count: Number of retries on failure

        Returns:
            Final answer to the original query
        """
        prompt = FINAL_ANSWER_PROMPT.format(
            original_query=original_query,
            final_subquery=final_subquery,
            context=context,
            chain_summary=chain_summary,
        )

        last_error = None
        for attempt in range(retry_count + 1):
            try:
                # Build API call parameters based on model type
                if self.is_reasoning_model:
                    # Reasoning models use max_completion_tokens (OpenAI SDK)
                    max_completion_tokens = min(500 * (2**attempt), 8192)
                    reasoning_effort = self.reasoning_effort if attempt == 0 else "low"
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=max_completion_tokens,
                            reasoning_effort=reasoning_effort,
                        )
                    except TypeError as e:
                        # Handle case where reasoning_effort is not supported
                        if "reasoning_effort" in str(e):
                            try:
                                response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=[{"role": "user", "content": prompt}],
                                    max_completion_tokens=max_completion_tokens,
                                )
                            except TypeError as e2:
                                if "max_completion_tokens" in str(e2):
                                    response = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=max_completion_tokens,
                                    )
                                else:
                                    raise
                        # Fallback to max_tokens if max_completion_tokens not supported
                        elif "max_completion_tokens" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_completion_tokens,
                                reasoning_effort=reasoning_effort,
                            )
                        else:
                            raise
                else:
                    # Standard models (gpt-4o, etc.) use max_tokens
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=500,
                    )

                if not response.choices:
                    logger.warning("Final answer API returned empty choices")
                    return ""

                choice = response.choices[0]
                content = choice.message.content
                if content is None:
                    logger.warning("Final answer API returned None content")
                    last_error = ValueError("Empty final answer output (None)")
                    if (
                        self.is_reasoning_model
                        and choice.finish_reason == "length"
                        and attempt < retry_count
                    ):
                        continue
                    return ""

                if not content.strip():
                    logger.warning("Final answer API returned empty content")
                    last_error = ValueError("Empty final answer output")
                    if (
                        self.is_reasoning_model
                        and choice.finish_reason == "length"
                        and attempt < retry_count
                    ):
                        continue
                    return ""

                return content.strip()

            except Exception as e:
                logger.warning(
                    f"Final answer generation failed (attempt {attempt + 1}): {e}"
                )
                last_error = e
                if attempt < retry_count:
                    continue
                logger.error(f"Final answer failed after retries: {last_error}")
                return ""

        return ""


# Singleton for convenience
_sequential_decomposer: Optional[SequentialQueryDecomposer] = None


def get_sequential_decomposer(
    config: Optional[DecompositionConfig] = None, **kwargs
) -> SequentialQueryDecomposer:
    """Get or create sequential decomposer singleton."""
    global _sequential_decomposer
    if _sequential_decomposer is None:
        if config is not None:
            _sequential_decomposer = SequentialQueryDecomposer(config)
        else:
            _sequential_decomposer = SequentialQueryDecomposer(**kwargs)
    return _sequential_decomposer
