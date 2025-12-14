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
import re
import time as time_module
from typing import Any, Dict, List, Optional, Union

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
7. **PRESERVE THE USER ASK:** Do NOT change the task type. If the user asks for a *rank*, the final step must ask for a *rank* (not a different metric like a count), unless you explicitly include the missing conversion step (e.g., if you ask for "how many buildings are taller", also include "what rank does that imply?").
8. **RETRIEVAL-FRIENDLY QUERIES:** Each sub-query should be phrased like a search query and include key terms, entities, locations, time constraints, and units (e.g., feet, August 2024). When relevant, include canonical page-style keywords (e.g., "List of tallest buildings in New York City") to improve retrieval.

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

CRITICAL RULES:
1. Use ONLY the information from the context and chain of answers.
2. Be concise and direct - single keyword or short phrase answers are preferred when appropriate.
3. For numeric questions (ranks, counts, years, ages, etc.), provide the exact number.
4. For ranking questions, use ordinal format (e.g., "35th", "1st", "2nd").
5. For "as of [date]" questions, use information current as of that specific date.
6. For yes/no questions, answer with "Yes" or "No" followed by brief factual basis if needed.
7. For entity identification (names, places, titles), provide the exact name/title.
8. For comparative questions, identify the specific entity and provide the comparison result.
9. For post-processing (rounding, conversion), apply the requested transformation.
10. For multiple constraints, ensure the answer satisfies ALL conditions.

Answer:"""


SUBQUERY_ANSWER_SYSTEM_PROMPT = """You answer the user's question using ONLY the provided context.

CRITICAL RULES:
1. Use ONLY the context. Do not use outside knowledge.
2. Provide a clear, direct answer without any explanations or ranges.
3. If you cite sources from the context, include the bracketed source numbers (e.g., [1], [3]) in your answer.
4. NEVER respond with "Unknown", "I don't know", "Unable to determine", or similar phrases.
5. If the answer is not clearly in the context, respond with an empty string (not "Unknown").
6. If the context contains any relevant information, extract and provide it as the answer.
7. Give a direct answer only - no explanations, no uncertainty, no ranges, no "Unknown" responses.

NUMERIC QUESTIONS:
8. For numeric questions (heights, ranks, counts, classifications, years, ages, populations, etc.), extract the exact number from the context.
9. For ranking questions, provide the rank as a number followed by "th", "st", "nd", or "rd" (e.g., "35th", "1st", "2nd", "3rd").
10. For classification numbers (Dewey Decimal, atomic numbers, etc.), extract the exact classification code/number from the context.
11. For years, dates, and ages, extract the exact numeric value (e.g., "1847", "1960", "52 years old").
12. For differences, calculations, or comparisons, extract the exact numeric result from the context.

TABULAR/LIST QUESTIONS:
13. When the question asks about items in a list, table, or ranking (e.g., "List of tallest buildings", "List of presidents"), extract the specific entry that matches the criteria.
14. For "nth" or ordinal questions (e.g., "15th first lady", "9th largest country"), find the exact item at that position in the list/table.
15. For "first", "last", "oldest", "youngest", "tallest", "shortest" questions, identify the specific entity from the context.

TEMPORAL QUESTIONS:
16. For "as of [date]" questions, use information current as of that specific date from the context.
17. For chronological ordering questions, extract the sequence or specific dates/years from the context.
18. For "how many years" or temporal differences, calculate or extract the exact number from the context.

COMPARATIVE QUESTIONS:
19. For "which is longer/shorter/taller/older/younger" questions, identify the specific entity and provide the comparison result.
20. For "how much more/less" questions, extract the exact difference from the context.

YES/NO QUESTIONS:
21. For yes/no questions, answer with "Yes" or "No" followed by a brief factual basis if available in the context.

ENTITY IDENTIFICATION:
22. For name, place, title, or entity questions, extract the exact name/title from the context.
23. For "what is the name of" questions, provide the exact name without additional context.

POST-PROCESSING:
24. If the question asks for rounding, conversion, or formatting, apply the requested transformation to the extracted value.
25. For "round to the nearest X" questions, round the extracted number accordingly.
26. For unit conversions, convert the extracted value to the requested unit if specified in the question.

MULTIPLE CONSTRAINTS:
27. When a question has multiple constraints (e.g., "who was X and also Y"), find the entity that satisfies ALL conditions from the context.
28. Extract the answer that matches all specified criteria simultaneously.
"""

SUBQUERY_ANSWER_NO_CONTEXT_SYSTEM_PROMPT = """You answer the user's question using your internal knowledge.

CRITICAL RULES:
1. You MUST provide a specific, factual answer based on your training data.
2. NEVER respond with "Unknown", "I don't know", "Unable to determine", or similar phrases.
3. If you have any relevant knowledge about the question, provide that answer even if you're not 100% certain.
4. Provide a clear, direct answer without any explanations, uncertainty statements, or ranges.
5. If the question asks about a specific entity, person, place, or fact, provide the most likely answer from your knowledge.
6. Do NOT include phrases like "I believe", "possibly", "might be", or any hedging language.
7. Give ONLY the answer itself - no explanations, no context, no disclaimers.

NUMERIC QUESTIONS:
8. For numeric questions (heights, ranks, counts, classifications, years, ages, populations, etc.), provide the exact number from your knowledge.
9. For ranking questions, provide the rank as a number followed by "th", "st", "nd", or "rd" (e.g., "35th", "1st", "2nd", "3rd").
10. For classification numbers (Dewey Decimal, atomic numbers, etc.), provide the exact classification code/number.
11. For years, dates, and ages, provide the exact numeric value (e.g., "1847", "1960", "52 years old").
12. For differences, calculations, or comparisons, provide the exact numeric result.

TABULAR/LIST QUESTIONS:
13. When the question asks about items in a list, table, or ranking, identify the specific entry that matches the criteria.
14. For "nth" or ordinal questions (e.g., "15th first lady", "9th largest country"), identify the exact item at that position.
15. For "first", "last", "oldest", "youngest", "tallest", "shortest" questions, identify the specific entity.

TEMPORAL QUESTIONS:
16. For "as of [date]" questions, use information current as of that specific date.
17. For chronological ordering questions, provide the sequence or specific dates/years.
18. For "how many years" or temporal differences, calculate or provide the exact number.

COMPARATIVE QUESTIONS:
19. For "which is longer/shorter/taller/older/younger" questions, identify the specific entity and provide the comparison result.
20. For "how much more/less" questions, provide the exact difference.

YES/NO QUESTIONS:
21. For yes/no questions, answer with "Yes" or "No" followed by a brief factual basis if relevant.

ENTITY IDENTIFICATION:
22. For name, place, title, or entity questions, provide the exact name/title.
23. For "what is the name of" questions, provide the exact name without additional context.

POST-PROCESSING:
24. If the question asks for rounding, conversion, or formatting, apply the requested transformation to your answer.
25. For "round to the nearest X" questions, round your answer accordingly.
26. For unit conversions, convert to the requested unit if specified.

MULTIPLE CONSTRAINTS:
27. When a question has multiple constraints (e.g., "who was X and also Y"), identify the entity that satisfies ALL conditions.
28. Provide the answer that matches all specified criteria simultaneously.

If you truly have no knowledge about the question, provide the most reasonable inference or related fact you do know, but still give a concrete answer.
"""


class SequentialQueryDecomposer:
    """Decomposes complex multi-hop queries into sequential sub-queries with placeholders."""

    def __init__(
        self,
        model: Union[str, DecompositionConfig] = "gpt-4o",
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
                    # For decomposition, use "low" reasoning effort from the start to ensure
                    # tokens are available for output (not consumed by reasoning).
                    # Start with larger token budget to account for reasoning overhead.
                    base_tokens = max(2000, self.max_tokens * 4)  # At least 2000 tokens
                    max_completion_tokens = min(base_tokens * (2**attempt), 16384)
                    reasoning_effort = "low"  # Always use low for decomposition
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

    def answer_subquery(
        self,
        subquery: str,
        context: str,
        original_query: Optional[str] = None,
        retry_count: int = 2,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Answer a sub-query using ONLY the provided context.

        Returns a dict:
          - answer: str
          - supporting_sources: List[int] (1-indexed source numbers like [1], [2])
          - used_internal_knowledge: bool (True if internal knowledge fallback was used)
        """
        subquery = (subquery or "").strip()
        context = (context or "").strip()

        # Validate inputs
        if not subquery:
            return {
                "answer": "Unable to determine from provided context",
                "supporting_sources": [],
                "used_internal_knowledge": False,
            }

        # IMPORTANT: Always try context-based prompt first, even if context is empty
        # Only fall back to internal knowledge if answer is "Unknown" or empty after trying with context

        # Build user message with original query context for better understanding
        user_content_parts = [f"<question>\n{subquery}\n</question>"]
        if original_query and original_query.strip() and original_query != subquery:
            user_content_parts.append(
                f"\n<original_user_query>\n{original_query}\n</original_user_query>"
            )
        user_content_parts.append(f"\n<context>\n{context}\n</context>")

        messages = [
            {"role": "system", "content": SUBQUERY_ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "".join(user_content_parts),
            },
        ]

        last_error: Optional[Exception] = None
        for attempt in range(retry_count + 1):
            max_completion_tokens = min(max_tokens * (2**attempt), 2048)
            reasoning_effort = self.reasoning_effort if attempt == 0 else "low"

            try:
                if self.is_reasoning_model:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_completion_tokens=max_completion_tokens,
                            reasoning_effort=reasoning_effort,
                        )
                    except (TypeError, ValueError, APIError, Exception) as e:
                        # response_format unsupported (can be TypeError, ValueError, or APIError)
                        error_str = str(e).lower()
                        if (
                            "response_format" in error_str
                            or "json_object" in error_str
                            or "must contain the word 'json'" in error_str
                        ):
                            try:
                                response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=messages,
                                    max_completion_tokens=max_completion_tokens,
                                    reasoning_effort=reasoning_effort,
                                )
                            except (TypeError, APIError) as e2:
                                error_str2 = str(e2).lower()
                                if "max_completion_tokens" in error_str2:
                                    response = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        max_tokens=max_completion_tokens,
                                        reasoning_effort=reasoning_effort,
                                    )
                                else:
                                    raise
                        # reasoning_effort unsupported
                        elif "reasoning_effort" in error_str:
                            try:
                                response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=messages,
                                    max_completion_tokens=max_completion_tokens,
                                )
                            except (TypeError, ValueError, APIError) as e2:
                                error_str2 = str(e2).lower()
                                if (
                                    "response_format" in error_str2
                                    or "json_object" in error_str2
                                    or "must contain the word 'json'" in error_str2
                                ):
                                    response = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        max_completion_tokens=max_completion_tokens,
                                    )
                                elif "max_completion_tokens" in error_str2:
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
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=max_completion_tokens,
                        )
                    except (TypeError, ValueError, APIError, Exception) as e:
                        error_str = str(e).lower()
                        if (
                            "response_format" in error_str
                            or "json_object" in error_str
                            or "must contain the word 'json'" in error_str
                        ):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=self.temperature,
                                max_tokens=max_completion_tokens,
                            )
                        else:
                            raise

                if not response.choices:
                    last_error = ValueError("Empty choices")
                    continue

                choice = response.choices[0]
                content = choice.message.content or ""

                if not content.strip():
                    last_error = ValueError("Empty content")
                    if (
                        self.is_reasoning_model
                        and choice.finish_reason == "length"
                        and attempt < retry_count
                    ):
                        continue
                    # Fallback: Extract from context if available
                    fallback_answer = ""
                    if context:
                        context_lines = context.split("\n")
                        for line in context_lines:
                            text = line.strip()
                            # Remove source markers like [1] at the start
                            if re.match(r"^\[\d+\]\s*", text):
                                text = re.sub(r"^\[\d+\]\s*", "", text).strip()
                            # Look for numeric answers (Dewey Decimal, heights, ranks, etc.)
                            if text and len(text) > 5:
                                # Try to extract numeric patterns (e.g., "823.8", "35th", "1st")
                                numeric_match = re.search(
                                    r"\b(\d+(?:\.\d+)?(?:th|st|nd|rd)?)\b", text
                                )
                                if numeric_match:
                                    fallback_answer = numeric_match.group(1)
                                    break
                                # Fallback: use first meaningful sentence
                                first_sentence = text.split(".")[0].strip()
                                if first_sentence and len(first_sentence) > 10:
                                    fallback_answer = first_sentence[:200]
                                    break
                    if not fallback_answer:
                        # Content is empty and context extraction failed - fall back to internal knowledge
                        logger.info(
                            "Content is empty. Falling back to internal knowledge..."
                        )
                        result = self._answer_with_internal_knowledge(
                            subquery, retry_count, max_tokens
                        )
                        result["used_internal_knowledge"] = True
                        return result
                    return {
                        "answer": fallback_answer,
                        "supporting_sources": [],
                        "used_internal_knowledge": False,
                    }

                content = content.strip()

                # Extract answer from plain text response
                answer = content.strip()

                # Extract supporting sources from the text (look for [1], [2], etc.)
                sources: List[int] = []
                source_pattern = r"\[(\d+)\]"
                matches = re.findall(source_pattern, answer)
                for match in matches:
                    try:
                        si = int(match)
                        if si > 0:
                            sources.append(si)
                    except Exception:
                        continue

                # Remove source markers from answer for cleaner output
                answer = re.sub(r"\[\d+\]", "", answer).strip()

                # Filter out "Unknown" answers even when context is available
                answer_lower = answer.lower()
                unknown_phrases = [
                    "unknown",
                    "i don't know",
                    "i do not know",
                    "unable to determine",
                    "cannot determine",
                    "no information",
                    "not available",
                    "not found",
                    "no answer",
                ]

                is_unknown = any(phrase in answer_lower for phrase in unknown_phrases)

                # If answer is unknown and we have retries left, try again
                if is_unknown and attempt < retry_count:
                    logger.warning(
                        f"LLM returned 'Unknown' answer despite context (attempt {attempt + 1}/{retry_count + 1}). "
                        "Retrying with stronger prompt..."
                    )
                    # Enhance the prompt to be more insistent
                    enhanced_messages = [
                        {
                            "role": "system",
                            "content": SUBQUERY_ANSWER_SYSTEM_PROMPT
                            + "\n\nIMPORTANT: You must provide a concrete answer from the context. "
                            "Do NOT say 'Unknown' or 'I don't know'. "
                            "If the context doesn't directly answer the question, provide the most relevant information from the context.",
                        },
                        {
                            "role": "user",
                            "content": (
                                f"<question>\n{subquery}\n</question>\n\n"
                                f"<context>\n{context}\n</context>\n\n"
                                "You must answer this question using the provided context. "
                                "Provide a specific answer. Do not say 'Unknown'."
                            ),
                        },
                    ]
                    messages = enhanced_messages
                    continue

                # If still unknown after retries, try to extract from context first
                if is_unknown:
                    logger.warning(
                        f"LLM returned 'Unknown' after retries. Extracting from context..."
                    )
                    # Try to extract answer from context
                    extracted_answer = None
                    if context:
                        context_lines = context.split("\n")
                        for line in context_lines:
                            if line.strip() and not line.strip().startswith("["):
                                text = line.strip()
                                if re.match(r"^\[\d+\]\s*", text):
                                    text = re.sub(r"^\[\d+\]\s*", "", text)
                                if text and len(text) > 10:
                                    first_sentence = text.split(".")[0].strip()
                                    if first_sentence and len(first_sentence) > 10:
                                        extracted_answer = first_sentence[:200]
                                        logger.info(
                                            f"Extracted answer from context: {extracted_answer[:50]}..."
                                        )
                                        break

                    # If we successfully extracted from context, use it
                    if extracted_answer:
                        answer = extracted_answer
                        # Check if extracted answer is still "Unknown"
                        extracted_lower = extracted_answer.lower()
                        is_still_unknown = any(
                            phrase in extracted_lower for phrase in unknown_phrases
                        )
                        if is_still_unknown:
                            # Extracted answer is still "Unknown" - fall back to internal knowledge
                            logger.info(
                                "Extracted answer is still 'Unknown'. "
                                "Falling back to internal knowledge..."
                            )
                            result = self._answer_with_internal_knowledge(
                                subquery, retry_count, max_tokens
                            )
                            result["used_internal_knowledge"] = True
                            return result
                        # Successfully extracted valid answer from context, return it
                        return {
                            "answer": answer,
                            "supporting_sources": sources,
                            "used_internal_knowledge": False,
                        }
                    else:
                        # Context extraction failed - fall back to internal knowledge
                        logger.info(
                            "Could not extract answer from context. "
                            "Falling back to internal knowledge..."
                        )
                        result = self._answer_with_internal_knowledge(
                            subquery, retry_count, max_tokens
                        )
                        result["used_internal_knowledge"] = True
                        return result

                # If answer is empty (but not "Unknown"), try fallbacks
                if not answer:
                    # Fallback: Try to extract from context if available
                    if context:
                        context_lines = context.split("\n")
                        for line in context_lines:
                            if line.strip() and not line.strip().startswith("["):
                                text = line.strip()
                                if re.match(r"^\[\d+\]\s*", text):
                                    text = re.sub(r"^\[\d+\]\s*", "", text)
                                if text and len(text) > 10:
                                    first_sentence = text.split(".")[0].strip()
                                    if first_sentence and len(first_sentence) > 10:
                                        answer = first_sentence[:200]
                                        break

                    # Answer is still empty - fall back to internal knowledge
                    if not answer:
                        logger.info(
                            "Answer is empty. Falling back to internal knowledge..."
                        )
                        result = self._answer_with_internal_knowledge(
                            subquery, retry_count, max_tokens
                        )
                        result["used_internal_knowledge"] = True
                        return result

                return {
                    "answer": answer,
                    "supporting_sources": sources,
                    "used_internal_knowledge": False,
                }

            except Exception as e:
                last_error = e
                if attempt < retry_count:
                    continue
                logger.warning(f"Subquery answer failed after retries: {last_error}")
                # Fallback: Extract from context if available
                fallback_answer = ""
                if context:
                    context_lines = context.split("\n")
                    for line in context_lines:
                        if line.strip() and not line.strip().startswith("["):
                            text = line.strip()
                            if re.match(r"^\[\d+\]\s*", text):
                                text = re.sub(r"^\[\d+\]\s*", "", text)
                            if text and len(text) > 10:
                                first_sentence = text.split(".")[0].strip()
                                if first_sentence and len(first_sentence) > 10:
                                    fallback_answer = first_sentence[:200]
                                    break
                if not fallback_answer:
                    # Exception occurred and extraction failed - fall back to internal knowledge
                    logger.info(
                        "Exception occurred and could not extract from context. "
                        "Falling back to internal knowledge..."
                    )
                    result = self._answer_with_internal_knowledge(
                        subquery, retry_count, max_tokens
                    )
                    result["used_internal_knowledge"] = True
                    return result
                return {
                    "answer": fallback_answer,
                    "supporting_sources": [],
                    "used_internal_knowledge": False,
                }

        logger.warning(f"Subquery answer failed. Last error: {last_error}")
        # Final fallback: Always use internal knowledge when all attempts fail
        logger.info("All attempts failed. Falling back to internal knowledge...")
        result = self._answer_with_internal_knowledge(subquery, retry_count, max_tokens)
        result["used_internal_knowledge"] = True
        return result

    def _answer_with_internal_knowledge(
        self,
        subquery: str,
        retry_count: int = 2,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Answer a sub-query using LLM's internal knowledge (no context provided).
        This is used as a fallback when context-based answering fails.

        Returns a dict:
          - answer: str
          - supporting_sources: List[int] (empty, as no context sources)
          - used_internal_knowledge: bool (always True for this method)
        """
        subquery = (subquery or "").strip()
        if not subquery:
            return {
                "answer": "Unable to determine",
                "supporting_sources": [],
                "used_internal_knowledge": True,
            }

        messages = [
            {"role": "system", "content": SUBQUERY_ANSWER_NO_CONTEXT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<question>\n{subquery}\n</question>",
            },
        ]

        last_error: Optional[Exception] = None
        for attempt in range(retry_count + 1):
            max_completion_tokens = min(max_tokens * (2**attempt), 2048)
            reasoning_effort = self.reasoning_effort if attempt == 0 else "low"

            try:
                if self.is_reasoning_model:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_completion_tokens=max_completion_tokens,
                            reasoning_effort=reasoning_effort,
                        )
                    except TypeError as e:
                        if "max_completion_tokens" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                max_tokens=max_completion_tokens,
                                reasoning_effort=reasoning_effort,
                            )
                        elif "reasoning_effort" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                max_completion_tokens=max_completion_tokens,
                            )
                        else:
                            raise
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=max_completion_tokens,
                    )

                if not response.choices:
                    last_error = ValueError("Empty choices")
                    continue

                choice = response.choices[0]
                content = choice.message.content or ""

                if not content.strip():
                    last_error = ValueError("Empty content")
                    if (
                        self.is_reasoning_model
                        and choice.finish_reason == "length"
                        and attempt < retry_count
                    ):
                        continue
                    # Even with internal knowledge, return a default if empty
                    return {
                        "answer": "Unable to determine",
                        "supporting_sources": [],
                        "used_internal_knowledge": True,
                    }

                content = content.strip()

                # Extract answer from plain text response
                answer = content.strip()

                # Filter out "Unknown" and similar phrases
                answer_lower = answer.lower()
                unknown_phrases = [
                    "unknown",
                    "i don't know",
                    "i do not know",
                    "unable to determine",
                    "cannot determine",
                    "no information",
                    "not available",
                    "not found",
                    "no answer",
                ]

                # Check if answer contains unknown phrases
                is_unknown = any(phrase in answer_lower for phrase in unknown_phrases)

                # If answer is unknown and we have retries left, try again with stronger prompt
                if is_unknown and attempt < retry_count:
                    logger.warning(
                        f"LLM returned 'Unknown' answer (attempt {attempt + 1}/{retry_count + 1}). "
                        "Retrying with stronger prompt..."
                    )
                    # Enhance the prompt to be more insistent
                    enhanced_messages = [
                        {
                            "role": "system",
                            "content": SUBQUERY_ANSWER_NO_CONTEXT_SYSTEM_PROMPT
                            + "\n\nIMPORTANT: You must provide a concrete answer. "
                            "Do NOT say 'Unknown' or 'I don't know'. "
                            "Use your best knowledge to provide the most likely answer.",
                        },
                        {
                            "role": "user",
                            "content": f"<question>\n{subquery}\n</question>\n\n"
                            "You must answer this question based on your knowledge. "
                            "Provide a specific, factual answer. Do not say 'Unknown'.",
                        },
                    ]
                    messages = enhanced_messages
                    continue

                # If still unknown after retries, try to extract any meaningful content
                if is_unknown:
                    # Try to find any entity or fact mentioned in the response
                    # Sometimes LLMs say "Unknown" but provide context
                    import re

                    # Look for capitalized words (potential entities)
                    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer)
                    if entities:
                        # Use the first substantial entity as answer
                        for entity in entities:
                            if (
                                len(entity) > 3
                                and entity.lower() not in unknown_phrases
                            ):
                                answer = entity
                                logger.info(
                                    f"Extracted entity '{entity}' from 'Unknown' response"
                                )
                                break

                    # If still unknown, log warning but return it
                    if any(phrase in answer.lower() for phrase in unknown_phrases):
                        logger.warning(
                            f"LLM returned 'Unknown' answer after {retry_count + 1} attempts "
                            f"for subquery: {subquery}"
                        )

                # No supporting sources for internal knowledge answers
                sources: List[int] = []

                if not answer:
                    answer = "Unable to determine"

                return {
                    "answer": answer,
                    "supporting_sources": sources,
                    "used_internal_knowledge": True,
                }

            except Exception as e:
                last_error = e
                if attempt < retry_count:
                    continue
                logger.warning(
                    f"Internal knowledge answer failed after retries: {last_error}"
                )
                return {
                    "answer": "Unable to determine",
                    "supporting_sources": [],
                    "used_internal_knowledge": True,
                }

        logger.warning(f"Internal knowledge answer failed. Last error: {last_error}")
        return {
            "answer": "Unable to determine",
            "supporting_sources": [],
            "used_internal_knowledge": True,
        }

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
