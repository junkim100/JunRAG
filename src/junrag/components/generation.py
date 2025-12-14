"""
Generation component for JunRAG.
LLM-based answer generation using OpenAI API.

Supports both standard models (gpt-4o, etc.) and reasoning models (gpt-5.1-2025-11-13, o3, etc.).
"""

import logging
import os
from typing import Dict, List, Optional, Union

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
)

from junrag.models import Chunk, GenerationConfig, GenerationResult, UsageInfo

# Configure logging
logger = logging.getLogger(__name__)

# Models that use reasoning/thinking mode (use max_completion_tokens, no temperature)
REASONING_MODELS = {
    "gpt-5.1-2025-11-13",
    "o3",
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-preview",
}

# Valid reasoning effort levels
VALID_REASONING_EFFORTS = {"low", "medium", "high"}

GENERATION_PROMPT = """
<system_instructions>
You are an expert AI assistant that answers questions strictly based on the provided context.
Your goal is to be accurate, concise, and helpful.
</system_instructions>

<context_data>
{context_str}
</context_data>

<user_query>
{query}
</user_query>

<response_guidelines>
1. **Answer the Query:** Answer <user_query> using ONLY the information from <context_data>. Provide a direct, concise answer.

2. **No Outside Knowledge:** Do not use your internal knowledge to answer.

3. **Extract Answers:** If the context contains ANY relevant information that could answer the question, extract and provide it. Do NOT say "cannot be determined" or "not enough information" - instead, provide the best answer available from the context.

4. **Be Concise:** Give the shortest possible answer. For most questions, a single word, number, or short phrase is sufficient. Do NOT write long explanations or disclaimers.

5. **Citations:** Cite your sources using the [Source ID] format at the end of relevant sentences. If using chain_outputs, cite [chain_outputs].

6. **Chain Outputs Caveat:** If <context_data> contains a synthetic "chain_outputs" section with lines like `answer (internal_knowledge): ...`, treat those as *unverified hints* (they were not grounded in retrieved context). Prefer other context chunks whenever possible; if you must rely on them, cite [chain_outputs].

7. **Numeric Questions:** For numeric questions (ranks, counts, years, ages, populations, etc.), extract the exact number from the context. For rankings, use ordinal format (e.g., "35th", "1st", "2nd"). Just provide the number/rank, not an explanation.

8. **Temporal Questions:** For "as of [date]" questions, use information current as of that specific date. For chronological ordering, extract the sequence from the context.

9. **Tabular/List Questions:** When the question asks about items in a list or table, extract the specific entry that matches the criteria. For "nth" or ordinal questions, find the exact item at that position.

10. **Comparative Questions:** For "which is longer/shorter/taller/older/younger" questions, identify the specific entity and provide the comparison result.

11. **Yes/No Questions:** For yes/no questions, answer with "Yes" or "No" followed by a brief factual basis if available in the context.

12. **Entity Identification:** For name, place, title, or entity questions, extract the exact name/title from the context.

13. **Post-Processing:** If the question asks for rounding, conversion, or formatting, apply the requested transformation to the extracted value.

14. **Multiple Constraints:** When a question has multiple constraints, find the entity that satisfies ALL conditions from the context.

15. **Tone:** Professional and direct. Single keyword or short phrase answers are STRONGLY preferred. Avoid long explanations, disclaimers, or uncertainty statements.

**Examples:**
- Query: "I am thinking of a province that has the smallest land area in it's particular country, but also has the the 10th largest population.  This country has 10 provinces.  This province joined the country in 1873.  What is the scientific name of the provincial flower?"
- Answer: "Cypripedium Acaule"

- Query: "What percentage of his total league appearances did footballer Derek Smith (born 1946) make with the team whose original name is shared by a bird impressionist born in the nineteenth century? Give your answer to two decimal places.?"
- Answer: "95.35%"

- Query: "What city does the band whose song spent the most weeks at No. 1 on the Billboard Hot Rock & Alternative Songs chart as of August 1, 2024 originate from?"
- Answer: "Las Vegas, Nevada"

</response_guidelines>
"""


class LLMGenerator:
    """LLM-based answer generator using OpenAI API."""

    def __init__(
        self,
        model: Union[str, GenerationConfig] = "gpt-5.1-2025-11-13",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 128,
        reasoning_effort: str = "medium",
    ):
        """
        Initialize LLM generator.

        Args:
            model: OpenAI model name (default: gpt-5.1-2025-11-13) or GenerationConfig instance
            api_key: OpenAI API key
            temperature: Generation temperature (ignored for reasoning models)
            max_tokens: Maximum tokens in response (uses max_completion_tokens for reasoning models)
            reasoning_effort: Reasoning effort level for reasoning models (low, medium, high)
        """
        # Handle Pydantic config or individual parameters
        if isinstance(model, GenerationConfig):
            config = model
            self.model = config.model
            api_key = config.api_key or api_key
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
            self.reasoning_effort = config.reasoning_effort
        else:
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens

            # Validate and set reasoning effort
            if reasoning_effort not in VALID_REASONING_EFFORTS:
                logger.warning(
                    f"Invalid reasoning_effort '{reasoning_effort}'. "
                    f"Valid values: {VALID_REASONING_EFFORTS}. Defaulting to 'medium'."
                )
                reasoning_effort = "medium"
            self.reasoning_effort = reasoning_effort

        # Check if this is a reasoning model
        self.is_reasoning_model = any(
            rm in model.lower() for rm in ["gpt-5", "o3", "o1"]
        )

        # Log warnings for incompatible settings
        if self.is_reasoning_model:
            if temperature != 0.1:  # Non-default temperature
                logger.warning(
                    f"Model '{model}' is a reasoning model. "
                    f"Temperature setting ({temperature}) will be ignored."
                )
            logger.info(
                f"Using reasoning model '{model}' with effort='{reasoning_effort}'"
            )
        else:
            if reasoning_effort != "medium":
                logger.warning(
                    f"Model '{model}' is not a reasoning model. "
                    f"reasoning_effort='{reasoning_effort}' will be ignored."
                )

        # Validate API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize client with error handling
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"LLMGenerator initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def _format_context(self, chunks: List[Dict]) -> str:
        """Format chunks into context string."""
        if not chunks:
            return ""

        context_parts = []
        for chunk in chunks:
            text = (
                chunk.get("text", "")
                or chunk.get("chunk_text", "")
                or chunk.get("content", "")
            )

            metadata = chunk.get("metadata", {})
            if isinstance(metadata, dict):
                source = metadata.get("source", "")
            else:
                source = ""

            if text and text.strip():
                if source:
                    context_parts.append(f"Source: {source}\n{text}")
                else:
                    context_parts.append(text)

        return "\n\n".join(context_parts)

    def generate(
        self,
        query: str,
        chunks: Union[List[Dict], List[Chunk]],
        prompt_template: Optional[str] = None,
        retry_count: int = 2,
    ) -> GenerationResult:
        """
        Generate answer from query and context chunks.

        Args:
            query: User query
            chunks: Context chunks (dicts or Chunk models)
            prompt_template: Custom prompt template
            retry_count: Number of retries on transient failures

        Returns:
            GenerationResult with answer and usage info
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not chunks:
            logger.warning("No chunks provided for generation")
            raise ValueError("No chunks provided for generation")

        # Convert Chunk models to dicts for processing
        chunks_dict = []
        for chunk in chunks:
            if isinstance(chunk, Chunk):
                chunks_dict.append(chunk.model_dump())
            else:
                chunks_dict.append(chunk)

        template = prompt_template or GENERATION_PROMPT
        context_str = self._format_context(chunks_dict)

        if not context_str:
            logger.error(
                f"No text content found in {len(chunks)} chunks. "
                "Check chunk format (expected 'text', 'chunk_text', or 'content' keys)."
            )
            raise ValueError("No context available from chunks")

        try:
            prompt = template.format(context_str=context_str, query=query)
        except KeyError as e:
            logger.error(f"Invalid prompt template - missing key: {e}")
            raise ValueError(f"Invalid prompt template: missing {e}")

        # Attempt API call with retries
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                # Build API call parameters based on model type
                if self.is_reasoning_model:
                    # Reasoning models (gpt-5.1, o3, o1) use different parameters
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=self.max_tokens,
                            reasoning_effort=self.reasoning_effort,
                        )
                    except TypeError as e:
                        # Handle case where reasoning_effort is not supported
                        if "reasoning_effort" in str(e):
                            logger.warning(
                                f"Model '{self.model}' does not support reasoning_effort. "
                                "Retrying without it."
                            )
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                max_completion_tokens=self.max_tokens,
                            )
                        else:
                            raise
                else:
                    # Standard models (gpt-4o, etc.)
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                    except TypeError as e:
                        # Handle case where max_tokens might need to be max_completion_tokens
                        if "max_tokens" in str(e):
                            logger.warning(
                                f"Model '{self.model}' may require max_completion_tokens. Retrying."
                            )
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=self.temperature,
                                max_completion_tokens=self.max_tokens,
                            )
                        else:
                            raise

                # Validate response
                if not response.choices:
                    raise ValueError("API returned empty choices")

                answer = response.choices[0].message.content
                if answer is None:
                    logger.warning("API returned None for message content")
                    answer = ""

                usage = UsageInfo(
                    prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                    completion_tokens=getattr(response.usage, "completion_tokens", 0),
                    total_tokens=getattr(response.usage, "total_tokens", 0),
                )

                return GenerationResult(
                    answer=answer,
                    model=self.model,
                    usage=usage,
                )

            except AuthenticationError as e:
                logger.error(f"OpenAI authentication failed: {e}")
                raise ValueError(
                    f"OpenAI API authentication failed. Check your API key."
                ) from e

            except RateLimitError as e:
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{retry_count + 1}): {e}"
                )
                last_error = e
                if attempt < retry_count:
                    import time

                    wait_time = 2**attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                raise

            except APIConnectionError as e:
                logger.error(f"Failed to connect to OpenAI API: {e}")
                last_error = e
                if attempt < retry_count:
                    continue
                raise ConnectionError(f"Failed to connect to OpenAI API: {e}") from e

            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                # Check for model-specific errors
                if "model" in str(e).lower() and "not found" in str(e).lower():
                    raise ValueError(
                        f"Model '{self.model}' not found. Check model name or API access."
                    ) from e
                last_error = e
                if attempt < retry_count:
                    continue
                raise

            except Exception as e:
                logger.error(
                    f"Unexpected error during generation: {type(e).__name__}: {e}"
                )
                last_error = e
                if attempt < retry_count:
                    continue
                raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Generation failed for unknown reason")


# Singleton for convenience
_generator: Optional[LLMGenerator] = None


def get_generator(config: Optional[GenerationConfig] = None, **kwargs) -> LLMGenerator:
    """Get or create generator singleton."""
    global _generator
    if _generator is None:
        if config is not None:
            _generator = LLMGenerator(config)
        else:
            _generator = LLMGenerator(**kwargs)
    return _generator


def generate_answer(
    query: str,
    chunks: Union[List[Dict], List[Chunk]],
    generator: Optional[LLMGenerator] = None,
    config: Optional[GenerationConfig] = None,
    **kwargs,
) -> GenerationResult:
    """
    Convenience function to generate answer.

    Args:
        query: User query
        chunks: Context chunks
        generator: Optional pre-initialized generator
        config: Optional GenerationConfig instance
        **kwargs: Arguments for LLMGenerator if creating new

    Returns:
        GenerationResult with answer and usage info
    """
    if generator is None:
        if config is not None:
            generator = get_generator(config=config)
        else:
            generator = get_generator(**kwargs)
    return generator.generate(query, chunks)
