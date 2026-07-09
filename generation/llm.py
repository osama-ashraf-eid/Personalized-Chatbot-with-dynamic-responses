from typing import List

from openai import OpenAI

from config import (
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)

from core.logging import get_logger
from core.exceptions import LLMConnectionError

logger = get_logger(__name__)


class LLMClient:
    """
    LLM Client using Hugging Face Inference Providers.

    Uses the OpenAI-compatible router at
    https://router.huggingface.co/v1, so it works with
    any HF-hosted model (append ':provider' to the model
    id, or ':auto' to let HF pick a provider).
    """

    _client = None

    def __init__(
        self,
        api_url: str = LLM_API_URL,
        api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ):

        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if LLMClient._client is None:

            logger.info(
                f"Initializing LLM client: {model} "
                f"at {api_url}"
            )

            LLMClient._client = OpenAI(
                base_url=api_url,
                api_key=api_key,
            )

        self.client = LLMClient._client

    # --------------------------------------------------
    # Generate with Messages
    # --------------------------------------------------

    def generate(
        self,
        messages: List[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Parameters
        ----------
        messages
            List of message dicts with 'role' and 'content'.

        max_tokens
            Override default max tokens.

        temperature
            Override default temperature.

        Returns
        -------
        str
            Generated text response.
        """

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
            )

            content = response.choices[0].message.content

            logger.info(
                f"LLM generated {len(content)} chars"
            )

            return content.strip()

        except Exception as e:

            logger.error(
                f"LLM generation failed: {e}"
            )

            raise LLMConnectionError(
                f"Failed to connect to LLM at "
                f"{self.api_url}: {e}"
            ) from e

    # --------------------------------------------------
    # Simple Generate
    # --------------------------------------------------

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Convenience method with system + user messages.
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        return self.generate(messages)