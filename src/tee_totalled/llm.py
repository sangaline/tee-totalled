"""LLM client using Phala's Confidential AI API (RedPill) with TEE verification."""

import json
import logging
import re

from openai import AsyncOpenAI

from .config import get_settings
from .verification import get_verifier

logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """Raised when TEE verification fails for an LLM response."""

    pass


class LLMClient:
    """LLM client using RedPill API with TEE-protected inference and verification."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.redpill_api_key,
        )
        self.model = settings.llm_model
        self._log_sensitive = settings.is_development
        self._verify_responses = True

    async def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """Make a completion request to the LLM with TEE verification."""
        if self._log_sensitive:
            logger.debug(f"LLM request - System: {system_prompt[:100]}...")
            logger.debug(f"LLM request - User: {user_prompt[:200]}...")

        # Build the request body for signature verification.
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 500,
        }

        response = await self.client.chat.completions.create(**request_body)

        result = response.choices[0].message.content or ""
        chat_id = response.id

        if self._log_sensitive:
            logger.debug(f"LLM response (chat_id={chat_id}): {result[:200]}...")

        # Verify the response signature to prove it came from the TEE.
        if self._verify_responses and chat_id:
            await self._verify_response(chat_id, request_body, result)

        return result

    async def _verify_response(
        self, chat_id: str, request_body: dict, response_text: str
    ) -> None:
        """Verify the LLM response signature.

        Signature verification is REQUIRED. Every response must be
        cryptographically verified to have come from the TEE.
        """
        verifier = get_verifier()

        # The request body must be serialized exactly as sent.
        # RedPill uses compact JSON (no spaces after separators).
        request_json = json.dumps(request_body, separators=(",", ":"))

        sig_result = await verifier.verify_response_signature(
            chat_id=chat_id,
            request_body=request_json,
            response_text=response_text,
        )

        if sig_result.valid:
            logger.info(f"TEE signature verified for response {chat_id}")
        else:
            # ANY verification failure is a security concern - fail hard.
            error_msg = f"TEE signature verification FAILED: {sig_result.error}"
            logger.error(error_msg)
            raise VerificationError(error_msg)

    async def score_offensiveness(self, message: str) -> int:
        """Score how offensive a message is on a scale of 1-100."""
        system_prompt = """You are an offensiveness scorer for a trust game. Your job is to rate
how offensive a message is on a scale of 1-100, where:
- 1-20: Mild or not offensive at all
- 21-40: Somewhat inappropriate
- 41-60: Moderately offensive
- 61-80: Very offensive
- 81-100: Extremely offensive

Respond with ONLY a JSON object in this exact format: {"score": <number>}
Do not include any other text or explanation."""

        user_prompt = f"Rate the offensiveness of this message:\n\n{message}"

        try:
            response = await self._complete(system_prompt, user_prompt)
            # Parse the JSON response.
            match = re.search(r"\{[^}]*\"score\"\s*:\s*(\d+)[^}]*\}", response)
            if match:
                score = int(match.group(1))
                return max(1, min(100, score))  # Clamp to 1-100.

            # Fallback: try to find any number.
            numbers = re.findall(r"\d+", response)
            if numbers:
                score = int(numbers[0])
                return max(1, min(100, score))

            logger.warning(f"Could not parse score from response: {response}")
            return 50  # Default middle score.

        except VerificationError:
            raise  # Re-raise verification errors.
        except Exception as e:
            logger.error(f"Error scoring message: {e}")
            return 50

    async def generate_moral_reaction(self, messages: list[str]) -> str:
        """Generate a moral reaction to a collection of messages without revealing specifics."""
        system_prompt = """You are providing a thoughtful moral reflection on messages submitted
to a trust game. Users submitted messages trying to be offensive to test their trust in TEE
privacy guarantees.

IMPORTANT: You must NOT reveal any specific content, quotes, or details from the messages.
Instead, provide a general moral reflection on:
- The themes or categories of content people felt comfortable sharing
- What this says about trust levels in privacy technology
- A philosophical observation about human nature and privacy

Keep your response concise (2-3 paragraphs) and thought-provoking.
Never quote or paraphrase any specific message."""

        # Create a summary prompt without exposing raw content in logs.
        user_prompt = f"""Reflect morally on {len(messages)} messages submitted to a trust game.
The messages ranged from mild to potentially very offensive content.
Provide your reflection without revealing any specific message content."""

        # In development, we can include the actual messages for better responses.
        settings = get_settings()
        if settings.is_development:
            user_prompt = f"""Here are {len(messages)} messages submitted to a trust game.
Provide a moral reflection without revealing specific content:

{chr(10).join(f'- Message {i+1}' for i in range(len(messages)))}

(The actual messages are provided for context but must not be quoted or paraphrased.)

Actual messages for your analysis only:
{json.dumps(messages)}"""

        try:
            return await self._complete(system_prompt, user_prompt)
        except VerificationError:
            raise  # Re-raise verification errors.
        except Exception as e:
            logger.error(f"Error generating moral reaction: {e}")
            return "The submissions revealed interesting patterns about trust and privacy."


# Singleton instance.
_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
