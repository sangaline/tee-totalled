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
            logger.debug(f"LLM request - System: {system_prompt}")
            logger.debug(f"LLM request - User: {user_prompt}")

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
            logger.debug(f"LLM response (chat_id={chat_id}): {result}")

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
        system_prompt = """You are an offensiveness scorer for a trust game. Rate how offensive a \
message is on a scale of 1-100. This scale is HEAVILY compressed at the bottom and VERY spread \
out at the top. Most content that people consider "offensive" should score 15-30. Scores above 50 \
are extraordinarily rare. Use this rubric:

1-5: Completely benign. Everyday statements, opinions, compliments, jokes, factual observations. \
"I like pizza" = 2, "Mondays are terrible" = 4.

6-10: Mildly impolite. Minor insults, sarcasm, light profanity used casually. "That's a dumb \
idea" = 7, a single mild swear word = 9.

11-15: Rude or crude. Harsher profanity, mean personal insults, crude humor, or dismissive \
generalizations about groups. Would make a stranger uncomfortable. "You're an idiot" with strong \
profanity = 13.

16-20: Clearly offensive. Aggressive personal attacks, explicit crude sexual references, casual \
use of a slur, or mocking a demographic group. Would be considered unacceptable in most social \
settings. A single slur used as an insult = 18.

21-30: Targeted hate speech against a specific group using dehumanizing language, graphic violent \
or sexual descriptions, or content promoting discrimination. Would be reported on social media. \
A hateful rant targeting one group = 25.

31-40: Intensely hateful content with dehumanizing language AND violent intent toward a group, \
graphic descriptions of violence against people, or deeply degrading content that combines two \
offensive dimensions (e.g. racism + violence, or sexism + graphic sexual degradation).

41-55: Combines multiple severe categories: e.g. extreme racial hatred with violent threats and \
dehumanization, or targeted bigotry across two or more groups combined with graphic content. \
Would shock even people accustomed to offensive content.

56-70: Maximally hateful across three or more dimensions simultaneously. Must combine extreme \
slurs, calls for violence, dehumanization, and graphic depravity targeting multiple groups. \
Content at this level is rare even on the most unmoderated corners of the internet.

71-85: Requires combining extreme hatred toward many groups at once (racial, gender, sexual \
orientation, religious, disability) with graphic violent and sexual depravity, references to \
harming the most vulnerable, and maximally dehumanizing framing. Each additional axis of \
offensiveness pushes the score higher.

86-95: Nearly the theoretical maximum. Must simultaneously deploy the most extreme forms of \
every category: intersecting racial and ethnic hatred, misogyny, homophobia, anti-religious \
bigotry, graphic references to harming children, extreme violent and sexual depravity, and \
dehumanization, all woven together with creative cruelty and maximum shock value.

96-100: The absolute ceiling. A score here means the message has achieved a level of \
offensiveness that is difficult to even conceive of exceeding. Every possible dimension of \
hatred, degradation, taboo violation, and depravity is present at its most extreme, combined \
in a way that is maximally harmful and shocking. Virtually impossible to achieve.

CRITICAL: You MUST use the low end of the scale. A message with one slur is around 18, not 60. \
A hateful rant about one group is 25, not 75. Most "offensive" content people write scores \
15-30. Resist the urge to inflate. Only content combining MANY extreme dimensions at once \
reaches above 50.

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

# Singleton instance.
_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
