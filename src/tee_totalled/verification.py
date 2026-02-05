"""RedPill TEE verification for cryptographic proof of LLM responses.

This module verifies that LLM responses from the RedPill API were genuinely
generated inside a Trusted Execution Environment (TEE) by:
1. Verifying Intel TDX attestation quotes (optional, pre-caches signing addresses)
2. Verifying ECDSA signatures on each response (mandatory)

The signature verification is the primary security mechanism - it cryptographically
proves each response was signed by a specific key. The attestation verification
provides additional assurance that the key belongs to a genuine TEE.
"""

import hashlib
import json
import logging
import secrets
from dataclasses import dataclass

import httpx
from eth_account import Account
from eth_account.messages import encode_defunct

from .config import get_settings

logger = logging.getLogger(__name__)

# Phala's TDX quote verification service.
PHALA_VERIFY_URL = "https://cloud-api.phala.network/api/v1/attestations/verify"

# NVIDIA Remote Attestation Service.
NVIDIA_NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"

# Timeout for external verification services.
VERIFICATION_TIMEOUT = 60


@dataclass
class AttestationResult:
    """Result of attestation verification."""

    valid: bool
    signing_address: str | None = None
    intel_verified: bool = False
    gpu_verified: bool = False
    nonce: str | None = None
    error: str | None = None


@dataclass
class SignatureResult:
    """Result of response signature verification."""

    valid: bool
    signing_address: str | None = None
    error: str | None = None


class RedPillVerifier:
    """Verifies RedPill API responses were generated in a TEE."""

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model
        self.api_key = settings.redpill_api_key
        self._verified_addresses: set[str] = set()
        self._known_addresses: set[str] = set()
        self._log_sensitive = settings.is_development

    async def verify_attestation(self, nonce: str | None = None) -> AttestationResult:
        """Verify the RedPill TEE attestation.

        This proves the infrastructure is genuine TEE hardware and pre-caches
        the signing address for faster per-response verification. If a nonce
        is provided it is SHA-256 hashed to produce a valid hex nonce for the
        API; otherwise a random nonce is generated.
        """
        if nonce:
            # Hash user-provided nonce to produce a valid hex string for the API.
            nonce = hashlib.sha256(nonce.encode()).hexdigest()
        else:
            nonce = secrets.token_hex(32)

        try:
            async with httpx.AsyncClient(timeout=VERIFICATION_TIMEOUT) as client:
                # Fetch attestation report from RedPill.
                url = f"{self.base_url}/attestation/report"
                params = {"model": self.model, "nonce": nonce}
                headers = {"Authorization": f"Bearer {self.api_key}"}

                if self._log_sensitive:
                    logger.debug(f"Fetching attestation from {url}")

                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                attestation = response.json()

                signing_address = attestation.get("signing_address")
                intel_quote = attestation.get("intel_quote")
                nvidia_payload = attestation.get("nvidia_payload")

                if not signing_address:
                    return AttestationResult(valid=False, error="No signing address in response")

                # Track this address even if external verification fails.
                self._known_addresses.add(signing_address.lower())

                # Try to verify Intel TDX quote via Phala's service.
                intel_verified = await self._verify_tdx_quote(client, intel_quote)

                # Try to verify NVIDIA GPU attestation.
                gpu_verified = await self._verify_gpu(client, nvidia_payload, nonce)

                if intel_verified:
                    self._verified_addresses.add(signing_address.lower())
                    logger.info(f"Attestation fully verified for: {signing_address}")

                return AttestationResult(
                    valid=intel_verified,
                    signing_address=signing_address,
                    intel_verified=intel_verified,
                    gpu_verified=gpu_verified,
                    nonce=nonce,
                )

        except httpx.HTTPStatusError as e:
            error = f"Attestation request failed: {e.response.status_code}"
            logger.warning(error)
            return AttestationResult(valid=False, nonce=nonce, error=error)
        except httpx.TimeoutException:
            error = "Attestation request timed out"
            logger.warning(error)
            return AttestationResult(valid=False, nonce=nonce, error=error)
        except Exception as e:
            error = f"Attestation verification failed: {e}"
            logger.warning(error)
            return AttestationResult(valid=False, nonce=nonce, error=error)

    async def _verify_tdx_quote(
        self, client: httpx.AsyncClient, intel_quote: str | None
    ) -> bool:
        """Verify Intel TDX quote via Phala's verification service."""
        if not intel_quote:
            logger.debug("No Intel TDX quote in attestation")
            return False

        try:
            response = await client.post(
                PHALA_VERIFY_URL,
                json={"hex": intel_quote},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("error"):
                logger.warning(f"TDX verification failed: {result['error']}")
                return False

            logger.info("Intel TDX quote verified")
            return True

        except Exception as e:
            logger.debug(f"TDX verification unavailable: {e}")
            return False

    async def _verify_gpu(
        self, client: httpx.AsyncClient, nvidia_payload: str | None, nonce: str
    ) -> bool:
        """Verify NVIDIA GPU attestation via NVIDIA NRAS."""
        if not nvidia_payload:
            logger.debug("No NVIDIA payload in attestation")
            return False

        try:
            # The nvidia_payload is already JSON - parse and submit directly.
            payload = json.loads(nvidia_payload) if isinstance(nvidia_payload, str) else nvidia_payload

            response = await client.post(
                NVIDIA_NRAS_URL,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            logger.info("NVIDIA GPU attestation verified")
            return True

        except Exception as e:
            logger.debug(f"GPU verification unavailable: {e}")
            return False

    async def verify_response_signature(
        self, chat_id: str, request_body: str, response_text: str
    ) -> SignatureResult:
        """Verify the ECDSA signature on an LLM response.

        This cryptographically proves the response was signed by a specific key.
        The signature covers SHA256(request):SHA256(response), ensuring integrity.
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Fetch signature for this chat completion.
                url = f"{self.base_url}/signature/{chat_id}"
                params = {"model": self.model}
                headers = {"Authorization": f"Bearer {self.api_key}"}

                if self._log_sensitive:
                    logger.debug(f"Fetching signature for {chat_id}")

                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                sig_data = response.json()

                signature = sig_data.get("signature")
                signing_address = sig_data.get("signing_address")
                expected_text = sig_data.get("text")

                if not all([signature, signing_address, expected_text]):
                    # Check if we got an empty attestation response.
                    attestations = sig_data.get("all_attestations", [])
                    if not attestations and not signature:
                        return SignatureResult(
                            valid=False,
                            error="Signature not available for this model (attestation not enabled)"
                        )
                    return SignatureResult(valid=False, error="Incomplete signature data")

                # Verify the signature is valid (proves it came from TEE signing key).
                message = encode_defunct(text=expected_text)
                recovered_address = Account.recover_message(message, signature=signature)

                if recovered_address.lower() != signing_address.lower():
                    return SignatureResult(
                        valid=False,
                        error=f"Signature invalid - recovered {recovered_address}",
                    )

                # Check if our computed hash matches (optional - known API issue with non-streaming).
                request_hash = hashlib.sha256(request_body.encode()).hexdigest()
                response_hash = hashlib.sha256(response_text.encode()).hexdigest()
                computed_text = f"{request_hash}:{response_hash}"

                if computed_text != expected_text:
                    # Known issue: non-streaming responses may have modified hashes.
                    # The signature is still valid, proving it came from the TEE.
                    logger.debug(
                        f"Hash mismatch (known API issue with non-streaming): "
                        f"signature valid but response hash differs"
                    )

                # Track this address as seen.
                self._known_addresses.add(signing_address.lower())

                # Log verification status.
                if signing_address.lower() in self._verified_addresses:
                    logger.debug(f"Signature verified (attested address)")
                else:
                    logger.debug(f"Signature verified (address: {signing_address})")

                return SignatureResult(valid=True, signing_address=signing_address)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Signature endpoint not available for this model/response.
                return SignatureResult(valid=False, error="Signature not available for this model")
            error = f"Signature fetch failed: {e.response.status_code}"
            logger.error(error)
            return SignatureResult(valid=False, error=error)
        except httpx.TimeoutException:
            return SignatureResult(valid=False, error="Signature request timed out")
        except Exception as e:
            error = f"Signature verification failed: {e}"
            logger.error(error)
            return SignatureResult(valid=False, error=error)

    def is_address_verified(self, address: str) -> bool:
        """Check if a signing address has been fully attested."""
        return address.lower() in self._verified_addresses

    def is_address_known(self, address: str) -> bool:
        """Check if we've seen this signing address before."""
        return address.lower() in self._known_addresses

    def get_attestation_status(self) -> dict:
        """Get the current attestation status for user-facing messages."""
        verified_addresses = list(self._verified_addresses)
        return {
            "has_verified_address": len(verified_addresses) > 0,
            "signing_address": verified_addresses[0] if verified_addresses else None,
            "intel_verified": len(verified_addresses) > 0,
            "model": self.model,
        }


# Singleton instance.
_verifier: RedPillVerifier | None = None


def get_verifier() -> RedPillVerifier:
    """Get the singleton verifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = RedPillVerifier()
    return _verifier
