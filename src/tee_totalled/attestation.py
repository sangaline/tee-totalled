"""dstack attestation helpers with development fallback."""

import logging
from typing import Any

from .config import get_settings
from .verification import get_verifier

logger = logging.getLogger(__name__)

# Try to import dstack-sdk, fall back gracefully if not available.
try:
    from dstack_sdk import DstackClient

    DSTACK_SDK_AVAILABLE = True
except ImportError:
    DSTACK_SDK_AVAILABLE = False
    logger.debug("dstack-sdk not installed, using fallback attestation")


class AttestationClient:
    """Client for dstack attestation via the official SDK."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: Any = None

        if DSTACK_SDK_AVAILABLE and self.settings.dstack_available:
            try:
                self._client = DstackClient()
                logger.info("dstack SDK client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize dstack SDK client: {e}")

    def get_quote(self, report_data: str = "") -> dict[str, Any]:
        """Get a TDX attestation quote."""
        if self._client is None:
            return self._dev_quote()

        try:
            # Use the official SDK method.
            report_bytes = report_data.encode() if report_data else b""
            result = self._client.get_quote(report_bytes)
            return {
                "status": "verified",
                "quote": result.quote,
                "event_log": result.event_log,
            }
        except Exception as e:
            logger.warning(f"Failed to get TDX quote: {e}")
            return self._dev_quote()

    def get_info(self) -> dict[str, Any]:
        """Get application info from the dstack service."""
        if self._client is None:
            return self._dev_info()

        try:
            result = self._client.info()
            return {
                "status": "verified",
                "app_id": result.app_id,
                "instance_id": result.instance_id,
                "tcb_info": result.tcb_info,
                "tee_available": True,
                "github_url": "https://github.com/sangaline/tee-totalled/",
            }
        except Exception as e:
            logger.warning(f"Failed to get app info: {e}")
            return self._dev_info()

    def is_reachable(self) -> bool:
        """Check if the dstack service is reachable."""
        if self._client is None:
            return False

        try:
            return self._client.is_reachable()
        except Exception:
            return False

    def _dev_quote(self) -> dict[str, Any]:
        """Return mock quote data for development."""
        return {
            "status": "development_mode",
            "message": "Running in development mode without TEE attestation.",
            "quote": None,
            "event_log": None,
        }

    def _dev_info(self) -> dict[str, Any]:
        """Return mock info for development."""
        return {
            "status": "development_mode",
            "app_name": "tee-totalled",
            "version": "0.1.0",
            "environment": "development",
            "tee_available": False,
            "github_url": "https://github.com/sangaline/tee-totalled/",
        }

    def is_available(self) -> bool:
        """Check if attestation is available."""
        return self._client is not None and self.is_reachable()


def get_attestation_message() -> str:
    """Get a user-friendly attestation status message combining dstack and RedPill status."""
    client = AttestationClient()
    verifier = get_verifier()
    redpill_status = verifier.get_attestation_status()

    # Build RedPill LLM attestation section.
    if redpill_status["has_verified_address"]:
        llm_section = (
            "LLM responses are cryptographically verified to come from "
            f"Intel TDX hardware.\n"
            f"Signing Address: `{redpill_status['signing_address']}`\n"
            f"Model: `{redpill_status['model']}`"
        )
    else:
        llm_section = (
            "LLM attestation pending verification.\n"
            f"Model: `{redpill_status['model']}`"
        )

    # Build dstack bot attestation section.
    if client.is_available():
        try:
            info = client.get_info()
            app_id = info.get("app_id", "unknown")
            bot_section = (
                f"Bot is running in dstack TEE.\n"
                f"App ID: `{app_id}`\n"
                f"Verify: https://trust.phala.com/app/{app_id}"
            )
        except Exception as e:
            logger.error(f"Error getting dstack attestation info: {e}")
            bot_section = "Bot dstack attestation unavailable."
    else:
        bot_section = "Bot running in development mode (no dstack TEE)."

    return (
        "🔐 *TEE Attestation Status*\n\n"
        f"*LLM (RedPill Confidential AI):*\n{llm_section}\n\n"
        f"*Bot Infrastructure:*\n{bot_section}\n\n"
        "Use /verify to perform a fresh attestation check with your own nonce.\n\n"
        f"Source: https://github.com/sangaline/tee-totalled/"
    )


# Singleton instance.
_client: AttestationClient | None = None


def get_attestation_client() -> AttestationClient:
    """Get the singleton attestation client instance."""
    global _client
    if _client is None:
        _client = AttestationClient()
    return _client
