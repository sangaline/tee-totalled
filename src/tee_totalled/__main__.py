"""Entry point for the TeeTotalled bot."""

import asyncio
import logging
import sys

from .bot import create_application
from .config import get_settings, setup_logging
from .verification import get_verifier

logger = logging.getLogger(__name__)


async def verify_redpill_attestation() -> bool:
    """Verify RedPill TEE attestation at startup.

    This is REQUIRED for the bot to start. We must cryptographically verify
    that the LLM is running in a genuine TEE before accepting user messages.
    """
    logger.info("Verifying RedPill TEE attestation...")
    verifier = get_verifier()

    result = await verifier.verify_attestation()

    if not result.valid:
        logger.error(f"TEE attestation verification FAILED: {result.error}")
        logger.error("The bot cannot start without verified TEE attestation.")
        return False

    logger.info(f"TEE attestation VERIFIED")
    logger.info(f"  Signing address: {result.signing_address}")
    logger.info(f"  Intel TDX: {'verified' if result.intel_verified else 'FAILED'}")
    logger.info(f"  NVIDIA GPU: {'verified' if result.gpu_verified else 'not available'}")

    if not result.intel_verified:
        logger.error("Intel TDX verification is required but failed.")
        return False

    return True


async def main() -> None:
    """Run the bot."""
    setup_logging()
    settings = get_settings()

    logger.info(f"Starting TeeTotalledBot in {settings.tee_env.value} mode")
    logger.info(f"Using LLM: {settings.llm_model} at {settings.llm_base_url}")

    # REQUIRED: Verify RedPill TEE attestation before starting.
    if not await verify_redpill_attestation():
        logger.error("Exiting due to failed TEE attestation verification.")
        sys.exit(1)

    if settings.dstack_available:
        logger.info("dstack attestation available")
    else:
        logger.info("dstack attestation not available (development mode)")

    application = create_application()

    # Use polling for now (webhooks would need a public URL).
    logger.info("Starting bot with polling...")
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)

    # Run until interrupted.
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
