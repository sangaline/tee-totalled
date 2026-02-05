"""Telegram bot handlers."""

import asyncio
import logging

from telegram import Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .attestation import (
    GITHUB_URL,
    AttestationClient,
    get_attestation_message,
    get_trust_footer,
)
from .config import get_settings
from .game import Game, get_game_manager
from .histogram import format_stats_message, generate_histogram
from .llm import get_llm_client
from .verification import get_verifier

logger = logging.getLogger(__name__)


class TeeTotalledBot:
    """Main bot class managing handlers and state."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.game_manager = get_game_manager()
        self.llm_client = get_llm_client()
        self.bot_username: str | None = None
        # Per-game background tasks.
        self._update_tasks: dict[str, asyncio.Task] = {}
        self._timeout_tasks: dict[str, asyncio.Task] = {}

    async def post_init(self, application: Application) -> None:
        """Called after the application is initialized."""
        me = await application.bot.get_me()
        self.bot_username = me.username
        if not self.bot_username:
            logger.error("Failed to get bot username!")
        else:
            logger.info(f"Bot initialized as @{self.bot_username}")

    async def _ensure_bot_username(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Fetch bot username if not already set."""
        if not self.bot_username:
            me = await context.bot.get_me()
            self.bot_username = me.username
            logger.info(f"Bot username fetched on demand: @{self.bot_username}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command in groups or private chats."""
        if not update.effective_chat or not update.effective_user or not update.message:
            return

        # Private chat with deep link args means joining an existing game.
        if update.effective_chat.type == ChatType.PRIVATE and context.args:
            await self._handle_deep_link(update, context, context.args[0])
            return

        # Otherwise, start a new game (works in groups and private chats).
        await self._start_new_game(update, context)

    async def _start_new_game(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Start a new game in the current chat (group or private)."""
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        # Check if there's already a game in this chat.
        existing = self.game_manager.get_game_for_chat(chat_id)
        if existing:
            await update.message.reply_text(
                "A game is already active in this chat! Wait for it to finish "
                "or use /stop to end it."
            )
            return

        await self._ensure_bot_username(context)

        msg = await update.message.reply_text(
            "🎲 *Starting Trust Game...*",
            parse_mode=ParseMode.MARKDOWN,
        )

        game = self.game_manager.start_game(
            chat_id=chat_id,
            creator_id=user_id,
            message_id=msg.message_id,
        )

        deep_link = f"https://t.me/{self.bot_username}?start={game.game_id}"
        await self._update_game_message(context, game, deep_link)
        self._start_background_tasks(context, game, deep_link)

        # For DM-started games, auto-enroll the creator so they can submit immediately.
        if update.effective_chat.type == ChatType.PRIVATE:
            context.user_data["active_game_id"] = game.game_id

        logger.info(f"Game {game.game_id} started by user {user_id} in chat {chat_id}")

    async def _handle_deep_link(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, game_id: str
    ) -> None:
        """Handle a user clicking the deep link to participate."""
        if not update.message:
            return

        game = self.game_manager.get_game_by_id(game_id)
        if not game:
            await update.message.reply_text(
                "This game has ended or doesn't exist. Wait for a new game to start!"
            )
            return

        # Store the game ID in user data for message handling.
        context.user_data["active_game_id"] = game_id

        existing_score = self.game_manager.get_user_best_score(game_id, update.effective_user.id)
        if existing_score:
            await update.message.reply_text(
                f"Welcome back! Your current best score is {existing_score}/100.\n\n"
                "Send me a message to try to beat it! The more offensive, the higher "
                "the score. Your messages are private and will never be revealed.\n\n"
                + get_attestation_message(),
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await update.message.reply_text(
                "🎲 *Welcome to the Trust Game!*\n\n"
                "Send me the most offensive message you can think of. "
                "I'll score it from 1-100 based on offensiveness.\n\n"
                "This is a test of how much you trust TEE privacy guarantees. "
                "Your messages are NEVER revealed - only aggregate statistics are shared.\n\n"
                "Use /verify to independently verify the TEE attestation "
                "(you can include your own nonce, e.g. `/verify my secret nonce`).\n\n"
                + get_attestation_message(),
                parse_mode=ParseMode.MARKDOWN,
            )

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stop command to end a game (works in groups and DMs)."""
        if not update.effective_chat or not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        chat_type = update.effective_chat.type

        # Find the game to stop.
        if chat_type in (ChatType.GROUP, ChatType.SUPERGROUP):
            # In a group, find the game running in this group.
            game = self.game_manager.get_game_for_chat(chat_id)
        else:
            # In a DM, find any game created by this user.
            games = self.game_manager.get_games_by_creator(user_id)
            game = games[0] if games else None

        if not game:
            await update.message.reply_text("No active game found.")
            return

        if game.creator_id != user_id:
            await update.message.reply_text(
                "Only the person who started the game can stop it."
            )
            return

        await self._end_game(context, game.game_id, "stopped by creator")

    async def verify_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /verify command to perform fresh TEE attestation with an optional user nonce."""
        if not update.effective_chat or not update.message:
            return

        # User-supplied nonce is everything after the command.
        user_nonce = " ".join(context.args) if context.args else None
        nonce_display = f"`{user_nonce}`" if user_nonce else "_(auto-generated)_"

        await update.message.reply_text(
            f"Verifying TEE attestation with nonce: {nonce_display}\n"
            "This may take a few seconds...",
            parse_mode=ParseMode.MARKDOWN,
        )

        # Fresh RedPill attestation with the user's nonce.
        verifier = get_verifier()
        result = await verifier.verify_attestation(nonce=user_nonce)

        # Fresh dstack attestation.
        dstack = AttestationClient()
        dstack_info = dstack.get_info() if dstack.is_available() else None

        # Build the response.
        lines = ["*TEE Verification Report*\n"]
        if user_nonce:
            lines.append(f"Your nonce: `{user_nonce}`")
            lines.append(f"Hex nonce (SHA-256): `{result.nonce}`")
        else:
            lines.append(f"Nonce: `{result.nonce}`")
        lines.append("")

        # RedPill LLM section.
        lines.append("*LLM Inference (RedPill Confidential AI):*")
        if result.valid:
            lines.append("Intel TDX: VERIFIED")
            lines.append(f"NVIDIA GPU: {'VERIFIED' if result.gpu_verified else 'not checked'}")
            lines.append(f"Signing Address: `{result.signing_address}`")
            lines.append(f"Model: `{verifier.model}`")
            lines.append("")
            lines.append(
                "This proves the LLM runs on genuine Intel TDX hardware "
                "and each response is signed by the TEE's private key."
            )
        else:
            lines.append(f"Verification failed: {result.error}")

        lines.append("")

        # dstack bot section.
        lines.append("*Bot Infrastructure (dstack):*")
        if dstack_info and dstack_info.get("status") != "development_mode":
            app_id = dstack_info.get('app_id', 'unknown')
            lines.append(f"App ID: `{app_id}`")
            lines.append(f"Verify: https://trust.phala.com/app/{app_id}")
            lines.append("")
            lines.append(
                "This proves the bot code itself runs in a TEE and matches "
                "the open-source build."
            )
        else:
            lines.append(
                "Bot is running in development mode (no dstack TEE). "
                "In production, the bot code runs inside a TEE and you "
                "can verify the image hash matches the open-source build."
            )

        lines.append("")

        # How to verify independently.
        lines.append("*Independent Verification:*")
        lines.append(
            "You can verify the LLM attestation yourself by calling "
            "the RedPill API with your own nonce:"
        )
        lines.append(f"`GET {verifier.base_url}/attestation/report"
                     f"?model={verifier.model}&nonce=YOUR_NONCE`")
        lines.append("")
        lines.append(f"Source: {GITHUB_URL}")

        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.MARKDOWN,
        )

    async def handle_private_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle messages in private chat (game submissions)."""
        if not update.effective_chat or not update.effective_user or not update.message:
            return

        if update.effective_chat.type != ChatType.PRIVATE:
            return

        if not update.message.text:
            await update.message.reply_text("Please send a text message.")
            return

        # Check if user is in an active game.
        game_id = context.user_data.get("active_game_id")
        if not game_id:
            await update.message.reply_text(
                "You're not in an active game! Click a game link to join."
            )
            return

        game = self.game_manager.get_game_by_id(game_id)
        if not game:
            context.user_data.pop("active_game_id", None)
            await update.message.reply_text(
                "The game you were in has ended. Wait for a new game!"
            )
            return

        message_text = update.message.text
        user_id = update.effective_user.id

        if self.settings.is_development:
            logger.debug(f"Scoring message from user {user_id}: {message_text[:50]}...")

        await update.message.reply_text("🤔 Analyzing your message...")

        score = await self.llm_client.score_offensiveness(message_text)
        improved = self.game_manager.add_submission(game_id, user_id, message_text, score)

        trust_footer = get_trust_footer()
        if improved:
            response = (
                f"Your message scored *{score}/100* on the offensiveness scale! 🎯\n\n"
                f"This is your new best score. Can you be even more offensive?\n\n"
                f"{trust_footer}"
            )
        else:
            best_score = self.game_manager.get_user_best_score(game_id, user_id) or score
            response = (
                f"Your message scored *{score}/100*, but your best is still "
                f"*{best_score}/100*.\n\n"
                f"Try to beat your high score! Be more creative (or offensive).\n\n"
                f"{trust_footer}"
            )

        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

        if self.settings.is_development:
            logger.debug(f"User {user_id} scored {score}")

    async def _update_game_message(
        self, context: ContextTypes.DEFAULT_TYPE, game: Game, deep_link: str
    ) -> None:
        """Update the game message with current status."""
        remaining = game.remaining_seconds
        minutes = remaining // 60
        seconds = remaining % 60

        trust_footer = get_trust_footer()
        text = (
            f"🎲 *Trust Game Active!*\n\n"
            f"Click the link below to submit your most offensive message privately. "
            f"Messages are scored by AI and never revealed.\n\n"
            f"👉 [Join the game]({deep_link})\n\n"
            f"⏱️ Time remaining: {minutes}:{seconds:02d}\n"
            f"👥 Participants: {game.participant_count}\n\n"
            f"{trust_footer}\n"
            f"_Use /stop to end early (creator only)_"
        )

        try:
            await context.bot.edit_message_text(
                chat_id=game.chat_id,
                message_id=game.message_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
        except Exception as e:
            if "not modified" not in str(e).lower():
                logger.warning(f"Failed to update game message: {e}")

    def _start_background_tasks(
        self, context: ContextTypes.DEFAULT_TYPE, game: Game, deep_link: str
    ) -> None:
        """Start background tasks for a specific game."""
        game_id = game.game_id

        async def update_loop():
            try:
                while self.game_manager.get_game_by_id(game_id):
                    await asyncio.sleep(self.settings.message_update_interval)
                    current_game = self.game_manager.get_game_by_id(game_id)
                    if current_game:
                        await self._update_game_message(context, current_game, deep_link)
            except asyncio.CancelledError:
                pass

        async def timeout_handler():
            try:
                await asyncio.sleep(self.settings.game_timeout_seconds)
                if self.game_manager.get_game_by_id(game_id):
                    await self._end_game(context, game_id, "time expired")
            except asyncio.CancelledError:
                pass

        self._update_tasks[game_id] = asyncio.create_task(update_loop())
        self._timeout_tasks[game_id] = asyncio.create_task(timeout_handler())

    def _cancel_game_tasks(self, game_id: str) -> None:
        """Cancel background tasks for a specific game."""
        task = self._update_tasks.pop(game_id, None)
        if task:
            task.cancel()
        task = self._timeout_tasks.pop(game_id, None)
        if task:
            task.cancel()

    async def _end_game(
        self, context: ContextTypes.DEFAULT_TYPE, game_id: str, reason: str
    ) -> None:
        """End a game and post results to the game chat and all participants."""
        self._cancel_game_tasks(game_id)
        game = self.game_manager.end_game(game_id)
        if not game:
            return

        logger.info(f"Game {game.game_id} ended: {reason}")

        # Update the game message to show it ended.
        try:
            await context.bot.edit_message_text(
                chat_id=game.chat_id,
                message_id=game.message_id,
                text=f"🎲 *Game Ended* ({reason})\n\nProcessing results...",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.warning(f"Failed to update final message: {e}")

        scores = game.get_scores()
        if not scores:
            no_results = "🎲 *Game Over!*\n\nNo submissions were received. Maybe next time!"
            await context.bot.send_message(
                chat_id=game.chat_id, text=no_results, parse_mode=ParseMode.MARKDOWN
            )
            return

        # Generate results once, send to multiple chats.
        image_bytes, stats = None, None
        try:
            image_bytes, stats = generate_histogram(scores)
        except Exception as e:
            logger.error(f"Failed to generate histogram: {e}")

        reaction = None
        try:
            reaction = await self.llm_client.generate_moral_reaction(game.get_messages())
        except Exception as e:
            logger.error(f"Failed to generate moral reaction: {e}")

        attestation_msg = get_attestation_message()

        # Send results to the game's origin chat.
        await self._send_results(context, game.chat_id, image_bytes, stats, reaction, attestation_msg)

        # DM results to each participant individually.
        for user_id in game.get_participant_ids():
            # Skip if the game was started in this user's DM (they already see results).
            if user_id == game.chat_id:
                continue
            try:
                await self._send_results(
                    context, user_id, image_bytes, stats, reaction, attestation_msg
                )
            except Exception as e:
                logger.warning(f"Failed to DM results to user {user_id}: {e}")

    async def _send_results(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        image_bytes: bytes | None,
        stats: dict[str, float] | None,
        reaction: str | None,
        attestation_msg: str,
    ) -> None:
        """Send game results (histogram, stats, reflection, attestation) to a chat."""
        if image_bytes and stats:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=image_bytes,
                caption=format_stats_message(stats),
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id, text="📊 Results processing encountered an error."
            )

        if reaction:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🤔 *Moral Reflection*\n\n{reaction}",
                parse_mode=ParseMode.MARKDOWN,
            )

        await context.bot.send_message(
            chat_id=chat_id, text=attestation_msg, parse_mode=ParseMode.MARKDOWN
        )


def create_application() -> Application:
    """Create and configure the Telegram bot application."""
    settings = get_settings()
    bot = TeeTotalledBot()

    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(bot.post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", bot.start_command))
    application.add_handler(CommandHandler("stop", bot.stop_command))
    application.add_handler(CommandHandler("verify", bot.verify_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_private_message)
    )

    return application
