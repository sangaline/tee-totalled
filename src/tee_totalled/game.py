"""Game state management supporting multiple concurrent games."""

import logging
import secrets
import time
from dataclasses import dataclass, field

from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Submission:
    """A user's submission to the game."""

    user_id: int
    message: str
    score: int


@dataclass
class Game:
    """Represents an active game session."""

    game_id: str
    chat_id: int  # Where the game was started (group or DM).
    creator_id: int
    message_id: int  # The bot's message in the chat (for editing).
    started_at: float = field(default_factory=time.time)
    submissions: dict[int, Submission] = field(default_factory=dict)

    @property
    def participant_count(self) -> int:
        return len(self.submissions)

    @property
    def elapsed_seconds(self) -> int:
        return int(time.time() - self.started_at)

    @property
    def remaining_seconds(self) -> int:
        settings = get_settings()
        return max(0, settings.game_timeout_seconds - self.elapsed_seconds)

    @property
    def is_expired(self) -> bool:
        return self.remaining_seconds <= 0

    def get_scores(self) -> list[int]:
        """Get all submission scores."""
        return [s.score for s in self.submissions.values()]

    def get_participant_ids(self) -> list[int]:
        """Get all participant user IDs."""
        return list(self.submissions.keys())


class GameManager:
    """Manages multiple concurrent game sessions."""

    def __init__(self) -> None:
        self._games: dict[str, Game] = {}

    def start_game(self, chat_id: int, creator_id: int, message_id: int) -> Game:
        """Start a new game and return it."""
        game_id = secrets.token_urlsafe(8)
        game = Game(
            game_id=game_id,
            chat_id=chat_id,
            creator_id=creator_id,
            message_id=message_id,
        )
        self._games[game_id] = game
        logger.info(f"Started game {game_id} in chat {chat_id} by user {creator_id}")
        return game

    def end_game(self, game_id: str) -> Game | None:
        """End a specific game and return it for final processing."""
        game = self._games.pop(game_id, None)
        if game:
            logger.info(f"Ended game {game.game_id} with {game.participant_count} participants")
        return game

    def get_game_by_id(self, game_id: str) -> Game | None:
        """Get a game by its ID."""
        return self._games.get(game_id)

    def get_game_for_chat(self, chat_id: int) -> Game | None:
        """Find an active game in a specific chat."""
        for game in self._games.values():
            if game.chat_id == chat_id:
                return game
        return None

    def get_games_by_creator(self, user_id: int) -> list[Game]:
        """Find all active games created by a user."""
        return [g for g in self._games.values() if g.creator_id == user_id]

    def add_submission(self, game_id: str, user_id: int, message: str, score: int) -> bool:
        """Add or update a submission. Returns True if this improved their score."""
        game = self._games.get(game_id)
        if not game:
            return False

        existing = game.submissions.get(user_id)
        if existing and existing.score >= score:
            logger.debug(
                f"User {user_id} submission not updated (old: {existing.score}, new: {score})"
            )
            return False

        game.submissions[user_id] = Submission(user_id=user_id, message=message, score=score)
        if existing:
            logger.debug(f"User {user_id} improved score from {existing.score} to {score}")
        else:
            logger.debug(f"User {user_id} first submission with score {score}")
        return True

    def get_user_best_score(self, game_id: str, user_id: int) -> int | None:
        """Get a user's best score in a specific game."""
        game = self._games.get(game_id)
        if not game:
            return None
        submission = game.submissions.get(user_id)
        return submission.score if submission else None

    def can_stop_game(self, game_id: str, user_id: int) -> bool:
        """Check if the user is allowed to stop a specific game."""
        game = self._games.get(game_id)
        return game is not None and game.creator_id == user_id


# Singleton instance.
_manager: GameManager | None = None


def get_game_manager() -> GameManager:
    """Get the singleton game manager instance."""
    global _manager
    if _manager is None:
        _manager = GameManager()
    return _manager
