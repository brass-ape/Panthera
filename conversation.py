"""
conversation.py
================

Manages in-memory conversation history and periodic summarization of
old turns into the vault, so the assistant's context window doesn't
grow without bound over a long-running session.

This module deliberately does not import agent.py (which owns the
Ollama client) to avoid a circular import. Instead, `ConversationManager`
accepts any callable matching the `Summarizer` protocol, so agent.py
can inject its LLM client's `complete` method at construction time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from config import CONFIG
from memory import VaultMemory
from prompts import CONVERSATION_SUMMARY_PROMPT

logger = logging.getLogger("assistant.conversation")

# A Summarizer takes a plain-text prompt and returns the model's
# plain-text completion. agent.py's OllamaClient.complete_text (or
# similar) satisfies this signature.
Summarizer = Callable[[str], str]


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ConversationManager:
    """Holds recent conversation turns and rolls old ones into vault
    summaries once the history grows past a configured threshold.
    """

    def __init__(self, memory: VaultMemory, summarizer: Summarizer | None = None) -> None:
        self._memory = memory
        self._summarizer = summarizer
        self._turns: list[Turn] = []

    def add_user_turn(self, content: str) -> None:
        self._turns.append(Turn(role="user", content=content))

    def add_assistant_turn(self, content: str) -> None:
        self._turns.append(Turn(role="assistant", content=content))

    def as_message_list(self) -> list[dict[str, str]]:
        """Return the retained turns in the {"role", "content"} shape
        expected by the LLM chat API, limited to the configured
        history size (most recent turns only).
        """
        recent = self._turns[-CONFIG.conversation_history_size :]
        return [{"role": t.role, "content": t.content} for t in recent]

    def turn_count(self) -> int:
        return len(self._turns)

    def maybe_summarize(self) -> None:
        """If the history has grown past the configured threshold,
        summarize the oldest turns into vault/conversations/ and drop
        them from RAM, keeping only the most recent
        conversation_history_size turns in memory.
        """
        if len(self._turns) <= CONFIG.summarize_after_turns:
            return
        if self._summarizer is None:
            logger.debug("No summarizer configured; skipping summarization.")
            return

        keep = CONFIG.conversation_history_size
        to_summarize = self._turns[:-keep] if keep > 0 else self._turns
        if not to_summarize:
            return

        transcript = "\n".join(f"{t.role}: {t.content}" for t in to_summarize)
        prompt = CONVERSATION_SUMMARY_PROMPT.format(conversation=transcript)

        try:
            summary = self._summarizer(prompt).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Conversation summarization failed, keeping history as-is: %s", exc)
            return

        if not summary:
            logger.debug("Empty summary produced; skipping write.")
            return

        filename = f"conversations/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.md"
        header = f"# Conversation summary ({datetime.now().isoformat(timespec='seconds')})\n\n"
        self._memory.write(filename, header + summary + "\n")
        logger.info("Summarized %d old turns into %s", len(to_summarize), filename)

        self._turns = self._turns[-keep:] if keep > 0 else []
