"""
agent.py
========

The LLM clients (Ollama and, optionally, Claude) and the agent loop
that ties memory retrieval, tool calling, and conversation history
together.

All communication with a language model happens through this module
-- no other module talks to Ollama or the Anthropic API directly.
Which backend is used is controlled by CONFIG.llm_backend; both
clients expose the same chat()/chat_stream()/complete_text() interface
so agent.py doesn't need to care which one is active.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Union

import plugins
import sysinfo
from config import CONFIG
from conversation import ConversationManager
from memory import VaultMemory
from parser import ParsedResponse, parse_llm_response
from prompts import (
    MEMORY_CONTEXT_TEMPLATE,
    MEMORY_CREATION_PROMPT,
    build_current_context,
    build_full_system_prompt,
    build_memory_block,
    build_resources_context,
)
from tools import ToolExecutor

logger = logging.getLogger("assistant.agent")


@dataclass
class TokenEvent:
    """A chunk of raw model output as it's generated. Emitted for
    every LLM call, whether that call turns out to produce a tool
    call or a final answer -- callers that want to distinguish a live
    "thinking" preview from a tool call should watch for the
    ToolCallEvent that follows instead of trying to sniff the tokens.
    """

    text: str


@dataclass
class ToolCallEvent:
    """A tool call the model just requested, about to be executed."""

    tool: str
    args: dict = field(default_factory=dict)


@dataclass
class ToolResultEvent:
    """The text result of executing a ToolCallEvent's tool call."""

    tool: str
    result: str


AgentEvent = Union[TokenEvent, ToolCallEvent, ToolResultEvent]
EventCallback = Callable[[AgentEvent], None]


class LLMConnectionError(Exception):
    """Raised when the configured LLM backend cannot be reached or errors.

    Common base for OllamaConnectionError and ClaudeConnectionError so
    callers (Agent, main.py) can handle either backend without caring
    which one is active.
    """


class OllamaConnectionError(LLMConnectionError):
    """Raised when Ollama cannot be reached or returns an error."""


class ClaudeConnectionError(LLMConnectionError):
    """Raised when the Claude API cannot be reached, errors, or declines."""


class OllamaClient:
    """Thin wrapper around Ollama's local chat API.

    This is one of two supported LLM backends -- the rest of the
    codebase should never build an HTTP request to Ollama directly.
    """

    def __init__(self, model: str | None = None, host: str | None = None, temperature: float | None = None) -> None:
        self._model = model or CONFIG.model
        self._host = (host or CONFIG.ollama_host).rstrip("/")
        self._temperature = CONFIG.temperature if temperature is None else temperature

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a full chat message list to Ollama and return the
        assistant's reply text.

        Raises OllamaConnectionError if the request fails or Ollama is
        offline, so callers can handle it gracefully (see agent's
        run_turn, which surfaces a friendly error to the user).
        """
        try:
            import requests
        except ImportError as exc:
            raise OllamaConnectionError(
                "The 'requests' package is required to talk to Ollama. "
                "Install it with `pip install requests`."
            ) from exc

        url = f"{self._host}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._temperature},
        }

        logger.debug("Sending %d messages to Ollama model=%s", len(messages), self._model)
        try:
            response = requests.post(url, json=payload, timeout=CONFIG.request_timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self._host}. Is it running? ({exc})"
            ) from exc

        try:
            data = response.json()
            return data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaConnectionError(f"Unexpected response shape from Ollama: {exc}") from exc

    def chat_stream(self, messages: list[dict[str, str]], on_token: Callable[[str], None]) -> str:
        """Like chat(), but calls on_token(chunk) as each piece of the
        reply arrives and returns the full concatenated text at the
        end.

        This isn't just for a live "thinking" preview: streaming reads
        via requests' timeout apply per chunk received, not to the
        response as a whole, so a slow model that would blow past
        CONFIG.request_timeout_seconds on a single non-streaming read
        (see chat()) can still succeed here as long as no single gap
        between tokens exceeds the timeout.
        """
        try:
            import requests
        except ImportError as exc:
            raise OllamaConnectionError(
                "The 'requests' package is required to talk to Ollama. "
                "Install it with `pip install requests`."
            ) from exc

        url = f"{self._host}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self._temperature},
        }

        logger.debug("Streaming %d messages to Ollama model=%s", len(messages), self._model)
        parts: list[str] = []
        try:
            with requests.post(
                url, json=payload, timeout=CONFIG.request_timeout_seconds, stream=True
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed streaming chunk from Ollama: %r", line)
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        parts.append(content)
                        on_token(content)
                    if chunk.get("done"):
                        break
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self._host}. Is it running? ({exc})"
            ) from exc

        return "".join(parts)

    def complete_text(self, prompt: str) -> str:
        """Convenience wrapper for a single-turn, system-prompt-free
        completion (used for summarization and memory-creation
        prompts where no conversation context is needed).
        """
        return self.chat([{"role": "user", "content": prompt}])


def _split_system_message(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    """Split Ollama-style messages (system role inline in the list)
    into Claude's shape: a separate top-level system string plus a
    user/assistant-only message list.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    other = [m for m in messages if m["role"] != "system"]
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, other


class ClaudeClient:
    """Wrapper around the Anthropic Claude API.

    This is the second supported LLM backend -- opt in via
    CONFIG.llm_backend="claude" (env ASSISTANT_LLM_BACKEND). Unlike
    OllamaClient, this sends conversation content to Anthropic's
    servers rather than keeping everything on-machine, so it is not
    the default. The API key is resolved by the `anthropic` SDK from
    the standard ANTHROPIC_API_KEY env var (or an `ant auth login`
    profile) -- it is never read or stored by this module or CONFIG.
    """

    def __init__(self, model: str | None = None, max_tokens: int | None = None) -> None:
        self._model = model or CONFIG.anthropic_model
        self._max_tokens = max_tokens or CONFIG.anthropic_max_tokens

    def chat(self, messages: list[dict[str, str]]) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ClaudeConnectionError(
                "The 'anthropic' package is required to use the Claude backend. "
                "Install it with `pip install anthropic`."
            ) from exc

        system_prompt, claude_messages = _split_system_message(messages)

        logger.debug("Sending %d messages to Claude model=%s", len(claude_messages), self._model)
        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=claude_messages,
            )
        except anthropic.APIConnectionError as exc:
            raise ClaudeConnectionError(f"Could not reach the Claude API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ClaudeConnectionError(f"Claude API error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise ClaudeConnectionError("Claude declined to respond to this request.")

        return "".join(block.text for block in response.content if block.type == "text")

    def chat_stream(self, messages: list[dict[str, str]], on_token: Callable[[str], None]) -> str:
        """Like chat(), but calls on_token(chunk) as text streams in.
        See python/claude-api/streaming.md: `messages.stream()` is the
        recommended helper -- it accumulates state and exposes
        `text_stream` / `get_final_message()`.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise ClaudeConnectionError(
                "The 'anthropic' package is required to use the Claude backend. "
                "Install it with `pip install anthropic`."
            ) from exc

        system_prompt, claude_messages = _split_system_message(messages)

        logger.debug("Streaming %d messages to Claude model=%s", len(claude_messages), self._model)
        try:
            client = anthropic.Anthropic()
            with client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=claude_messages,
            ) as stream:
                for text in stream.text_stream:
                    on_token(text)
                final_message = stream.get_final_message()
        except anthropic.APIConnectionError as exc:
            raise ClaudeConnectionError(f"Could not reach the Claude API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ClaudeConnectionError(f"Claude API error: {exc}") from exc

        if final_message.stop_reason == "refusal":
            raise ClaudeConnectionError("Claude declined to respond to this request.")

        return "".join(block.text for block in final_message.content if block.type == "text")

    def complete_text(self, prompt: str) -> str:
        """Convenience wrapper matching OllamaClient.complete_text."""
        return self.chat([{"role": "user", "content": prompt}])


def build_llm_client() -> OllamaClient | ClaudeClient:
    """Construct the LLM client selected by CONFIG.llm_backend."""
    if CONFIG.llm_backend == "claude":
        return ClaudeClient()
    return OllamaClient()


class Agent:
    """Orchestrates one user turn: retrieval, the tool-calling loop,
    conversation bookkeeping, and automatic memory creation.
    """

    def __init__(
        self,
        client: OllamaClient | ClaudeClient | None = None,
        memory: VaultMemory | None = None,
    ) -> None:
        self.client = client or build_llm_client()
        self.memory = memory or VaultMemory()
        self.tools = ToolExecutor(self.memory)
        self.conversation = ConversationManager(self.memory, summarizer=self.client.complete_text)
        self.loaded_plugins = plugins.register_loaded_plugins()

    def _build_messages(self, user_message: str) -> list[dict[str, str]]:
        """Assemble the full message list for this turn: system
        prompt, retrieved memory context, recent history, and the new
        user message.
        """
        system_prompt = build_full_system_prompt()
        system_prompt = system_prompt + "\n\n" + build_current_context(sysinfo.context_block())

        plugin_block = plugins.describe_loaded_plugins(self.loaded_plugins)
        if plugin_block:
            system_prompt = system_prompt + "\n\n" + plugin_block

        resources = self.memory.read_resources()
        if resources:
            resource_blocks = "\n".join(build_memory_block(path, content) for path, content in resources)
            system_prompt = system_prompt + "\n\n" + build_resources_context(resource_blocks)

        retrieved = self.memory.retrieve_context_for(user_message)
        if retrieved:
            blocks = "\n".join(build_memory_block(path, content) for path, content in retrieved)
            memory_context = MEMORY_CONTEXT_TEMPLATE.format(memory_blocks=blocks)
            system_prompt = system_prompt + "\n\n" + memory_context
            logger.info("Retrieved %d memory file(s) for context: %s", len(retrieved), [p for p, _ in retrieved])
        else:
            logger.info("No relevant memory files found for this message.")

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation.as_message_list())
        messages.append({"role": "user", "content": user_message})
        return messages

    def run_turn(self, user_message: str, on_event: EventCallback | None = None) -> str:
        """Run a full agent turn for one user message: retrieval, the
        tool-calling loop (bounded by CONFIG.max_tool_iterations), and
        returns the final natural-language answer.

        If on_event is given, it's called synchronously with
        TokenEvent/ToolCallEvent/ToolResultEvent as they happen (see
        the dataclasses above) -- front-ends use this to show a live
        "thinking" preview instead of just a spinner. Optional and
        backward compatible: omit it and this behaves exactly as
        before.
        """
        self.conversation.add_user_turn(user_message)
        messages = self._build_messages(user_message)

        final_answer = self._tool_loop(messages, on_event)

        self.conversation.add_assistant_turn(final_answer)
        self._maybe_create_memory()
        self.conversation.maybe_summarize()
        return final_answer

    def _tool_loop(self, messages: list[dict[str, str]], on_event: EventCallback | None = None) -> str:
        """Repeatedly call the LLM, executing any tool calls it makes,
        until it produces a final natural-language answer or the
        iteration limit is reached.
        """
        emit: EventCallback = on_event or (lambda event: None)
        can_stream = hasattr(self.client, "chat_stream")

        for iteration in range(CONFIG.max_tool_iterations):
            try:
                if can_stream:
                    raw_response = self.client.chat_stream(
                        messages, lambda token: emit(TokenEvent(token))
                    )
                else:
                    raw_response = self.client.chat(messages)
            except LLMConnectionError as exc:
                logger.error("LLM request failed: %s", exc)
                return f"I couldn't reach the language model: {exc}"

            parsed: ParsedResponse = parse_llm_response(raw_response)

            if not parsed.is_tool_call:
                return parsed.final_text or ""

            call = parsed.tool_call
            assert call is not None
            logger.info("Tool call requested (iteration %d): %s", iteration + 1, call.tool)
            emit(ToolCallEvent(call.tool, call.args))

            result_text = self.tools.execute(call)
            emit(ToolResultEvent(call.tool, result_text))

            # Feed the tool call and its result back into the
            # conversation so the model can decide what to do next.
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": f"[tool result]\n{result_text}"})

        logger.warning("Reached max_tool_iterations (%d) without a final answer.", CONFIG.max_tool_iterations)
        return (
            "I made several tool calls but didn't reach a final answer in time. "
            "Here's what I found so far -- feel free to ask me to continue."
        )

    def _maybe_create_memory(self) -> None:
        """After a turn, repeatedly ask the model whether anything is
        worth remembering, executing each write/append it requests,
        until it says nothing further is worth saving or
        CONFIG.max_memory_writes_per_turn is reached.

        This loops (rather than asking once) because the memory policy
        is deliberately permissive -- a single turn can easily contain
        several distinct facts worth their own note, and asking only
        once would silently drop everything but the first.
        """
        recent = self.conversation.as_message_list()[-2:]  # last user+assistant turn
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in recent)

        saved_summaries: list[str] = []
        for _ in range(CONFIG.max_memory_writes_per_turn):
            already_saved_section = ""
            if saved_summaries:
                already_saved_section = (
                    "Already saved so far this turn (do not repeat these):\n"
                    + "\n".join(f"- {s}" for s in saved_summaries)
                    + "\n\n"
                )
            prompt = MEMORY_CREATION_PROMPT.format(conversation=transcript, already_saved_section=already_saved_section)

            try:
                raw_response = self.client.complete_text(prompt)
            except LLMConnectionError as exc:
                logger.warning("Automatic memory creation skipped (LLM unreachable): %s", exc)
                return

            parsed = parse_llm_response(raw_response)
            if not parsed.is_tool_call:
                logger.debug("Memory-creation response was not a tool call; stopping.")
                return

            call = parsed.tool_call
            assert call is not None
            if call.tool == "none":
                logger.debug("Model decided nothing further was worth remembering this turn.")
                return
            if call.tool not in ("write", "append"):
                logger.debug("Ignoring unexpected tool %r from memory-creation prompt.", call.tool)
                return

            result = self.tools.execute(call)
            logger.info("Automatic memory creation: %s", result)
            saved_summaries.append(f"{call.tool} {call.args.get('file', '?')}")

        # Only reached if the loop ran to completion without an early
        # `return` above -- i.e. the model kept finding more to save
        # right up to the cap.
        logger.debug("Reached max_memory_writes_per_turn (%d).", CONFIG.max_memory_writes_per_turn)
