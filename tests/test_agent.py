from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import agent as agent_module
from agent import (
    Agent,
    ClaudeClient,
    LLMConnectionError,
    OllamaClient,
    OllamaConnectionError,
    ToolCallEvent,
    ToolResultEvent,
    TokenEvent,
    build_llm_client,
)
from memory import VaultMemory


class ScriptedClient:
    """A fake LLM client that returns a scripted sequence of raw
    responses, one per call to .chat(), so the tool-calling loop can
    be exercised without a real Ollama/Claude backend.
    """

    def __init__(self, script: list[str]):
        self._script = list(script)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self._script.pop(0)

    def complete_text(self, prompt):
        # Used by the automatic memory-creation step; "none" means
        # "nothing worth remembering" so tests don't trigger writes.
        return '{"tool": "none"}'


@pytest.fixture
def agent(vault_dir):
    return Agent(client=ScriptedClient(["placeholder"]), memory=VaultMemory())


class TestToolLoop:
    def test_final_text_response_returned_directly(self, vault_dir):
        client = ScriptedClient(["Hello there, nothing to look up."])
        a = Agent(client=client, memory=VaultMemory())
        answer = a.run_turn("hi")
        assert answer == "Hello there, nothing to look up."
        assert client.calls == 1

    def test_tool_call_then_final_answer(self, vault_dir):
        VaultMemory().write("facts/rust.md", "Rust is a systems programming language.")
        client = ScriptedClient(
            [
                json.dumps({"tool": "read", "file": "facts/rust.md"}),
                "Rust is a systems programming language, as I just read.",
            ]
        )
        a = Agent(client=client, memory=VaultMemory())
        answer = a.run_turn("what is rust?")
        assert answer == "Rust is a systems programming language, as I just read."
        assert client.calls == 2

    def test_llm_connection_error_surfaces_as_friendly_message(self, vault_dir):
        class FailingClient:
            def chat(self, messages):
                raise OllamaConnectionError("Ollama is offline")

            def complete_text(self, prompt):
                return '{"tool": "none"}'

        a = Agent(client=FailingClient(), memory=VaultMemory())
        answer = a.run_turn("hi")
        assert "couldn't reach the language model" in answer

    def test_max_iterations_reached_returns_fallback_message(self, vault_dir, config_override):
        config_override(agent_module, max_tool_iterations=2)
        client = ScriptedClient(
            [
                json.dumps({"tool": "list_files"}),
                json.dumps({"tool": "list_files"}),
            ]
        )
        a = Agent(client=client, memory=VaultMemory())
        answer = a.run_turn("loop forever")
        assert "didn't reach a final answer" in answer


class StreamingScriptedClient:
    """Like ScriptedClient, but implements chat_stream so the tool
    loop's streaming path (and its on_event callback) can be exercised
    without a real backend. Each scripted response is streamed back
    one character at a time.
    """

    def __init__(self, script: list[str]):
        self._script = list(script)
        self.calls = 0

    def chat_stream(self, messages, on_token):
        self.calls += 1
        text = self._script.pop(0)
        for char in text:
            on_token(char)
        return text

    def complete_text(self, prompt):
        return '{"tool": "none"}'


class TestStreamingToolLoop:
    def test_token_events_reassemble_to_final_answer(self, vault_dir):
        client = StreamingScriptedClient(["Hello, streamed world."])
        a = Agent(client=client, memory=VaultMemory())
        events: list = []
        answer = a.run_turn("hi", on_event=events.append)
        assert answer == "Hello, streamed world."
        assert all(isinstance(e, TokenEvent) for e in events)
        assert "".join(e.text for e in events) == "Hello, streamed world."

    def test_tool_call_and_result_events_emitted(self, vault_dir):
        VaultMemory().write("facts/a.md", "some fact")
        client = StreamingScriptedClient(
            [
                json.dumps({"tool": "read", "file": "facts/a.md"}),
                "Here's what I found.",
            ]
        )
        a = Agent(client=client, memory=VaultMemory())
        events: list = []
        answer = a.run_turn("what's in facts/a.md?", on_event=events.append)
        assert answer == "Here's what I found."

        tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_call_events) == 1
        assert tool_call_events[0].tool == "read"
        assert tool_call_events[0].args == {"file": "facts/a.md"}
        assert len(tool_result_events) == 1
        assert "some fact" in tool_result_events[0].result

    def test_no_on_event_still_works(self, vault_dir):
        client = StreamingScriptedClient(["fine without a callback"])
        a = Agent(client=client, memory=VaultMemory())
        assert a.run_turn("hi") == "fine without a callback"

    def test_non_streaming_client_falls_back_to_chat(self, vault_dir):
        # ScriptedClient (defined above) has no chat_stream method --
        # the tool loop must not assume every client can stream.
        client = ScriptedClient(["a plain non-streaming answer"])
        a = Agent(client=client, memory=VaultMemory())
        answer = a.run_turn("hi", on_event=lambda e: None)
        assert answer == "a plain non-streaming answer"
        assert client.calls == 1


class TestOllamaClientChatStream:
    def test_streams_and_reassembles_ndjson_chunks(self, monkeypatch):
        lines = [
            json.dumps({"message": {"content": "Hel"}, "done": False}).encode(),
            json.dumps({"message": {"content": "lo"}, "done": False}).encode(),
            json.dumps({"done": True}).encode(),
        ]

        class FakeStreamResponse:
            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(lines)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        import requests

        monkeypatch.setattr(
            requests, "post", lambda *a, **k: FakeStreamResponse() if k.get("stream") else None
        )

        client = OllamaClient()
        received: list[str] = []
        result = client.chat_stream([{"role": "user", "content": "hi"}], received.append)
        assert result == "Hello"
        assert received == ["Hel", "lo"]

    def test_connection_failure_raises(self, monkeypatch):
        import requests

        def raise_error(*args, **kwargs):
            raise requests.RequestException("offline")

        monkeypatch.setattr(requests, "post", raise_error)
        client = OllamaClient()
        with pytest.raises(OllamaConnectionError):
            client.chat_stream([{"role": "user", "content": "hi"}], lambda t: None)


class ScriptedMemoryClient:
    """A client whose .chat() always gives a plain final answer (so
    run_turn's tool loop finishes immediately), and whose
    .complete_text() -- used only by the memory-creation step --
    returns a scripted sequence of responses.
    """

    def __init__(self, memory_script: list[str]):
        self._memory_script = list(memory_script)
        self.complete_text_calls: list[str] = []

    def chat(self, messages):
        return "Final answer, no tool calls needed."

    def complete_text(self, prompt):
        self.complete_text_calls.append(prompt)
        return self._memory_script.pop(0)


class TestAutomaticMemoryCreation:
    def test_saves_multiple_distinct_facts_in_one_turn(self, vault_dir):
        client = ScriptedMemoryClient(
            [
                json.dumps({"tool": "write", "file": "facts/a.md", "content": "Fact A"}),
                json.dumps({"tool": "write", "file": "facts/b.md", "content": "Fact B"}),
                '{"tool": "none"}',
            ]
        )
        a = Agent(client=client, memory=VaultMemory())
        a.run_turn("tell me about A and B")

        assert VaultMemory().read("facts/a.md") == "Fact A"
        assert VaultMemory().read("facts/b.md") == "Fact B"
        # Stopped as soon as the model said "none" -- did not keep
        # going until the configured cap.
        assert len(client.complete_text_calls) == 3

    def test_stops_at_max_memory_writes_per_turn(self, vault_dir, config_override):
        config_override(agent_module, max_memory_writes_per_turn=2)
        client = ScriptedMemoryClient(
            [
                json.dumps({"tool": "write", "file": "facts/a.md", "content": "Fact A"}),
                json.dumps({"tool": "write", "file": "facts/b.md", "content": "Fact B"}),
                json.dumps({"tool": "write", "file": "facts/c.md", "content": "Fact C"}),
            ]
        )
        a = Agent(client=client, memory=VaultMemory())
        a.run_turn("tell me about A, B, and C")

        assert VaultMemory().read("facts/a.md") == "Fact A"
        assert VaultMemory().read("facts/b.md") == "Fact B"
        # The cap is 2 writes -- the third scripted response is never
        # even requested.
        assert len(client.complete_text_calls) == 2

    def test_already_saved_summaries_are_relayed_to_the_next_prompt(self, vault_dir):
        client = ScriptedMemoryClient(
            [
                json.dumps({"tool": "write", "file": "facts/a.md", "content": "Fact A"}),
                '{"tool": "none"}',
            ]
        )
        a = Agent(client=client, memory=VaultMemory())
        a.run_turn("tell me about A")

        second_prompt = client.complete_text_calls[1]
        assert "facts/a.md" in second_prompt
        assert "Already saved" in second_prompt

    def test_no_facts_worth_saving_writes_nothing(self, vault_dir):
        client = ScriptedMemoryClient(['{"tool": "none"}'])
        a = Agent(client=client, memory=VaultMemory())
        a.run_turn("just chatting")
        assert VaultMemory().list_files("facts") == []


class TestBuildLlmClient:
    def test_defaults_to_ollama(self, config_override):
        config_override(agent_module, llm_backend="ollama")
        client = build_llm_client()
        assert type(client).__name__ == "OllamaClient"

    def test_selects_claude_backend(self, config_override):
        config_override(agent_module, llm_backend="claude")
        client = build_llm_client()
        assert isinstance(client, ClaudeClient)


class TestClaudeClient:
    def test_splits_system_message_out_of_conversation(self):
        from agent import _split_system_message

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        system, rest = _split_system_message(messages)
        assert system == "You are helpful."
        assert rest == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_missing_anthropic_package_raises_connection_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("no anthropic installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        client = ClaudeClient(model="claude-opus-4-8")
        with pytest.raises(LLMConnectionError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_refusal_stop_reason_raises_connection_error(self, monkeypatch):
        fake_response = SimpleNamespace(stop_reason="refusal", content=[])

        class FakeMessages:
            def create(self, **kwargs):
                return fake_response

        class FakeAnthropicClient:
            def __init__(self, *a, **k):
                self.messages = FakeMessages()

        fake_anthropic_module = SimpleNamespace(
            Anthropic=FakeAnthropicClient,
            APIConnectionError=Exception,
            APIStatusError=Exception,
        )

        import sys

        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

        client = ClaudeClient(model="claude-opus-4-8")
        with pytest.raises(LLMConnectionError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_successful_response_joins_text_blocks(self, monkeypatch):
        text_block = SimpleNamespace(type="text", text="Hello ")
        text_block_2 = SimpleNamespace(type="text", text="world.")
        fake_response = SimpleNamespace(stop_reason="end_turn", content=[text_block, text_block_2])

        class FakeMessages:
            def create(self, **kwargs):
                return fake_response

        class FakeAnthropicClient:
            def __init__(self, *a, **k):
                self.messages = FakeMessages()

        fake_anthropic_module = SimpleNamespace(
            Anthropic=FakeAnthropicClient,
            APIConnectionError=Exception,
            APIStatusError=Exception,
        )

        import sys

        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

        client = ClaudeClient(model="claude-opus-4-8")
        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "Hello world."

    def test_chat_stream_yields_tokens_and_returns_final_text(self, monkeypatch):
        text_block = SimpleNamespace(type="text", text="Hello world.")
        final_message = SimpleNamespace(stop_reason="end_turn", content=[text_block])

        class FakeStreamContext:
            def __init__(self):
                self.text_stream = iter(["Hello ", "world."])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return final_message

        class FakeMessages:
            def stream(self, **kwargs):
                return FakeStreamContext()

        class FakeAnthropicClient:
            def __init__(self, *a, **k):
                self.messages = FakeMessages()

        fake_anthropic_module = SimpleNamespace(
            Anthropic=FakeAnthropicClient,
            APIConnectionError=Exception,
            APIStatusError=Exception,
        )

        import sys

        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

        client = ClaudeClient(model="claude-opus-4-8")
        received: list[str] = []
        result = client.chat_stream([{"role": "user", "content": "hi"}], received.append)
        assert received == ["Hello ", "world."]
        assert result == "Hello world."

    def test_chat_stream_refusal_raises_connection_error(self, monkeypatch):
        final_message = SimpleNamespace(stop_reason="refusal", content=[])

        class FakeStreamContext:
            def __init__(self):
                self.text_stream = iter([])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return final_message

        class FakeMessages:
            def stream(self, **kwargs):
                return FakeStreamContext()

        class FakeAnthropicClient:
            def __init__(self, *a, **k):
                self.messages = FakeMessages()

        fake_anthropic_module = SimpleNamespace(
            Anthropic=FakeAnthropicClient,
            APIConnectionError=Exception,
            APIStatusError=Exception,
        )

        import sys

        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

        client = ClaudeClient(model="claude-opus-4-8")
        with pytest.raises(LLMConnectionError):
            client.chat_stream([{"role": "user", "content": "hi"}], lambda t: None)
