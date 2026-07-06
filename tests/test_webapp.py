from __future__ import annotations

import json

import config as config_module
import webapp
from agent import OllamaConnectionError, ToolCallEvent, ToolResultEvent, TokenEvent
from config import CONFIG


class FakeAgent:
    """Stand-in for Agent so chat-endpoint tests don't need a real
    Ollama/Claude backend.
    """

    def __init__(self, reply="stub reply", raise_exc=None, events=None):
        self.reply = reply
        self.raise_exc = raise_exc
        self.events = events or []
        self.calls: list[str] = []

    def run_turn(self, message, on_event=None):
        self.calls.append(message)
        if on_event:
            for event in self.events:
                on_event(event)
        if self.raise_exc:
            raise self.raise_exc
        return self.reply


def _parse_sse(body: bytes) -> list[dict]:
    text = body.decode("utf-8")
    events = []
    for frame in text.split("\n\n"):
        frame = frame.strip()
        if frame.startswith("data: "):
            events.append(json.loads(frame[len("data: "):]))
    return events


def _isolate_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ASSISTANT_CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(config_module, "_config_file_cache", None)


class TestIndexAndStatus:
    def test_index_renders(self):
        client = webapp.app.test_client()
        response = client.get("/")
        assert response.status_code == 200
        assert b"Local AI Assistant" in response.data

    def test_status_returns_current_backend(self):
        client = webapp.app.test_client()
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.get_json()
        assert data["backend"] == CONFIG.llm_backend
        assert "model" in data
        assert "web_search_backend" in data


class TestChatEndpoint:
    def test_empty_message_rejected(self, monkeypatch):
        monkeypatch.setattr(webapp, "_agent", FakeAgent())
        client = webapp.app.test_client()
        response = client.post("/api/chat", json={"message": "   "})
        assert response.status_code == 400

    def test_missing_body_rejected(self, monkeypatch):
        monkeypatch.setattr(webapp, "_agent", FakeAgent())
        client = webapp.app.test_client()
        response = client.post("/api/chat", json={})
        assert response.status_code == 400

    def test_successful_reply(self, monkeypatch):
        fake = FakeAgent(reply="hello back")
        monkeypatch.setattr(webapp, "_agent", fake)
        client = webapp.app.test_client()
        response = client.post("/api/chat", json={"message": "hi"})
        assert response.status_code == 200
        assert response.get_json()["reply"] == "hello back"
        assert fake.calls == ["hi"]

    def test_llm_connection_error_returns_502(self, monkeypatch):
        monkeypatch.setattr(webapp, "_agent", FakeAgent(raise_exc=OllamaConnectionError("offline")))
        client = webapp.app.test_client()
        response = client.post("/api/chat", json={"message": "hi"})
        assert response.status_code == 502
        assert "error" in response.get_json()

    def test_unexpected_error_returns_500(self, monkeypatch):
        monkeypatch.setattr(webapp, "_agent", FakeAgent(raise_exc=RuntimeError("boom")))
        client = webapp.app.test_client()
        response = client.post("/api/chat", json={"message": "hi"})
        assert response.status_code == 500


class TestChatStreamEndpoint:
    def test_empty_message_rejected(self, monkeypatch):
        monkeypatch.setattr(webapp, "_agent", FakeAgent())
        client = webapp.app.test_client()
        response = client.post("/api/chat/stream", json={"message": ""})
        assert response.status_code == 400

    def test_streams_token_and_final_events(self, monkeypatch):
        fake = FakeAgent(
            reply="Hello world.",
            events=[TokenEvent("Hello "), TokenEvent("world.")],
        )
        monkeypatch.setattr(webapp, "_agent", fake)
        client = webapp.app.test_client()

        response = client.post("/api/chat/stream", json={"message": "hi"})
        assert response.status_code == 200
        assert response.mimetype == "text/event-stream"

        events = _parse_sse(response.get_data())
        assert events[0] == {"type": "token", "text": "Hello "}
        assert events[1] == {"type": "token", "text": "world."}
        assert events[2] == {"type": "final", "text": "Hello world."}
        assert fake.calls == ["hi"]

    def test_streams_tool_call_and_result_events(self, monkeypatch):
        fake = FakeAgent(
            reply="done",
            events=[
                ToolCallEvent("web_search", {"query": "cats"}),
                ToolResultEvent("web_search", "some results"),
            ],
        )
        monkeypatch.setattr(webapp, "_agent", fake)
        client = webapp.app.test_client()

        response = client.post("/api/chat/stream", json={"message": "search cats"})
        events = _parse_sse(response.get_data())

        assert events[0] == {"type": "tool_call", "tool": "web_search", "args": {"query": "cats"}}
        assert events[1] == {"type": "tool_result", "tool": "web_search"}
        assert events[2] == {"type": "final", "text": "done"}

    def test_llm_connection_error_streams_error_event(self, monkeypatch):
        monkeypatch.setattr(webapp, "_agent", FakeAgent(raise_exc=OllamaConnectionError("offline")))
        client = webapp.app.test_client()
        response = client.post("/api/chat/stream", json={"message": "hi"})
        events = _parse_sse(response.get_data())
        assert events[-1]["type"] == "error"
        assert "offline" in events[-1]["text"]

    def test_unexpected_error_streams_error_event(self, monkeypatch):
        monkeypatch.setattr(webapp, "_agent", FakeAgent(raise_exc=RuntimeError("boom")))
        client = webapp.app.test_client()
        response = client.post("/api/chat/stream", json={"message": "hi"})
        events = _parse_sse(response.get_data())
        assert events[-1]["type"] == "error"


class TestConfigEndpoints:
    def test_get_config_lists_editable_fields(self):
        client = webapp.app.test_client()
        response = client.get("/api/config")
        data = response.get_json()
        assert "model" in data["fields"]
        assert data["values"]["model"] == CONFIG.model
        assert "overridden" in data

    def test_post_rejects_unknown_field(self, monkeypatch, tmp_path):
        _isolate_config_file(monkeypatch, tmp_path)
        client = webapp.app.test_client()
        response = client.post("/api/config", json={"nonexistent_setting": 1})
        assert response.status_code == 400
        assert "nonexistent_setting" in response.get_json()["details"]

    def test_post_rejects_wrong_type(self, monkeypatch, tmp_path):
        _isolate_config_file(monkeypatch, tmp_path)
        client = webapp.app.test_client()
        response = client.post("/api/config", json={"temperature": "not-a-number"})
        assert response.status_code == 400

    def test_post_updates_value_and_rebuilds_agent(self, monkeypatch, tmp_path):
        _isolate_config_file(monkeypatch, tmp_path)
        original_temperature = CONFIG.temperature
        original_agent = webapp._agent
        try:
            client = webapp.app.test_client()
            response = client.post("/api/config", json={"temperature": 0.77})
            assert response.status_code == 200
            assert response.get_json()["values"]["temperature"] == 0.77
            assert CONFIG.temperature == 0.77
            # A fresh Agent should have been built so the new value
            # actually takes effect (OllamaClient captures it once).
            assert webapp._agent is not original_agent
        finally:
            config_module.save_overrides({"temperature": original_temperature})
            webapp._agent = original_agent

    def test_post_empty_body_rejected(self, monkeypatch, tmp_path):
        _isolate_config_file(monkeypatch, tmp_path)
        client = webapp.app.test_client()
        response = client.post("/api/config", json={})
        assert response.status_code == 400
