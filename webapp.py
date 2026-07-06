"""
webapp.py
=========

Optional web front-end for the assistant. This is just another
front-end calling `Agent.run_turn(message)`, the same function
main.py's CLI loop uses -- see claude.md's "Future extension ideas"
(now implemented). No agent logic lives here; this module only wires
HTTP requests to the existing `Agent`.

Run with:

    python webapp.py

Then open http://127.0.0.1:5000 in a browser. Intended for local,
single-user use only -- there is no authentication, and one shared
`Agent` instance (and its `ConversationManager` history) is reused
across every request.
"""

from __future__ import annotations

import json
import logging
import queue
import threading

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

import config as config_module
from agent import Agent, LLMConnectionError, TokenEvent, ToolCallEvent, ToolResultEvent
from config import CONFIG
from main import configure_logging

configure_logging()

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
logger = logging.getLogger("assistant.webapp")

# One shared Agent for the lifetime of the process -- same rationale
# as main.py's REPL: a single local user, one running conversation.
_agent = Agent()


def _current_model() -> str:
    return CONFIG.anthropic_model if CONFIG.llm_backend == "claude" else CONFIG.model


@app.route("/")
def index():
    return render_template(
        "index.html",
        backend=CONFIG.llm_backend,
        model=_current_model(),
        web_search_backend=CONFIG.web_search_backend,
    )


@app.route("/api/status")
def status():
    """Plain-JSON status info for non-HTML front-ends (e.g. the Rust
    desktop GUI in gui/), which can't scrape backend/model info out of
    the rendered index.html template the way the browser UI does.
    """
    return jsonify(
        {
            "backend": CONFIG.llm_backend,
            "model": _current_model(),
            "web_search_backend": CONFIG.web_search_backend,
        }
    )


@app.route("/api/config", methods=["GET"])
def get_config():
    """Current values of every user-editable setting, plus which ones
    are explicitly overridden in config.json (vs. falling back to an
    env var or default) -- the settings panel uses this to render a
    form and show what's been customized.
    """
    return jsonify(
        {
            "fields": config_module.EDITABLE_FIELDS,
            "values": {name: getattr(CONFIG, name) for name in config_module.EDITABLE_FIELDS},
            "overridden": sorted(config_module.current_overrides().keys()),
        }
    )


@app.route("/api/config", methods=["POST"])
def update_config():
    """Save one or more settings. Values are validated against
    EDITABLE_FIELDS' declared type and converted before being written
    to config.json; anything not in EDITABLE_FIELDS (including any
    would-be secret field -- see config.py's module docstring, there
    are none) is rejected outright rather than silently ignored.
    """
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict) or not data:
        return jsonify({"error": "Request body must be a non-empty JSON object of settings."}), 400

    overrides: dict = {}
    errors: dict = {}
    for key, value in data.items():
        kind = config_module.EDITABLE_FIELDS.get(key)
        if kind is None:
            errors[key] = "unknown setting"
            continue
        try:
            if kind == "int":
                overrides[key] = int(value)
            elif kind == "float":
                overrides[key] = float(value)
            else:
                overrides[key] = str(value)
        except (TypeError, ValueError):
            errors[key] = f"expected a {kind}"

    if errors:
        return jsonify({"error": "Invalid settings.", "details": errors}), 400

    config_module.save_overrides(overrides)

    # OllamaClient/ClaudeClient/ToolExecutor all capture CONFIG values
    # into their own __init__ once; only a fresh Agent picks up
    # backend/model/etc. changes made via reload_from_config_file().
    global _agent
    _agent = Agent()

    return jsonify(
        {
            "values": {name: getattr(CONFIG, name) for name in config_module.EDITABLE_FIELDS},
            "overridden": sorted(config_module.current_overrides().keys()),
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message must not be empty."}), 400

    try:
        answer = _agent.run_turn(message)
    except LLMConnectionError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception:  # noqa: BLE001 - never let the request crash the server
        logger.exception("Unexpected error handling chat request")
        return jsonify({"error": "Something went wrong on my end -- check assistant.log for details."}), 500

    return jsonify({"reply": answer})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Server-Sent Events version of /api/chat: emits one JSON object
    per SSE frame as the turn progresses --
    {"type": "token", "text": ...} for each streamed chunk,
    {"type": "tool_call"/"tool_result", ...} around tool execution, and
    a final {"type": "final", "text": ...} or {"type": "error", ...}.

    Agent.run_turn's on_event callback is synchronous, so the actual
    turn runs on a background thread; this generator just relays
    whatever that thread pushes onto a queue, as it arrives, which is
    what makes this an actual stream rather than one big blocking
    response.
    """
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message must not be empty."}), 400

    events: queue.Queue = queue.Queue()

    def on_event(event) -> None:
        if isinstance(event, TokenEvent):
            events.put({"type": "token", "text": event.text})
        elif isinstance(event, ToolCallEvent):
            events.put({"type": "tool_call", "tool": event.tool, "args": event.args})
        elif isinstance(event, ToolResultEvent):
            events.put({"type": "tool_result", "tool": event.tool})

    def worker() -> None:
        try:
            answer = _agent.run_turn(message, on_event=on_event)
            events.put({"type": "final", "text": answer})
        except LLMConnectionError as exc:
            events.put({"type": "error", "text": str(exc)})
        except Exception:  # noqa: BLE001 - never let the worker thread crash silently
            logger.exception("Unexpected error handling streamed chat request")
            events.put({"type": "error", "text": "Something went wrong on my end -- check assistant.log for details."})
        finally:
            events.put(None)  # sentinel: tells generate() the turn is over

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            item = events.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
