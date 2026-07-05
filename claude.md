# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this
before making changes — it captures architectural decisions and
conventions that aren't obvious from any single file.

## What this project is

A local-first AI assistant that uses Ollama as its default LLM
backend (with an opt-in Claude/Anthropic API backend) and an
Obsidian-compatible markdown vault as persistent long-term memory. It
runs via a CLI (`main.py`). Outbound network access is limited to: the
configured LLM backend (local Ollama by default, or the Anthropic API
if `ASSISTANT_LLM_BACKEND=claude`), and web search/fetch (`websearch.py`,
enabled by default via DuckDuckGo, no API key required).

## Module map (don't blur these boundaries)

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth for all settings (`CONFIG` singleton). Everything configurable lives here, override via env vars. |
| `utils.py` | `resolve_safe_path` — the only function allowed to decide whether a vault-relative path is safe. Also small helpers (truncate, tokenize). |
| `parser.py` | Turns raw LLM text into either a validated `ToolCall` or final-answer text. Owns `VALID_TOOLS` and the required-args schema. |
| `tools.py` | Executes a `ToolCall` against `VaultMemory`. One handler function per tool, registered in the `_HANDLERS` dict. |
| `memory.py` | `VaultMemory` — all vault file I/O (read/write/append/remove/list/create_folder) plus `Retriever` protocol + `KeywordRetriever` (current default search). |
| `embeddings.py` | Optional semantic search: `EmbeddingRetriever` implements the same `Retriever` protocol so it's a drop-in replacement for `KeywordRetriever`. Not wired in by default. |
| `websearch.py` | Web search/fetch: `SearchProvider` Protocol with `DuckDuckGoSearchProvider` (default, no key) and `BraveSearchProvider` (needs `BRAVE_API_KEY`), plus `fetch_url` for `web_fetch`. The only module besides `agent.py` with outbound internet access. |
| `conversation.py` | `ConversationManager` — rolling history + summarization of old turns into `vault/conversations/`. |
| `agent.py` | `OllamaClient` + `ClaudeClient` (the *only* module that talks to an LLM backend, selected via `CONFIG.llm_backend`) + `Agent` (the tool-calling loop, retrieval wiring, automatic memory creation). |
| `prompts.py` | Every prompt template. Nothing else should hardcode prompt text. |
| `main.py` | CLI entry point / REPL (rich-rendered: banner panel, markdown replies, spinner). |
| `webapp.py` | Optional Flask web front-end — same rules as `main.py`: calls `Agent.run_turn`, no agent logic of its own. Templates/assets in `web/`. Also exposes `/api/status` (JSON) for non-HTML clients. |
| `gui/` | Rust/egui desktop GUI — a separate Cargo crate, HTTP client of `webapp.py` only (`/api/status`, `/api/chat`). No agent/tool logic; don't add any there. See `gui/README.md`. |
| `tests/` | pytest suite. See "Running / testing locally" below for the CONFIG-patching fixtures. |

**Rule of thumb:** if you're adding a capability, ask which of these
layers it belongs to before writing code. New tools go in
`tools.py` + `parser.py`'s schema; new prompt text goes in
`prompts.py`; new LLM-backend behavior goes in `agent.py`'s
`OllamaClient`.

## Non-negotiable invariants

- **Path safety**: every filesystem operation on the vault MUST go
  through `utils.resolve_safe_path`. No new code should call `open()`,
  `Path()`, or similar on a model-supplied path directly.
- **No shell or code execution tools.** This was an explicit
  requirement from the start and should stay that way unless the user
  asks for it directly and understands the risk.
- **Tool calls are strict JSON, one object, no XML.** `parser.py`'s
  `_extract_json_object` already handles the common case of models
  wrapping JSON in markdown fences — don't relax the schema validation
  in `validate_tool_call` to work around a model's bad output; fix the
  prompt instead.
- **`prompts.py` templates that use `.format()` must escape literal
  braces as `{{` / `{}}`.** (`MEMORY_CREATION_PROMPT` had a bug here —
  its embedded JSON example `{"tool": "none"}` broke `.format()` until
  it was escaped to `{{"tool": "none"}}`. If you add new prompt
  templates containing example JSON and call `.format()` on them,
  escape the braces the same way, or use `.replace()` instead of
  `.format()` for that template.)
- **One dedicated LLM module.** All LLM communication goes through
  `agent.py` — `OllamaClient` for the local backend, `ClaudeClient` for
  the opt-in Anthropic backend, selected by `build_llm_client()` via
  `CONFIG.llm_backend`. Don't build a third HTTP client elsewhere, and
  don't let other modules call Ollama or Anthropic directly. Both
  clients implement the same `chat()`/`complete_text()` shape and raise
  a subclass of `LLMConnectionError` so `Agent` and `main.py` can
  handle either backend identically.
- **No global mutable state.** `CONFIG` is a frozen dataclass singleton;
  everything else is passed explicitly (`VaultMemory`, `OllamaClient`,
  `ConversationManager` are all constructed and threaded through
  `Agent`).

## Style conventions already in use

- Python 3.11+, full type hints, docstrings on every public
  class/function, PEP8.
- Dataclasses for simple value objects (`ToolCall`, `SearchHit`,
  `Turn`, `ParsedResponse`).
- `Protocol` classes (not ABCs) for swappable interfaces —
  see `Retriever` in `memory.py` and `EmbeddingProvider`/`VectorStore`
  in `embeddings.py`. Follow this pattern for any new pluggable
  backend (e.g. a `SearchProvider` protocol for web search).
- Errors are caught at the boundary and turned into strings the model
  can see (`[tool error] ...` in `tools.py`), never allowed to crash
  `main.py`'s loop. Keep that contract for new tools.
- Logging via `logging.getLogger("assistant.<module>")`, configured
  once in `main.py`. Use `logger.info` for tool calls/writes/retrieval,
  `logger.warning` for recoverable failures, `logger.exception` only
  at the last-resort catch-all.

## Running / testing locally

```bash
pip install -r requirements.txt
ollama serve                 # separate terminal
ollama pull qwen3.5          # or whatever CONFIG.model points at
python main.py
```

A pytest suite lives in `tests/` (`pytest.ini` adds the project root to
`sys.path` via `pythonpath = .`). Run it with:

```bash
pip install -r requirements-dev.txt
pytest
```

Because `CONFIG` is a frozen dataclass singleton, tests can't reassign
its fields directly — see `tests/conftest.py`'s `vault_dir` fixture
(patches the `Config.vault_dir` *property* at the class level) and
`config_override` fixture (swaps the module-level `CONFIG` name a
module imported for a forwarding proxy) for the two patterns in use.
Follow one of those rather than trying `monkeypatch.setattr(CONFIG, ...)`
directly, which raises `FrozenInstanceError`.

When making changes, at minimum run the test suite plus:

```bash
python -m py_compile *.py
```

and a quick manual smoke test of any touched tool via `ToolExecutor`
directly (see the pattern used during initial development — construct
a `VaultMemory()`, wrap in `ToolExecutor`, call `.execute(ToolCall(...))`
for each new tool path, including an intentionally malicious path like
`"../escape.md"` to confirm `resolve_safe_path` still rejects it).

## Web search (implemented)

`websearch.py` provides `web_search` and `web_fetch` tools, following
the same shape originally planned here:

- `SearchProvider` Protocol (`search(query, top_k) -> list[WebSearchHit]`),
  with `DuckDuckGoSearchProvider` (default, no API key) and
  `BraveSearchProvider` (needs `BRAVE_API_KEY`) implementations,
  selected via `CONFIG.web_search_backend` ("none" disables the tool
  entirely). API keys come from env vars only — never stored in the
  vault, never in `config.py` literals.
- `parser.py`'s `VALID_TOOLS`/`_REQUIRED_ARGS` include `web_search`
  (`{"query": str}`) and `web_fetch` (`{"url": str}`); handlers live in
  `tools.py`'s `_HANDLERS`.
- **Prompt-injection mitigation:** both handlers wrap their output in
  `tools.py`'s `_wrap_untrusted()` delimiter
  (`=== BEGIN/END UNTRUSTED WEB CONTENT ===`), and `prompts.py`'s
  `TOOL_INSTRUCTIONS` tells the model that content between those
  markers is data to read, never instructions to follow.
- Both handlers truncate via `utils.truncate` (`CONFIG.max_web_fetch_chars`
  for `web_fetch`) so a large page or result set can't blow the
  context window.
- `websearch.py` and `agent.py`'s LLM clients are the *only* modules
  with outbound internet access — don't let other modules make their
  own HTTP requests.

## Claude/Anthropic backend (implemented)

`agent.py` supports a second LLM backend alongside Ollama:
`CONFIG.llm_backend` ("ollama" default, or "claude") picks between
`OllamaClient` and `ClaudeClient` via `build_llm_client()`. Both
implement the same `chat(messages)`/`complete_text(prompt)` interface
and raise a subclass of `LLMConnectionError`, so `Agent`, `main.py`,
and `ConversationManager`'s summarizer callback don't need to know
which backend is active.

- `ClaudeClient` uses the official `anthropic` SDK (lazy-imported, same
  pattern as `OllamaClient`'s `requests` import) and reads
  `ANTHROPIC_API_KEY` the standard way — never through `CONFIG`.
  `CONFIG.anthropic_model` / `CONFIG.anthropic_max_tokens` are
  configurable via env vars.
- Ollama's inline `{"role": "system", ...}` message shape doesn't match
  Claude's API (system prompt is a separate top-level field) —
  `agent.py`'s `_split_system_message()` converts between the two.
- Switching to Claude sends conversation content to Anthropic's
  servers, so it's opt-in only, never the default.