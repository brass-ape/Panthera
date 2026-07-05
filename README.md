# Local AI Assistant

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

*(Replace `OWNER/REPO` above once this is pushed to GitHub.)*

A local, privacy-respecting AI assistant that uses [Ollama](https://ollama.com)
as its language model and an Obsidian-compatible markdown vault as its
persistent long-term memory. Everything runs on your own machine —
no data leaves it (unless you opt into the Claude backend, see below).

Three front-ends, one backend: a [rich](https://github.com/Textualize/rich)-rendered
CLI (`main.py`), a browser UI (`webapp.py`), and a native desktop GUI
(`gui/`, Rust/egui) all drive the same `Agent.run_turn` — no logic is
duplicated between them.

## Features

- **Persistent memory** — a markdown vault the assistant reads from and
  writes to, organized into `people/`, `projects/`, `journal/`, `facts/`,
  `conversations/`, and `resources/`.
- **Automatic retrieval** — before answering, the assistant searches the
  vault for relevant notes and inserts only those into context (never
  the whole vault).
- **Always-on context** — the assistant always knows the current local
  date/time and basic system specs (no tool call needed), and
  `vault/resources/` (reference material you place there) is always
  included regardless of relevance, unlike the rest of the vault.
- **Streamed responses** — replies stream token-by-token in the CLI
  (also in the web UI/GUI via the underlying API), so you see the
  answer forming instead of waiting on one long request.
- **Tool calling** — the model can call `read`, `write`, `append`,
  `remove`, `search`, `list_files`, `read_multiple`, `create_folder`,
  `web_search`, and `web_fetch` as strict JSON tool calls, chained
  across multiple steps.
- **Web search** — `web_search` and `web_fetch` tools, backed by
  DuckDuckGo by default (no API key) or Brave Search
  (`BRAVE_API_KEY`). Results are wrapped as untrusted content so the
  model can't be steered by anything a fetched page says.
- **Pluggable LLM backend** — Ollama by default (fully local); set
  `ASSISTANT_LLM_BACKEND=claude` to use the Anthropic API instead.
- **Editable settings, no restart needed** — every tunable in
  `config.py` can also be changed from the web UI's or desktop GUI's
  settings panel (⚙), backed by `config.json` (see Configuration).
- **Automatic memory creation** — after each turn, the assistant asks
  itself whether anything is worth remembering, and if so, writes a
  concise note.
- **Conversation summarization** — old conversation turns are
  periodically summarized into `vault/conversations/` and dropped from
  RAM, so long sessions don't blow out the context window.
- **Sandboxed file access** — every path is validated to stay inside
  the vault; no `..`, absolute paths, or symlink escapes.
- **Pluggable search** — keyword search today, with an
  `embeddings.py` module ready for semantic search (Ollama embeddings +
  FAISS/Chroma) without changing the rest of the architecture.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- A pulled chat model (e.g. `qwen3.5`, `llama3.1`, `mistral`, ...)

## Installation

```bash
git clone <this-repo>
cd assistant
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running Ollama

Make sure the Ollama service is running:

```bash
ollama serve
```

(On macOS/Windows, the Ollama desktop app runs this for you.)

## Downloading a model

```bash
ollama pull qwen3.5
```

Update `ASSISTANT_MODEL` (see Configuration below) or edit the default
in `config.py` if you use a different model name.

Optional, only if you enable semantic search later:

```bash
ollama pull nomic-embed-text
```

## Running the assistant

```bash
python main.py
```

You'll get a [rich](https://github.com/Textualize/rich)-rendered REPL:
a bordered banner panel with the active backend/model/vault, a
spinner while the model is thinking, and the assistant's replies
rendered as markdown in their own panel.

```
You: My name is Alex and I'm learning Rust.
Assistant: Nice to meet you, Alex! I'll remember that you're learning Rust.
```

Behind the scenes, that turn likely triggered an automatic `write` or
`append` tool call creating/updating `people/alex.md`.

## Running the web UI

An optional browser-based front-end (`webapp.py`) calls the same
`Agent.run_turn` the CLI uses, styled as a glassmorphic "liquid glass"
chat window:

```bash
pip install flask
python webapp.py
```

Then open <http://127.0.0.1:5000>. Like the CLI, it's meant for local,
single-user use — there's no authentication and one shared `Agent` (and
its conversation history) is reused across requests.

## Running the desktop GUI

A cross-platform native desktop client lives in `gui/` (Rust,
[egui](https://github.com/emilk/egui)/eframe). It's purely a client
for `webapp.py`'s HTTP API — start the web server first, then:

```bash
cd gui
cargo run --release
```

See `gui/README.md` for connecting to a non-default host/port and
cross-platform build notes.

## Continuous integration

`.github/workflows/ci.yml` runs on every push/PR: the pytest suite
against Python 3.11 and 3.12, and a release build of the Rust GUI. Run
the same checks locally with:

```bash
pip install -r requirements-dev.txt && pytest -q
cd gui && cargo build --release
```

## Project structure

```
assistant/
├── main.py           CLI entry point (rich-based REPL, streamed replies)
├── webapp.py          Optional Flask web front-end (+ /api/status, /api/config)
├── config.py           All configuration (config.json > env vars > defaults)
├── agent.py             Ollama/Claude clients + the tool-calling agent loop
├── memory.py             Vault file operations + keyword search/retrieval
├── tools.py               Executes validated tool calls against memory.py
├── parser.py                Strict JSON tool-call parsing & validation
├── prompts.py                 All prompt templates
├── sysinfo.py                   Local date/time + system specs for context
├── conversation.py                Conversation history + summarization
├── embeddings.py                    Semantic search interface (optional)
├── websearch.py                       Web search / web fetch tools
├── utils.py                             Path sandboxing & small helpers
├── web/                                    Templates/static assets for webapp.py
│   ├── templates/index.html
│   └── static/{css,js}/
├── gui/                                       Rust/egui desktop GUI (client of webapp.py)
├── tests/                                    pytest suite
├── .github/workflows/ci.yml                     GitHub Actions: pytest + gui build
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── vault/                                          (gitignored -- your data)
    ├── people/
    ├── projects/
    ├── journal/
    ├── facts/
    ├── resources/       always-included reference material (see Features)
    └── conversations/
```

## How memory works

1. **Retrieval** — every user message is keyword-searched against the
   vault (`memory.py: VaultMemory.retrieve_context_for`). The top
   matching files (capped by `ASSISTANT_MAX_CONTEXT_FILES`) are
   inserted into the system prompt.
2. **Tool calling** — if the model needs more, it can call `search`,
   `read`, or `read_multiple` itself, chaining calls up to
   `ASSISTANT_MAX_TOOL_ITERATIONS` times before giving a final answer.
3. **Writing** — the model can call `write` or `append` directly, or
   the automatic memory-creation step (run after every turn) can do
   it for you, guided by the memory policy in `prompts.py` (remember
   names, preferences, projects, goals, specs, skills; never remember
   secrets or one-off trivia).
4. **Summarization** — once conversation history passes
   `ASSISTANT_SUMMARIZE_AFTER` turns, the oldest turns are summarized
   by the model and written to `vault/conversations/`, then dropped
   from RAM.

## Configuration

Settings come from three places, highest precedence first:

1. **`config.json`** (project root, gitignored, created on first save)
   — edit it by hand, or use the web UI's/desktop GUI's settings panel
   (the ⚙ button), which write here and take effect immediately, no
   restart needed.
2. **Environment variables**, e.g.:

   ```bash
   export ASSISTANT_MODEL="llama3.1:8b"
   export ASSISTANT_MAX_TOOL_ITERATIONS=12
   export ASSISTANT_LOG_LEVEL=DEBUG
   python main.py
   ```
3. **Hardcoded defaults** in `config.py`.

See `config.py`'s `EDITABLE_FIELDS` for the full list of settings (vault
location, model, temperature, embedding model/backend, history sizes,
logging, etc) and their names in both env-var and `config.json` form.
There is no API-key field anywhere in `Config` — `ANTHROPIC_API_KEY`
is resolved by the `anthropic` SDK directly, never stored here.

## Using the Claude/Anthropic backend instead of Ollama

By default the assistant talks to a local Ollama model. To use the
Anthropic API instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-..."   # or run `ant auth login`
export ASSISTANT_LLM_BACKEND=claude
export ASSISTANT_ANTHROPIC_MODEL=claude-opus-4-8   # optional, this is the default
python main.py
```

Note this sends your conversation (including any retrieved vault
content) to Anthropic's servers, unlike the Ollama backend which never
leaves your machine.

## Web search

Enabled by default via DuckDuckGo's HTML endpoint (no API key
needed). To use Brave Search instead:

```bash
export BRAVE_API_KEY="..."
export ASSISTANT_WEB_SEARCH_BACKEND=brave
```

To disable web search entirely: `export ASSISTANT_WEB_SEARCH_BACKEND=none`.

## Example conversation

```
You: I've got an RTX 4070 and 64GB of DDR4 in my main rig.
Assistant: Got it — I'll note your hardware specs for future reference.

You: What GPU do I have again?
Assistant: You have an RTX 4070.
```

(The second answer came from `facts/` or `people/` memory retrieved
via keyword search — no need to repeat yourself.)

## Enabling semantic search (optional)

Keyword search is the default and requires no extra setup. To switch
to embeddings:

```bash
ollama pull nomic-embed-text
export ASSISTANT_EMBED_BACKEND=ollama
export ASSISTANT_VECTOR_STORE=faiss   # or "chroma", or leave unset for in-memory
pip install faiss-cpu                  # if using the faiss backend
```

Then construct `VaultMemory` with an `EmbeddingRetriever` instead of
the default `KeywordRetriever`:

```python
from pathlib import Path
from embeddings import EmbeddingRetriever
from memory import VaultMemory
from config import CONFIG

memory = VaultMemory(retriever=EmbeddingRetriever(CONFIG.vault_dir))
```

Nothing else in the codebase needs to change — `agent.py` and
`tools.py` only depend on the `VaultMemory` interface.

## Future extension ideas

The architecture was designed so these can be added without a
rewrite:

- **Voice input / output** — add a `speech.py` module that turns
  audio into text before calling `Agent.run_turn`, and text into audio
  from its return value.
- **Vision models** — extend `OllamaClient.chat` to accept image
  attachments (Ollama's multimodal models support this natively).
- **Desktop automation** — add a `desktop.py` module exposing tools
  like `open_app` or `click`, following the same validate-then-execute
  pattern as `tools.py`.
- **Home Assistant / Discord bot** — each is just a new front-end that
  calls `Agent.run_turn(message)`, the same function `main.py` and
  `webapp.py` use.
- **Native desktop client** — implemented in `gui/` (Rust/egui),
  talking to `webapp.py`'s HTTP API rather than reimplementing the
  agent loop.
- **Plugin API** — the `_HANDLERS` dict in `tools.py` and `VALID_TOOLS`
  set in `parser.py` are the two places a new tool needs to register;
  a future `plugins/` loader could populate both dynamically.

## Safety notes

- All file paths are validated by `utils.resolve_safe_path` before any
  filesystem access: no absolute paths, `..` traversal, or symlinks
  leaving the vault.
- No shell or code execution tools are implemented or planned as
  "just another tool" — anything like that should be reviewed
  carefully and sandboxed independently.
- Secrets (passwords, API keys) are explicitly excluded from the
  memory policy in `prompts.py`.
