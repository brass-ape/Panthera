"""
config.py
=========

Central configuration for the local AI assistant.

Every tunable value used elsewhere in the codebase lives here so the
rest of the application never hardcodes a model name, path, or limit.
Values can come from three places, in this precedence order:

1. config.json (project root by default; see EDITABLE_FIELDS) --
   highest precedence, so a settings panel (webapp.py's /api/config,
   the desktop GUI) can change behavior immediately without needing
   the process restarted with new env vars.
2. Environment variables (e.g. ASSISTANT_MODEL) -- for
   scripting/deployment, and as the only way to configure anything
   before config.json exists.
3. The hardcoded default.

config.json intentionally never holds secrets: there is no API-key
field anywhere in Config, and the `anthropic` SDK resolves
ANTHROPIC_API_KEY on its own (see agent.py's ClaudeClient) -- nothing
here ever reads, stores, or writes it.

This module intentionally has no side effects at import time beyond
building the dataclass instance and ensuring the vault directories
exist, so it is safe to import from anywhere.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("assistant.config")

_PROJECT_ROOT = Path(__file__).resolve().parent
_config_file_cache: dict | None = None


def _config_file_path() -> Path:
    return Path(os.environ.get("ASSISTANT_CONFIG_FILE", str(_PROJECT_ROOT / "config.json")))


def _load_config_file() -> dict:
    """Read and cache config.json. Missing file or invalid JSON is
    treated as "no overrides" rather than an error -- this file is
    optional and entirely user-managed.
    """
    global _config_file_cache
    if _config_file_cache is not None:
        return _config_file_cache

    path = _config_file_path()
    if not path.exists():
        _config_file_cache = {}
        return _config_file_cache

    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        _config_file_cache = data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s, ignoring it: %s", path, exc)
        _config_file_cache = {}
    return _config_file_cache


def _setting_str(key: str, env_name: str, default: str) -> str:
    data = _load_config_file()
    if key in data and isinstance(data[key], str):
        return data[key]
    return os.environ.get(env_name, default)


def _setting_int(key: str, env_name: str, default: int) -> int:
    data = _load_config_file()
    if key in data:
        try:
            return int(data[key])
        except (TypeError, ValueError):
            pass
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _setting_float(key: str, env_name: str, default: float) -> float:
    data = _load_config_file()
    if key in data:
        try:
            return float(data[key])
        except (TypeError, ValueError):
            pass
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    """Immutable-in-spirit configuration object for the assistant.

    ("Immutable-in-spirit" because reload_from_config_file() does
    mutate the shared CONFIG singleton in place via object.__setattr__
    -- see that function's docstring. Everywhere else, treat CONFIG as
    read-only.)

    All paths are resolved to absolute paths at construction time so
    that other modules never need to worry about relative-path
    ambiguity (this matters a lot for the vault sandboxing logic in
    utils.py).
    """

    # --- Paths -----------------------------------------------------
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    vault_dir_name: str = field(default_factory=lambda: _setting_str("vault_dir_name", "ASSISTANT_VAULT_DIR", "vault"))

    # --- Ollama / LLM ------------------------------------------------
    ollama_host: str = field(default_factory=lambda: _setting_str("ollama_host", "OLLAMA_HOST", "http://localhost:11434"))
    model: str = field(default_factory=lambda: _setting_str("model", "ASSISTANT_MODEL", "qwen3.5:latest"))
    temperature: float = field(default_factory=lambda: _setting_float("temperature", "ASSISTANT_TEMPERATURE", 0.4))
    request_timeout_seconds: int = field(
        default_factory=lambda: _setting_int("request_timeout_seconds", "ASSISTANT_LLM_TIMEOUT", 120)
    )

    # --- LLM backend selection ----------------------------------------
    # "ollama" (default, fully local) | "claude" (calls the Anthropic API --
    # sends conversation content off-machine and requires ANTHROPIC_API_KEY).
    llm_backend: str = field(default_factory=lambda: _setting_str("llm_backend", "ASSISTANT_LLM_BACKEND", "ollama"))
    anthropic_model: str = field(
        default_factory=lambda: _setting_str("anthropic_model", "ASSISTANT_ANTHROPIC_MODEL", "claude-opus-4-8")
    )
    anthropic_max_tokens: int = field(
        default_factory=lambda: _setting_int("anthropic_max_tokens", "ASSISTANT_ANTHROPIC_MAX_TOKENS", 4096)
    )
    # The Anthropic API key itself is never read here -- the `anthropic`
    # SDK resolves it from the standard ANTHROPIC_API_KEY env var (or an
    # `ant auth login` profile) on its own, so it never needs to pass
    # through CONFIG, config.json, or get logged.

    # --- Web search -----------------------------------------------------
    # "none" (disabled) | "duckduckgo" (default, no API key) | "brave"
    # (requires BRAVE_API_KEY).
    web_search_backend: str = field(
        default_factory=lambda: _setting_str("web_search_backend", "ASSISTANT_WEB_SEARCH_BACKEND", "duckduckgo")
    )
    max_web_search_results: int = field(
        default_factory=lambda: _setting_int("max_web_search_results", "ASSISTANT_MAX_WEB_RESULTS", 5)
    )
    max_web_fetch_chars: int = field(
        default_factory=lambda: _setting_int("max_web_fetch_chars", "ASSISTANT_MAX_WEB_FETCH_CHARS", 4000)
    )
    web_request_timeout_seconds: int = field(
        default_factory=lambda: _setting_int("web_request_timeout_seconds", "ASSISTANT_WEB_TIMEOUT", 15)
    )

    # --- Embeddings (for future semantic search) --------------------
    embedding_model: str = field(
        default_factory=lambda: _setting_str("embedding_model", "ASSISTANT_EMBED_MODEL", "nomic-embed-text")
    )
    embedding_backend: str = field(default_factory=lambda: _setting_str("embedding_backend", "ASSISTANT_EMBED_BACKEND", "none"))
    # "none" | "ollama" -- vector store backend used if/when enabled
    vector_store_backend: str = field(
        default_factory=lambda: _setting_str("vector_store_backend", "ASSISTANT_VECTOR_STORE", "none")
    )
    # "none" | "faiss" | "chroma"

    # --- Agent loop ---------------------------------------------------
    max_tool_iterations: int = field(
        default_factory=lambda: _setting_int("max_tool_iterations", "ASSISTANT_MAX_TOOL_ITERATIONS", 16)
    )

    # --- Memory / retrieval -------------------------------------------
    max_search_results: int = field(default_factory=lambda: _setting_int("max_search_results", "ASSISTANT_MAX_SEARCH_RESULTS", 5))
    max_context_files: int = field(default_factory=lambda: _setting_int("max_context_files", "ASSISTANT_MAX_CONTEXT_FILES", 5))
    max_file_chars_in_context: int = field(
        default_factory=lambda: _setting_int("max_file_chars_in_context", "ASSISTANT_MAX_FILE_CHARS", 4000)
    )
    # Cap on how many vault/resources/ files are always injected into
    # context (see memory.py's read_resources) -- distinct from
    # max_context_files, which caps *retrieved* (relevance-matched)
    # files, since resources/ is included unconditionally every turn.
    max_resource_files: int = field(default_factory=lambda: _setting_int("max_resource_files", "ASSISTANT_MAX_RESOURCE_FILES", 10))

    # --- Conversation history -----------------------------------------
    conversation_history_size: int = field(
        default_factory=lambda: _setting_int("conversation_history_size", "ASSISTANT_HISTORY_SIZE", 20)
    )
    # Number of turns kept in RAM before older turns are summarized
    # into vault/conversations/ and dropped from memory.
    summarize_after_turns: int = field(default_factory=lambda: _setting_int("summarize_after_turns", "ASSISTANT_SUMMARIZE_AFTER", 30))

    # --- Logging --------------------------------------------------------
    log_level: str = field(default_factory=lambda: _setting_str("log_level", "ASSISTANT_LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: _setting_str("log_file", "ASSISTANT_LOG_FILE", "assistant.log"))

    @property
    def vault_dir(self) -> Path:
        """Absolute path to the vault root directory."""
        return (self.project_root / self.vault_dir_name).resolve()

    @property
    def log_path(self) -> Path:
        return self.project_root / self.log_file

    def ensure_vault_structure(self) -> None:
        """Create the standard vault subfolders if they do not exist yet.

        This is idempotent and safe to call on every startup.
        """
        subfolders = ("people", "projects", "journal", "facts", "conversations", "resources")
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        for name in subfolders:
            (self.vault_dir / name).mkdir(parents=True, exist_ok=True)


# Fields a settings UI (webapp.py's /api/config, the desktop GUI) may
# read and write, mapped to the type used to validate/convert incoming
# values. Deliberately excludes project_root (not user-configurable)
# and anything secret (there is nothing secret in Config -- see the
# module docstring).
EDITABLE_FIELDS: dict[str, str] = {
    "vault_dir_name": "str",
    "ollama_host": "str",
    "model": "str",
    "temperature": "float",
    "request_timeout_seconds": "int",
    "llm_backend": "str",
    "anthropic_model": "str",
    "anthropic_max_tokens": "int",
    "web_search_backend": "str",
    "max_web_search_results": "int",
    "max_web_fetch_chars": "int",
    "web_request_timeout_seconds": "int",
    "max_tool_iterations": "int",
    "max_search_results": "int",
    "max_context_files": "int",
    "max_file_chars_in_context": "int",
    "max_resource_files": "int",
    "conversation_history_size": "int",
    "summarize_after_turns": "int",
    "log_level": "str",
}


def current_overrides() -> dict:
    """The raw contents of config.json, for a settings UI to show
    which values are explicitly overridden (vs. falling back to an
    env var or default).
    """
    return dict(_load_config_file())


def save_overrides(overrides: dict) -> None:
    """Merge `overrides` into config.json on disk and refresh the live
    CONFIG singleton so the change takes effect immediately -- this is
    what a settings UI calls. Callers are responsible for restricting
    `overrides` to EDITABLE_FIELDS and converting values to the right
    type first (see webapp.py's /api/config handler); this function
    trusts its input.
    """
    path = _config_file_path()
    merged = dict(_load_config_file())
    merged.update(overrides)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reload_from_config_file()


def reload_from_config_file() -> None:
    """Re-read config.json and refresh every field on the shared
    CONFIG singleton in place (via object.__setattr__, bypassing the
    dataclass's normal frozen-ness deliberately, only here), so a
    settings-panel save takes effect immediately without restarting
    the process. Every module does `from config import CONFIG`, so
    mutating the existing object -- rather than rebinding the module
    attribute to a new instance -- is what makes already-imported
    references see the update too.

    Note this does NOT update objects that already copied a CONFIG
    value into their own `__init__` (e.g. OllamaClient captures
    CONFIG.model once at construction) -- callers that change
    LLM-backend-affecting settings need to reconstruct the Agent
    afterwards (see webapp.py's /api/config POST handler).
    """
    global _config_file_cache
    _config_file_cache = None  # force the next _load_config_file() to re-read from disk
    fresh = Config()
    for f in dataclasses.fields(Config):
        object.__setattr__(CONFIG, f.name, getattr(fresh, f.name))
    CONFIG.ensure_vault_structure()


# A single shared configuration instance. Modules should import this
# rather than constructing their own Config(), so the whole
# application shares one source of truth.
CONFIG = Config()
CONFIG.ensure_vault_structure()
