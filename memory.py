"""
memory.py
=========

Everything to do with reading, writing, and searching the Obsidian
vault that serves as the assistant's long-term memory.

Design note on future semantic search
--------------------------------------
`VaultMemory.search` currently does keyword search only. It is
written against a small `Retriever` interface so that swapping in
embeddings.py's semantic search later only requires changing which
Retriever is constructed in `VaultMemory.__init__` -- nothing in
agent.py, tools.py, or elsewhere needs to change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from config import CONFIG
from utils import UnsafePathError, resolve_safe_path, word_tokens

logger = logging.getLogger("assistant.memory")


class VaultError(Exception):
    """Raised for vault operation failures that should be reported
    back to the model as a tool error rather than crashing the app.
    """


@dataclass
class SearchHit:
    """A single search result: a file and a relevance score."""

    path: str  # vault-relative path, e.g. "people/me.md"
    score: float
    snippet: str


class Retriever(Protocol):
    """Interface any search backend (keyword, embeddings, ...) must
    implement so VaultMemory can use it interchangeably.
    """

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        ...

    def index_file(self, relative_path: str, content: str) -> None:
        ...

    def remove_file(self, relative_path: str) -> None:
        ...


class KeywordRetriever:
    """Simple, dependency-free keyword search over the vault.

    Scores files by token overlap between the query and file content,
    with a bonus for matches in the filename. This is intentionally
    simple -- it exists to be correct and predictable, not clever.
    """

    def __init__(self, vault_dir: Path) -> None:
        self._vault_dir = vault_dir

    def _iter_markdown_files(self) -> list[Path]:
        return sorted(self._vault_dir.rglob("*.md"))

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_tokens = word_tokens(query)
        if not query_tokens:
            return []

        hits: list[SearchHit] = []
        for path in self._iter_markdown_files():
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.warning("Could not read %s during search: %s", path, exc)
                continue

            content_tokens = word_tokens(content)
            filename_tokens = word_tokens(path.stem.replace("_", " "))

            overlap = query_tokens & content_tokens
            filename_overlap = query_tokens & filename_tokens

            if not overlap and not filename_overlap:
                continue

            score = len(overlap) + 2.0 * len(filename_overlap)
            relative = str(path.relative_to(self._vault_dir)).replace("\\", "/")
            snippet = _make_snippet(content, query_tokens)
            hits.append(SearchHit(path=relative, score=score, snippet=snippet))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def index_file(self, relative_path: str, content: str) -> None:
        # Keyword search reads files directly from disk on every
        # query, so there is no index to maintain here. This method
        # exists purely to satisfy the Retriever interface so a real
        # (embedding-based) retriever can be swapped in later without
        # touching call sites.
        return

    def remove_file(self, relative_path: str) -> None:
        return


def _make_snippet(content: str, query_tokens: set[str], radius: int = 80) -> str:
    """Return a short excerpt of content around the first query-token hit."""
    lowered = content.lower()
    for token in query_tokens:
        idx = lowered.find(token)
        if idx != -1:
            start = max(0, idx - radius)
            end = min(len(content), idx + radius)
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(content) else ""
            return prefix + content[start:end].strip().replace("\n", " ") + suffix
    return content[:radius].strip().replace("\n", " ")


class VaultMemory:
    """High-level API for interacting with the vault.

    This is the class agent.py and tools.py should use -- they should
    never touch pathlib or the filesystem directly.
    """

    def __init__(self, retriever: Retriever | None = None) -> None:
        CONFIG.ensure_vault_structure()
        self._vault_dir = CONFIG.vault_dir
        self._retriever: Retriever = retriever or KeywordRetriever(self._vault_dir)

    # -- basic file operations -------------------------------------------------

    def read(self, relative_path: str) -> str:
        path = resolve_safe_path(relative_path)
        if not path.exists():
            raise VaultError(f"File not found: {relative_path}")
        if not path.is_file():
            raise VaultError(f"Not a file: {relative_path}")
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise VaultError(f"Could not read {relative_path}: {exc}") from exc

    def read_multiple(self, relative_paths: list[str]) -> dict[str, str]:
        results: dict[str, str] = {}
        for rel in relative_paths:
            try:
                results[rel] = self.read(rel)
            except (VaultError, UnsafePathError) as exc:
                results[rel] = f"[error: {exc}]"
        return results

    def write(self, relative_path: str, content: str) -> None:
        path = resolve_safe_path(relative_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise VaultError(f"Could not write {relative_path}: {exc}") from exc
        self._retriever.index_file(relative_path, content)
        logger.info("Wrote file: %s (%d chars)", relative_path, len(content))

    def append(self, relative_path: str, content: str) -> None:
        path = resolve_safe_path(relative_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            separator = "\n" if existing and not existing.endswith("\n") else ""
            new_content = existing + separator + content
            if not new_content.endswith("\n"):
                new_content += "\n"
            path.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            raise VaultError(f"Could not append to {relative_path}: {exc}") from exc
        self._retriever.index_file(relative_path, new_content)
        logger.info("Appended to file: %s (+%d chars)", relative_path, len(content))

    def remove(self, relative_path: str) -> None:
        path = resolve_safe_path(relative_path)
        if not path.exists():
            raise VaultError(f"File not found: {relative_path}")
        try:
            path.unlink()
        except OSError as exc:
            raise VaultError(f"Could not remove {relative_path}: {exc}") from exc
        self._retriever.remove_file(relative_path)
        logger.info("Removed file: %s", relative_path)

    def create_folder(self, relative_folder: str) -> None:
        path = resolve_safe_path(relative_folder)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VaultError(f"Could not create folder {relative_folder}: {exc}") from exc
        logger.info("Created folder: %s", relative_folder)

    def list_files(self, relative_folder: str | None = None) -> list[str]:
        base = resolve_safe_path(relative_folder) if relative_folder else self._vault_dir
        if not base.exists():
            raise VaultError(f"Folder not found: {relative_folder or '.'}")
        if not base.is_dir():
            raise VaultError(f"Not a folder: {relative_folder}")
        files = sorted(p for p in base.rglob("*") if p.is_file())
        return [str(p.relative_to(self._vault_dir)).replace("\\", "/") for p in files]

    # -- search / retrieval -----------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        k = top_k or CONFIG.max_search_results
        logger.info("Searching vault for query=%r top_k=%d", query, k)
        return self._retriever.search(query, k)

    def retrieve_context_for(self, user_message: str) -> list[tuple[str, str]]:
        """Automatically retrieve the most relevant files for a message.

        Returns a list of (relative_path, content) tuples, capped at
        CONFIG.max_context_files, with each file's content truncated
        to CONFIG.max_file_chars_in_context. This is what agent.py
        calls before sending the user's message to the LLM.
        """
        hits = self.search(user_message, top_k=CONFIG.max_context_files)
        context: list[tuple[str, str]] = []
        for hit in hits:
            try:
                content = self.read(hit.path)
            except VaultError as exc:
                logger.warning("Retrieval could not read %s: %s", hit.path, exc)
                continue
            from utils import truncate

            context.append((hit.path, truncate(content, CONFIG.max_file_chars_in_context)))
        return context

    def read_resources(self) -> list[tuple[str, str]]:
        """Read every markdown file in vault/resources/, truncated the
        same way as retrieve_context_for.

        Unlike retrieve_context_for, this ignores relevance entirely --
        resources/ is meant to be reference material the user has
        deliberately placed for the assistant to always have on hand
        (e.g. a project brief, a style guide), not something surfaced
        only when it happens to match the current message. Capped at
        CONFIG.max_resource_files so a large folder can't blow the
        context window.
        """
        resources_dir = self._vault_dir / "resources"
        if not resources_dir.is_dir():
            return []

        from utils import truncate

        results: list[tuple[str, str]] = []
        for path in sorted(resources_dir.rglob("*.md"))[: CONFIG.max_resource_files]:
            relative = str(path.relative_to(self._vault_dir)).replace("\\", "/")
            try:
                content = self.read(relative)
            except VaultError as exc:
                logger.warning("Could not read resource %s: %s", relative, exc)
                continue
            results.append((relative, truncate(content, CONFIG.max_file_chars_in_context)))
        return results
