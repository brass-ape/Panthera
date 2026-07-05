"""
utils.py
========

Small, dependency-free helper utilities shared across the codebase.

The most important thing in this module is `resolve_safe_path`, which
is the single choke point every filesystem-touching tool call must go
through. Nothing outside this function should decide whether a path
is "safe" -- that keeps the security logic in one auditable place.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from config import CONFIG

logger = logging.getLogger("assistant.utils")


class UnsafePathError(Exception):
    """Raised whenever a requested file path would escape the vault,
    use an absolute path, contain traversal sequences, or point at a
    symlink leaving the sandbox.
    """


# Filenames are restricted to a conservative charset: letters, digits,
# spaces, dashes, underscores, dots (for extensions) and forward
# slashes (for subfolders). Anything else is rejected outright rather
# than "cleaned", because silently rewriting a path is how traversal
# bugs sneak in.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9 _\-./]+$")


def resolve_safe_path(relative_path: str) -> Path:
    """Resolve a user/model-supplied relative path against the vault root.

    Raises UnsafePathError for anything that looks like it is trying to
    escape the sandbox: absolute paths, ``..`` traversal, null bytes,
    disallowed characters, or symlinks that resolve outside the vault.

    Args:
        relative_path: A path like ``"people/me.md"`` as supplied by a
            tool call.

    Returns:
        The resolved, absolute Path guaranteed to live inside the vault.
    """
    if relative_path is None:
        raise UnsafePathError("No path was provided.")

    candidate = unicodedata.normalize("NFKC", relative_path).strip()

    if not candidate:
        raise UnsafePathError("Empty path is not allowed.")

    if "\x00" in candidate:
        raise UnsafePathError("Null bytes are not allowed in paths.")

    # Reject absolute paths outright (covers both POSIX and Windows
    # style, e.g. "/etc/passwd" or "C:\\Windows").
    if candidate.startswith("/") or candidate.startswith("\\"):
        raise UnsafePathError(f"Absolute paths are not allowed: {relative_path!r}")
    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        raise UnsafePathError(f"Absolute paths are not allowed: {relative_path!r}")

    # Reject traversal sequences before we even touch the filesystem.
    normalized_slashes = candidate.replace("\\", "/")
    parts = normalized_slashes.split("/")
    if any(part == ".." for part in parts):
        raise UnsafePathError(f"Path traversal is not allowed: {relative_path!r}")

    if not _SAFE_PATH_RE.match(candidate):
        raise UnsafePathError(f"Path contains disallowed characters: {relative_path!r}")

    vault_root = CONFIG.vault_dir
    target = (vault_root / normalized_slashes).resolve()

    # The definitive check: after resolution (which also follows
    # symlinks), the target must still live under the vault root.
    try:
        target.relative_to(vault_root)
    except ValueError as exc:
        raise UnsafePathError(
            f"Resolved path escapes the vault: {relative_path!r} -> {target}"
        ) from exc

    # Explicitly reject any symlink anywhere along the path that
    # points outside the vault, even if the final resolved path
    # happens to land inside it (defence in depth).
    probe = vault_root
    for part in normalized_slashes.split("/"):
        if not part:
            continue
        probe = probe / part
        if probe.is_symlink():
            real = probe.resolve()
            try:
                real.relative_to(vault_root)
            except ValueError as exc:
                raise UnsafePathError(
                    f"Symlink escapes the vault: {probe} -> {real}"
                ) from exc

    return target


def truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending a marker if it was cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n...[truncated]"


def today_journal_filename() -> str:
    """Return today's journal filename, e.g. 'journal/2026-07-05.md'."""
    from datetime import date

    return f"journal/{date.today().isoformat()}.md"


def word_tokens(text: str) -> set[str]:
    """Lowercase, alnum-only tokenization used by the keyword search."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t}
