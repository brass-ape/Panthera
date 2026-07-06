"""
parser.py
=========

Parses raw text produced by the LLM into either:

* a validated ToolCall, or
* plain-text final-answer content.

The model is instructed (see prompts.py) to emit a tool call as a
single bare JSON object with no surrounding prose. This module is
deliberately strict: if the JSON is malformed or fails schema
validation, we treat the response as a final answer rather than
guessing at the model's intent, and we log the failure so it is
visible during development.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("assistant.parser")

# Plugin tool names double as Python module/identifier names (see
# plugins.py), so they're restricted to a conservative bare-identifier
# charset -- no path separators, no "..", nothing that could escape
# vault/plugins_proposed/ when used as a filename.
_PLUGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

VALID_TOOLS = {
    "read",
    "write",
    "append",
    "remove",
    "search",
    "list_files",
    "read_multiple",
    "create_folder",
    "web_search",
    "web_fetch",
    "propose_plugin",
    "none",  # used only by the memory-creation prompt
}

# Models trained on other agent frameworks' tool-naming conventions
# reach for plausible-sounding synonyms (e.g. "create_file") instead of
# this app's actual tool names, even with TOOL_INSTRUCTIONS spelling
# the real ones out. Rather than losing that tool call entirely (see
# validate_tool_call, which would otherwise raise "Unknown or invalid
# tool"), rewrite known synonyms to the real tool name before
# validating -- same rationale as json.loads(strict=False) above: the
# model's intent is unambiguous, only the exact spelling was wrong.
# This does NOT bypass argument/path validation -- an aliased call
# still has to satisfy the real tool's schema afterward.
_TOOL_ALIASES: dict[str, str] = {
    "create_file": "write",
    "write_file": "write",
    "save_file": "write",
    "edit_file": "write",
    "update_file": "write",
    "delete_file": "remove",
    "remove_file": "remove",
    "read_file": "read",
    "get_file": "read",
    "list_dir": "list_files",
    "list_directory": "list_files",
    "ls": "list_files",
    "mkdir": "create_folder",
    "make_folder": "create_folder",
    "make_directory": "create_folder",
    "create_directory": "create_folder",
    "search_web": "web_search",
    "websearch": "web_search",
    "fetch_url": "web_fetch",
    "fetch_web": "web_fetch",
    "get_url": "web_fetch",
}

# Required argument names per tool, beyond "tool" itself.
# NOTE: register_loaded_plugins() (see plugins.py) extends this dict
# at startup with any human-approved plugin tools -- this is the
# built-in set only.
_REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "read": ("file",),
    "write": ("file", "content"),
    "append": ("file", "content"),
    "remove": ("file",),
    "search": ("query",),
    "list_files": (),  # "folder" optional, defaults to vault root
    "read_multiple": ("files",),
    "create_folder": ("folder",),
    "web_search": ("query",),
    "web_fetch": ("url",),
    "propose_plugin": ("name", "description", "code"),
    "none": (),
}


class ToolCallValidationError(Exception):
    """Raised when a parsed JSON object does not match the tool schema."""


@dataclass
class ToolCall:
    """A validated request to execute a tool."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedResponse:
    """Result of parsing one LLM turn.

    Exactly one of `tool_call` or `final_text` will be set.
    """

    tool_call: Optional[ToolCall] = None
    final_text: Optional[str] = None

    @property
    def is_tool_call(self) -> bool:
        return self.tool_call is not None


def _extract_json_object(text: str) -> Optional[str]:
    """Best-effort extraction of a single top-level JSON object from text.

    Handles the common case of the model wrapping JSON in markdown
    fences despite instructions not to, and the case of a bare object.
    Returns None if no plausible JSON object is found.
    """
    stripped = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    if not stripped.startswith("{"):
        # Look for the first '{' ... last '}' as a fallback, but only
        # trust it if it spans almost the whole response -- otherwise
        # this is prose that merely mentions a brace somewhere.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = stripped[start : end + 1]
        # Require the JSON-looking region to dominate the response,
        # or we risk mis-parsing prose that happens to contain braces.
        if len(candidate) < 0.8 * len(stripped):
            return None
        return candidate

    return stripped


def validate_tool_call(obj: dict[str, Any]) -> ToolCall:
    """Validate a decoded JSON object against the tool schema.

    Raises ToolCallValidationError on any mismatch.
    """
    if "tool" not in obj:
        raise ToolCallValidationError("Missing required 'tool' field.")

    tool = obj["tool"]
    if isinstance(tool, str) and tool in _TOOL_ALIASES:
        logger.debug("Rewriting tool alias %r to %r.", tool, _TOOL_ALIASES[tool])
        tool = _TOOL_ALIASES[tool]
    if not isinstance(tool, str) or tool not in VALID_TOOLS:
        raise ToolCallValidationError(f"Unknown or invalid tool: {tool!r}")

    required = _REQUIRED_ARGS[tool]
    missing = [name for name in required if name not in obj]
    if missing:
        raise ToolCallValidationError(
            f"Tool {tool!r} is missing required argument(s): {missing}"
        )

    if tool == "read_multiple":
        files = obj.get("files")
        if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
            raise ToolCallValidationError("'files' must be a list of strings.")

    if tool == "propose_plugin":
        name = obj.get("name")
        if not isinstance(name, str) or not _PLUGIN_NAME_RE.match(name):
            raise ToolCallValidationError(
                "'name' must be a lowercase identifier matching "
                f"{_PLUGIN_NAME_RE.pattern!r} (no paths, no traversal)."
            )

    args = {k: v for k, v in obj.items() if k != "tool"}
    return ToolCall(tool=tool, args=args)


def parse_llm_response(text: str) -> ParsedResponse:
    """Parse a raw LLM response into a tool call or final answer text.

    This never raises: any parsing/validation failure simply results
    in the raw text being treated as the final answer, since that is
    the safest fallback (the user still gets a response).
    """
    candidate = _extract_json_object(text)
    if candidate is None:
        return ParsedResponse(final_text=text.strip())

    try:
        # strict=False allows raw control characters (unescaped literal
        # newlines/tabs) inside JSON string values. Small local models
        # very commonly emit multi-line "content" fields as literal
        # newlines instead of "\n" escapes -- that's invalid JSON by
        # the strict spec, but the intent is completely unambiguous,
        # so tolerating it here recovers a real, well-formed tool call
        # instead of falling back to dumping the raw JSON as the
        # "final answer" (this is not a schema relaxation -- the
        # object still has to fully validate below; it just stops a
        # cosmetic escaping slip from discarding an otherwise-valid
        # tool call).
        obj = json.loads(candidate, strict=False)
    except json.JSONDecodeError as exc:
        logger.debug("JSON decode failed, treating response as final text: %s", exc)
        return ParsedResponse(final_text=text.strip())

    if not isinstance(obj, dict):
        return ParsedResponse(final_text=text.strip())

    try:
        tool_call = validate_tool_call(obj)
    except ToolCallValidationError as exc:
        logger.warning("Tool call failed schema validation (%s); treating as text.", exc)
        return ParsedResponse(final_text=text.strip())

    return ParsedResponse(tool_call=tool_call)
