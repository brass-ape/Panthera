"""
plugins.py
==========

Lets the assistant extend its own tool set -- with a mandatory human
review step in between "the model wrote some code" and "that code
runs". This is deliberately NOT a code-execution tool: the model can
only ever write text to a sandboxed vault folder (see tools.py's
propose_plugin handler); nothing it writes is imported or executed
until a human runs manage_plugins.py to approve it into plugins/,
which only takes effect the next time the app starts.

Flow
----
1. Model calls the "propose_plugin" tool -> writes
   vault/plugins_proposed/<name>.py (+ a .manifest.json sidecar) via
   the normal sandboxed VaultMemory.write path. Inert text, nothing
   runs.
2. A human reviews it: `python manage_plugins.py show <name>`.
3. A human approves it: `python manage_plugins.py approve <name>`
   -- copies the file into plugins/ (trusted, project-level, NOT
   inside the vault sandbox) and removes the proposal.
4. On the next process start, main.py/webapp.py call
   register_loaded_plugins(), which imports everything in plugins/
   and registers each valid module as a new tool.

Plugin contract
---------------
A file in plugins/ is loaded as a tool only if it defines all of:
    TOOL_NAME: str
    REQUIRED_ARGS: tuple[str, ...]
    DESCRIPTION: str
    def handle(memory: VaultMemory, args: dict) -> str: ...
following the same handler shape as tools.py's built-in handlers.
Anything that fails to import, or doesn't define all four, is skipped
(logged, not fatal) -- one broken plugin can't take down the app.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import CONFIG
from memory import VaultMemory

logger = logging.getLogger("assistant.plugins")

PLUGINS_DIR = CONFIG.project_root / "plugins"
PROPOSED_SUBDIR = "plugins_proposed"  # vault-relative

_REQUIRED_ATTRS = ("TOOL_NAME", "REQUIRED_ARGS", "DESCRIPTION", "handle")


@dataclass
class PluginSpec:
    """A validated, loaded plugin ready to be registered as a tool."""

    tool_name: str
    required_args: tuple[str, ...]
    description: str
    handle: Callable[[VaultMemory, dict], str]
    source_file: Path


class PluginError(Exception):
    """Raised for plugin-system failures that should be logged and
    skipped rather than crashing startup.
    """


def _load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(f"assistant_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise PluginError(f"Could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_plugin_module(module, path: Path) -> PluginSpec:
    missing = [attr for attr in _REQUIRED_ATTRS if not hasattr(module, attr)]
    if missing:
        raise PluginError(f"{path.name} is missing required attribute(s): {missing}")

    tool_name = module.TOOL_NAME
    required_args = module.REQUIRED_ARGS
    description = module.DESCRIPTION
    handle = module.handle

    if not isinstance(tool_name, str) or not tool_name:
        raise PluginError(f"{path.name}: TOOL_NAME must be a non-empty string")
    if not isinstance(required_args, tuple) or not all(isinstance(a, str) for a in required_args):
        raise PluginError(f"{path.name}: REQUIRED_ARGS must be a tuple of strings")
    if not callable(handle):
        raise PluginError(f"{path.name}: handle must be callable")

    return PluginSpec(
        tool_name=tool_name,
        required_args=required_args,
        description=str(description),
        handle=handle,
        source_file=path,
    )


def load_approved_plugins() -> list[PluginSpec]:
    """Import every *.py file in plugins/ (project root, NOT the vault
    -- this is the trusted, human-curated directory) and return the
    ones that satisfy the plugin contract. Anything that fails to
    import or doesn't match the contract is logged and skipped, never
    raised -- one bad plugin file must not prevent the app from
    starting.
    """
    if not PLUGINS_DIR.is_dir():
        return []

    specs: list[PluginSpec] = []
    for path in sorted(PLUGINS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue  # e.g. __init__.py, _helpers.py -- not a tool itself
        try:
            module = _load_module_from_path(path)
            specs.append(_validate_plugin_module(module, path))
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not be fatal
            logger.warning("Skipping plugin %s: %s", path.name, exc)
            continue

    return specs


_registered_specs: list[PluginSpec] | None = None


def register_loaded_plugins() -> list[PluginSpec]:
    """Load plugins/ and register each one as a real tool: extends
    parser.VALID_TOOLS/_REQUIRED_ARGS and tools._HANDLERS in place.

    Safe to call more than once (e.g. Agent() may be reconstructed
    after a settings change) -- only does the actual import/register
    work the first time and returns the cached specs afterwards, so a
    plugin tool that was already registered as "existing" doesn't spam
    a skip-warning on every subsequent Agent construction.
    """
    global _registered_specs
    if _registered_specs is not None:
        return _registered_specs

    import parser as parser_module
    import tools as tools_module

    specs = load_approved_plugins()
    for spec in specs:
        if spec.tool_name in parser_module.VALID_TOOLS:
            logger.warning(
                "Plugin %s wants tool name %r, which already exists -- skipping.",
                spec.source_file.name,
                spec.tool_name,
            )
            continue
        parser_module.VALID_TOOLS.add(spec.tool_name)
        parser_module._REQUIRED_ARGS[spec.tool_name] = spec.required_args
        tools_module._HANDLERS[spec.tool_name] = spec.handle
        logger.info("Registered plugin tool: %s (from %s)", spec.tool_name, spec.source_file.name)

    _registered_specs = specs
    return specs


def describe_loaded_plugins(specs: list[PluginSpec]) -> str:
    """Format loaded plugin tools for insertion into the system
    prompt, so the model knows they exist -- prompts.py's static
    TOOL_INSTRUCTIONS can't describe tools it doesn't know about yet.
    Returns "" if there are none (callers should omit the block).
    """
    if not specs:
        return ""
    lines = ["Additional tools available in this installation (human-approved plugins):"]
    for spec in specs:
        args = ", ".join(spec.required_args) or "no arguments"
        lines.append(f'- {spec.tool_name} ({args}): {spec.description}')
        lines.append(
            f'  {{"tool": "{spec.tool_name}"' + "".join(f', "{a}": ...' for a in spec.required_args) + "}"
        )
    return "\n".join(lines)
