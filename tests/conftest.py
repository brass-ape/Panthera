"""Shared pytest fixtures for the assistant test suite."""

from __future__ import annotations

import pytest

from config import Config


@pytest.fixture
def vault_dir(tmp_path, monkeypatch):
    """Point CONFIG.vault_dir at an isolated tmp_path for the duration
    of a test, and make sure the standard subfolders exist.

    Config.vault_dir is a property on the frozen Config dataclass, so
    instance attributes can't be reassigned directly; patching the
    property at the class level (reverted automatically by monkeypatch)
    is the cleanest way to redirect it without touching production code.
    """
    from config import CONFIG

    monkeypatch.setattr(Config, "vault_dir", property(lambda self: tmp_path))
    CONFIG.ensure_vault_structure()
    return tmp_path


class ConfigOverride:
    """Forwards attribute access to the real CONFIG singleton except
    for explicitly overridden names.

    CONFIG's fields (unlike `vault_dir`) are plain dataclass fields
    living in the frozen instance's __dict__, so patching `Config`
    class attributes doesn't affect the already-constructed instance.
    Swapping the module-level `CONFIG` name a module imported (e.g.
    `agent.CONFIG`) for one of these is the reliable way to override a
    single setting for a test.
    """

    def __init__(self, **overrides):
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        from config import CONFIG as real_config

        if name in self._overrides:
            return self._overrides[name]
        return getattr(real_config, name)


@pytest.fixture
def config_override(monkeypatch):
    """Returns a function to override one or more CONFIG fields as seen
    by a given module, e.g. `config_override(agent_module, max_tool_iterations=2)`.
    """

    def _apply(module, **overrides):
        monkeypatch.setattr(module, "CONFIG", ConfigOverride(**overrides))

    return _apply
