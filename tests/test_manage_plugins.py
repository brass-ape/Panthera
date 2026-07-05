from __future__ import annotations

import json

import pytest

import manage_plugins
import plugins as plugins_module


@pytest.fixture
def isolated_dirs(vault_dir, tmp_path, monkeypatch):
    """vault_dir already redirects CONFIG.vault_dir; manage_plugins.py
    computed PROPOSED_DIR from CONFIG.vault_dir at import time, so it
    needs its own patch too. PLUGINS_DIR gets redirected to a fresh
    tmp_path so approving never touches the real project's plugins/.
    """
    proposed_dir = vault_dir / "plugins_proposed"
    proposed_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(manage_plugins, "PROPOSED_DIR", proposed_dir)

    plugins_dir = tmp_path / "plugins"
    monkeypatch.setattr(manage_plugins, "PLUGINS_DIR", plugins_dir)
    monkeypatch.setattr(plugins_module, "PLUGINS_DIR", plugins_dir)

    return proposed_dir, plugins_dir


def _propose(proposed_dir, name="roll_dice", description="Rolls a die", code="TOOL_NAME = 'roll_dice'\n"):
    (proposed_dir / f"{name}.py").write_text(code)
    manifest = {"name": name, "description": description, "proposed_at": "2026-01-01T00:00:00+00:00", "status": "pending"}
    (proposed_dir / f"{name}.manifest.json").write_text(json.dumps(manifest))


class Args:
    """Minimal stand-in for argparse.Namespace."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestList:
    def test_list_with_nothing(self, isolated_dirs, capsys):
        manage_plugins.cmd_list(Args())
        out = capsys.readouterr().out
        assert "(none)" in out

    def test_list_shows_pending_and_approved(self, isolated_dirs, capsys):
        proposed_dir, plugins_dir = isolated_dirs
        _propose(proposed_dir)
        plugins_dir.mkdir(parents=True, exist_ok=True)
        (plugins_dir / "already_approved.py").write_text("TOOL_NAME = 'x'\n")

        manage_plugins.cmd_list(Args())
        out = capsys.readouterr().out
        assert "roll_dice" in out
        assert "already_approved" in out


class TestShow:
    def test_show_missing_proposal_exits(self, isolated_dirs):
        with pytest.raises(SystemExit):
            manage_plugins.cmd_show(Args(name="nonexistent"))

    def test_show_prints_source_and_manifest(self, isolated_dirs, capsys):
        proposed_dir, _ = isolated_dirs
        _propose(proposed_dir)
        manage_plugins.cmd_show(Args(name="roll_dice"))
        out = capsys.readouterr().out
        assert "Rolls a die" in out
        assert "TOOL_NAME = 'roll_dice'" in out

    def test_show_rejects_invalid_name(self, isolated_dirs):
        with pytest.raises(SystemExit):
            manage_plugins.cmd_show(Args(name="../escape"))


class TestApprove:
    def test_approve_copies_to_plugins_and_removes_proposal(self, isolated_dirs):
        proposed_dir, plugins_dir = isolated_dirs
        _propose(proposed_dir)

        manage_plugins.cmd_approve(Args(name="roll_dice", force=False))

        assert (plugins_dir / "roll_dice.py").read_text() == "TOOL_NAME = 'roll_dice'\n"
        assert not (proposed_dir / "roll_dice.py").exists()
        assert not (proposed_dir / "roll_dice.manifest.json").exists()

    def test_approve_missing_proposal_exits(self, isolated_dirs):
        with pytest.raises(SystemExit):
            manage_plugins.cmd_approve(Args(name="nonexistent", force=False))

    def test_approve_refuses_to_overwrite_without_force(self, isolated_dirs):
        proposed_dir, plugins_dir = isolated_dirs
        _propose(proposed_dir)
        plugins_dir.mkdir(parents=True, exist_ok=True)
        (plugins_dir / "roll_dice.py").write_text("existing content")

        with pytest.raises(SystemExit):
            manage_plugins.cmd_approve(Args(name="roll_dice", force=False))
        assert (plugins_dir / "roll_dice.py").read_text() == "existing content"

    def test_approve_overwrites_with_force(self, isolated_dirs):
        proposed_dir, plugins_dir = isolated_dirs
        _propose(proposed_dir)
        plugins_dir.mkdir(parents=True, exist_ok=True)
        (plugins_dir / "roll_dice.py").write_text("existing content")

        manage_plugins.cmd_approve(Args(name="roll_dice", force=True))
        assert (plugins_dir / "roll_dice.py").read_text() == "TOOL_NAME = 'roll_dice'\n"


class TestReject:
    def test_reject_removes_proposal(self, isolated_dirs):
        proposed_dir, plugins_dir = isolated_dirs
        _propose(proposed_dir)

        manage_plugins.cmd_reject(Args(name="roll_dice"))

        assert not (proposed_dir / "roll_dice.py").exists()
        assert not (proposed_dir / "roll_dice.manifest.json").exists()
        assert not (plugins_dir / "roll_dice.py").exists()

    def test_reject_missing_proposal_exits(self, isolated_dirs):
        with pytest.raises(SystemExit):
            manage_plugins.cmd_reject(Args(name="nonexistent"))
