from __future__ import annotations

import pytest

from memory import VaultMemory
from parser import ToolCall
from tools import ToolExecutor
from websearch import WebSearchHit


@pytest.fixture
def executor(vault_dir):
    return ToolExecutor(VaultMemory())


class TestBasicToolHandlers:
    def test_write_then_read(self, executor):
        executor.execute(ToolCall(tool="write", args={"file": "facts/a.md", "content": "hello"}))
        result = executor.execute(ToolCall(tool="read", args={"file": "facts/a.md"}))
        assert "hello" in result

    def test_unknown_tool_returns_error_string(self, executor):
        result = executor.execute(ToolCall(tool="not_a_real_tool", args={}))
        assert result.startswith("[tool error]")

    def test_read_missing_file_returns_error_string_not_exception(self, executor):
        result = executor.execute(ToolCall(tool="read", args={"file": "facts/missing.md"}))
        assert result.startswith("[tool error]")

    def test_path_traversal_is_rejected_as_tool_error(self, executor):
        result = executor.execute(ToolCall(tool="read", args={"file": "../escape.md"}))
        assert result.startswith("[tool error]")

    def test_list_files_empty_vault(self, executor):
        result = executor.execute(ToolCall(tool="list_files", args={}))
        assert "No files found" in result


class TestWebSearchTool:
    def test_web_search_wraps_results_as_untrusted(self, executor, monkeypatch):
        import tools

        class FakeProvider:
            def search(self, query, top_k):
                return [WebSearchHit(title="Example", url="https://example.com", snippet="a snippet")]

        monkeypatch.setattr(tools, "build_search_provider", lambda: FakeProvider())
        result = executor.execute(ToolCall(tool="web_search", args={"query": "example"}))
        assert "BEGIN UNTRUSTED WEB CONTENT" in result
        assert "END UNTRUSTED WEB CONTENT" in result
        assert "Example" in result
        assert "https://example.com" in result

    def test_web_search_disabled_backend_reports_tool_error(self, executor, monkeypatch):
        import tools

        monkeypatch.setattr(tools, "build_search_provider", lambda: None)
        result = executor.execute(ToolCall(tool="web_search", args={"query": "example"}))
        assert result.startswith("[tool error]")
        assert "disabled" in result

    def test_web_search_no_hits_still_wrapped_as_untrusted(self, executor, monkeypatch):
        import tools

        class EmptyProvider:
            def search(self, query, top_k):
                return []

        monkeypatch.setattr(tools, "build_search_provider", lambda: EmptyProvider())
        result = executor.execute(ToolCall(tool="web_search", args={"query": "nothing"}))
        assert "BEGIN UNTRUSTED WEB CONTENT" in result
        assert "No web results found" in result


class TestProposePluginTool:
    def test_writes_source_and_manifest_without_executing(self, executor, vault_dir):
        result = executor.execute(
            ToolCall(
                tool="propose_plugin",
                args={
                    "name": "roll_dice",
                    "description": "Rolls a die",
                    "code": "TOOL_NAME = 'roll_dice'\n",
                },
            )
        )
        assert "not active" in result
        assert "manage_plugins.py approve roll_dice" in result

        source = vault_dir / "plugins_proposed" / "roll_dice.py"
        manifest = vault_dir / "plugins_proposed" / "roll_dice.manifest.json"
        assert source.read_text() == "TOOL_NAME = 'roll_dice'\n"
        assert manifest.exists()

        import json

        data = json.loads(manifest.read_text())
        assert data["name"] == "roll_dice"
        assert data["description"] == "Rolls a die"
        assert data["status"] == "pending"

    def test_does_not_touch_the_real_plugins_directory(self, executor, vault_dir):
        import plugins as plugins_module

        executor.execute(
            ToolCall(
                tool="propose_plugin",
                args={"name": "sneaky", "description": "x", "code": "import os; os.system('echo hi')"},
            )
        )
        # Nothing should exist in the trusted plugins/ dir just because
        # a plugin was proposed -- only manage_plugins.py can put
        # something there.
        assert not (plugins_module.PLUGINS_DIR / "sneaky.py").exists()


class TestWebFetchTool:
    def test_web_fetch_wraps_content_as_untrusted(self, executor, monkeypatch):
        import tools

        monkeypatch.setattr(tools, "fetch_url", lambda url: "short page content")
        result = executor.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))
        assert "BEGIN UNTRUSTED WEB CONTENT" in result
        assert "short page content" in result
        assert "https://example.com" in result

    def test_web_fetch_truncates_long_content(self, executor, monkeypatch):
        import tools

        # Longer than CONFIG.max_web_fetch_chars' default (4000).
        monkeypatch.setattr(tools, "fetch_url", lambda url: "x" * 10000)
        result = executor.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))
        assert "[truncated]" in result
