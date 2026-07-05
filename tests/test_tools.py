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
