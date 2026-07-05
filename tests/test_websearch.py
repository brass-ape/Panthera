from __future__ import annotations

from types import SimpleNamespace

import pytest

import websearch
from websearch import (
    BraveSearchProvider,
    DuckDuckGoSearchProvider,
    WebSearchBackendError,
    build_search_provider,
    html_to_text,
)


class FakeResponse:
    def __init__(self, text="", json_data=None, headers=None, status_code=200):
        self.text = text
        self._json_data = json_data
        self.headers = headers or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class TestHtmlToText:
    def test_strips_tags_and_keeps_text(self):
        html = "<html><body><h1>Title</h1><p>Some text.</p></body></html>"
        assert html_to_text(html) == "Title\nSome text."

    def test_drops_script_and_style_content(self):
        html = "<p>Visible</p><script>alert('x')</script><style>.a{}</style>"
        assert html_to_text(html) == "Visible"


class TestDuckDuckGoSearchProvider:
    def test_search_parses_results(self, monkeypatch):
        html = (
            '<a class="result__a" href="https://example.com">Example Title</a>'
            '<a class="result__snippet">An example snippet</a>'
        )
        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(text=html))

        provider = DuckDuckGoSearchProvider()
        hits = provider.search("example", top_k=5)
        assert len(hits) == 1
        assert hits[0].title == "Example Title"
        assert hits[0].url == "https://example.com"
        assert hits[0].snippet == "An example snippet"

    def test_search_respects_top_k(self, monkeypatch):
        result_block = (
            '<a class="result__a" href="https://example.com/{i}">Title {i}</a>'
            '<a class="result__snippet">Snippet {i}</a>'
        )
        html = "".join(result_block.format(i=i) for i in range(5))

        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(text=html))

        provider = DuckDuckGoSearchProvider()
        hits = provider.search("example", top_k=2)
        assert len(hits) == 2

    def test_request_failure_raises_backend_error(self, monkeypatch):
        import requests

        def raise_error(*args, **kwargs):
            raise requests.RequestException("network down")

        monkeypatch.setattr(requests, "post", raise_error)

        provider = DuckDuckGoSearchProvider()
        with pytest.raises(WebSearchBackendError):
            provider.search("example", top_k=5)


class TestBraveSearchProvider:
    def test_missing_api_key_raises_immediately(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        with pytest.raises(WebSearchBackendError):
            BraveSearchProvider()

    def test_search_parses_json_results(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        import requests

        payload = {
            "web": {
                "results": [
                    {"title": "A", "url": "https://a.example", "description": "desc a"},
                    {"title": "B", "url": "https://b.example", "description": "desc b"},
                ]
            }
        }
        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(json_data=payload))

        provider = BraveSearchProvider()
        hits = provider.search("query", top_k=1)
        assert len(hits) == 1
        assert hits[0].title == "A"
        assert hits[0].url == "https://a.example"


class TestBuildSearchProvider:
    """build_search_provider() only reads CONFIG.web_search_backend, so a
    minimal stand-in swapped in for the module-level `CONFIG` name is
    enough -- CONFIG is a frozen dataclass instance and can't have its
    fields reassigned directly.
    """

    def test_none_backend_returns_none(self, monkeypatch):
        monkeypatch.setattr(websearch, "CONFIG", SimpleNamespace(web_search_backend="none"))
        assert build_search_provider() is None

    def test_duckduckgo_backend_builds_provider(self, monkeypatch):
        monkeypatch.setattr(websearch, "CONFIG", SimpleNamespace(web_search_backend="duckduckgo"))
        assert isinstance(build_search_provider(), DuckDuckGoSearchProvider)

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setattr(websearch, "CONFIG", SimpleNamespace(web_search_backend="not-a-real-backend"))
        with pytest.raises(WebSearchBackendError):
            build_search_provider()


class TestFetchUrl:
    def test_rejects_non_http_scheme(self):
        with pytest.raises(WebSearchBackendError):
            websearch.fetch_url("file:///etc/passwd")

    def test_fetches_and_strips_html(self, monkeypatch):
        import requests

        html_response = FakeResponse(
            text="<p>Hello page</p>",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )
        monkeypatch.setattr(requests, "get", lambda *a, **k: html_response)
        assert websearch.fetch_url("https://example.com") == "Hello page"

    def test_fetches_plain_text_unchanged(self, monkeypatch):
        import requests

        text_response = FakeResponse(text="raw text", headers={"Content-Type": "text/plain"})
        monkeypatch.setattr(requests, "get", lambda *a, **k: text_response)
        assert websearch.fetch_url("https://example.com/file.txt") == "raw text"
