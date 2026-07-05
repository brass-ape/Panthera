"""
websearch.py
============

The assistant's only source of outbound internet access besides the
local Ollama calls in agent.py. Provides a `SearchProvider` Protocol
(mirroring `embeddings.py`'s `EmbeddingProvider` pattern) so the
concrete backend can be swapped via `CONFIG.web_search_backend`
without touching tools.py or agent.py.

Content returned from here is untrusted external text -- callers
(tools.py) are responsible for delimiting it clearly before it goes
back into the model's context, per the prompt-injection guidance in
prompts.py's TOOL_INSTRUCTIONS.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol

from config import CONFIG

logger = logging.getLogger("assistant.websearch")


class WebSearchBackendError(Exception):
    """Raised when a web search/fetch backend fails (offline, bad response,
    missing API key, etc.).
    """


@dataclass
class WebSearchHit:
    """A single web search result."""

    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    """Interface any web search backend must implement."""

    def search(self, query: str, top_k: int) -> list[WebSearchHit]:
        ...


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor used for web_fetch results.

    Deliberately simple (stdlib-only, no BeautifulSoup dependency):
    drops <script>/<style> content and collapses everything else to
    whitespace-separated text. Good enough for feeding page content to
    an LLM, not intended as a general-purpose HTML renderer.
    """

    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.chunks.append(data.strip())


def html_to_text(html: str) -> str:
    """Strip an HTML document down to its visible text content."""
    extractor = _TextExtractor()
    extractor.feed(html)
    return "\n".join(extractor.chunks)


class DuckDuckGoSearchProvider:
    """Searches via DuckDuckGo's HTML-only endpoint (no API key needed).

    This is the default backend since it requires no signup or secret
    -- appropriate for a local assistant where the whole point is
    minimal external dependencies.
    """

    _ENDPOINT = "https://html.duckduckgo.com/html/"
    _RESULT_RE = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    _TAG_RE = re.compile(r"<[^>]+>")

    def search(self, query: str, top_k: int) -> list[WebSearchHit]:
        try:
            import requests
        except ImportError as exc:
            raise WebSearchBackendError(
                "The 'requests' package is required for web search."
            ) from exc

        try:
            response = requests.post(
                self._ENDPOINT,
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; local-assistant/1.0)"},
                timeout=CONFIG.web_request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WebSearchBackendError(f"DuckDuckGo search request failed: {exc}") from exc

        hits: list[WebSearchHit] = []
        for match in self._RESULT_RE.finditer(response.text):
            url, title_html, snippet_html = match.groups()
            title = self._TAG_RE.sub("", title_html).strip()
            snippet = self._TAG_RE.sub("", snippet_html).strip()
            if title and url:
                hits.append(WebSearchHit(title=title, url=url, snippet=snippet))
            if len(hits) >= top_k:
                break
        return hits


class BraveSearchProvider:
    """Searches via the Brave Search API. Requires BRAVE_API_KEY."""

    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self) -> None:
        import os

        self._api_key = os.environ.get("BRAVE_API_KEY")
        if not self._api_key:
            raise WebSearchBackendError(
                "BRAVE_API_KEY is not set. Export it or switch "
                "ASSISTANT_WEB_SEARCH_BACKEND to 'duckduckgo'."
            )

    def search(self, query: str, top_k: int) -> list[WebSearchHit]:
        try:
            import requests
        except ImportError as exc:
            raise WebSearchBackendError(
                "The 'requests' package is required for web search."
            ) from exc

        try:
            response = requests.get(
                self._ENDPOINT,
                params={"q": query, "count": top_k},
                headers={"Accept": "application/json", "X-Subscription-Token": self._api_key},
                timeout=CONFIG.web_request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WebSearchBackendError(f"Brave search request failed: {exc}") from exc

        try:
            data = response.json()
            results = data.get("web", {}).get("results", [])
        except (ValueError, AttributeError) as exc:
            raise WebSearchBackendError(f"Unexpected Brave search response shape: {exc}") from exc

        return [
            WebSearchHit(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
            )
            for r in results[:top_k]
        ]


def build_search_provider() -> SearchProvider | None:
    """Factory that builds the configured search backend.

    Returns None if web search is disabled (CONFIG.web_search_backend
    == "none"), which callers treat as "tool unavailable".
    """
    backend = CONFIG.web_search_backend
    if backend == "none":
        return None
    if backend == "brave":
        return BraveSearchProvider()
    if backend == "duckduckgo":
        return DuckDuckGoSearchProvider()
    raise WebSearchBackendError(f"Unknown web_search_backend: {backend!r}")


def fetch_url(url: str) -> str:
    """Fetch a URL and return its visible text content.

    Raises WebSearchBackendError on any network failure or non-HTML
    response the extractor can't make sense of.
    """
    try:
        import requests
    except ImportError as exc:
        raise WebSearchBackendError(
            "The 'requests' package is required for web_fetch."
        ) from exc

    if not url.lower().startswith(("http://", "https://")):
        raise WebSearchBackendError(f"Refusing to fetch a non-http(s) URL: {url!r}")

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; local-assistant/1.0)"},
            timeout=CONFIG.web_request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WebSearchBackendError(f"Could not fetch {url!r}: {exc}") from exc

    content_type = response.headers.get("Content-Type", "")
    if "html" in content_type:
        return html_to_text(response.text)
    return response.text
