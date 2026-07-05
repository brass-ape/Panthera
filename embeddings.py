"""
embeddings.py
=============

Interface and implementations for semantic (embedding-based) search
over the vault, intended as a drop-in replacement for the keyword
search in memory.py's `KeywordRetriever`.

Nothing in the rest of the codebase currently *requires* this module
-- CONFIG.embedding_backend defaults to "none" and VaultMemory falls
back to keyword search. This module exists so that turning on
semantic search later is a one-line change (constructing an
EmbeddingRetriever and passing it into VaultMemory) rather than a
rewrite.

Two optional dependencies are used, both imported lazily so the rest
of the application works even if they are not installed:

* `requests`      -- to call Ollama's embeddings endpoint
* `faiss` / `chromadb` -- as pluggable vector store backends
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from config import CONFIG
from memory import SearchHit  # re-used dataclass, avoids duplicating the shape

logger = logging.getLogger("assistant.embeddings")

Vector = list[float]


class EmbeddingBackendError(Exception):
    """Raised when an embedding backend fails (offline, bad response, etc.)."""


class EmbeddingProvider(Protocol):
    """Turns text into a fixed-size vector."""

    def embed(self, text: str) -> Vector:
        ...


class OllamaEmbeddingProvider:
    """Generates embeddings via Ollama's local /api/embeddings endpoint.

    Designed to work with models such as `nomic-embed-text`, which
    must be pulled separately (`ollama pull nomic-embed-text`).
    """

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self._model = model or CONFIG.embedding_model
        self._host = (host or CONFIG.ollama_host).rstrip("/")

    def embed(self, text: str) -> Vector:
        try:
            import requests  # lazy import: optional dependency
        except ImportError as exc:
            raise EmbeddingBackendError(
                "The 'requests' package is required for Ollama embeddings."
            ) from exc

        url = f"{self._host}/api/embeddings"
        try:
            response = requests.post(
                url,
                json={"model": self._model, "prompt": text},
                timeout=CONFIG.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise EmbeddingBackendError(f"Ollama embeddings request failed: {exc}") from exc

        try:
            data = response.json()
            return list(data["embedding"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EmbeddingBackendError(f"Unexpected embeddings response shape: {exc}") from exc


class VectorStore(Protocol):
    """Minimal vector store interface: add, remove, and query by vector."""

    def upsert(self, doc_id: str, vector: Vector, text: str) -> None:
        ...

    def delete(self, doc_id: str) -> None:
        ...

    def query(self, vector: Vector, top_k: int) -> list[tuple[str, float, str]]:
        """Returns a list of (doc_id, similarity_score, stored_text)."""
        ...


class InMemoryVectorStore:
    """Trivial fallback vector store using cosine similarity in pure
    Python. Fine for a personal vault of a few hundred/thousand notes;
    not meant to scale to a large corpus.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, Vector] = {}
        self._texts: dict[str, str] = {}

    def upsert(self, doc_id: str, vector: Vector, text: str) -> None:
        self._vectors[doc_id] = vector
        self._texts[doc_id] = text

    def delete(self, doc_id: str) -> None:
        self._vectors.pop(doc_id, None)
        self._texts.pop(doc_id, None)

    def query(self, vector: Vector, top_k: int) -> list[tuple[str, float, str]]:
        scored = [
            (doc_id, _cosine_similarity(vector, vec), self._texts[doc_id])
            for doc_id, vec in self._vectors.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


class FaissVectorStore:
    """Optional FAISS-backed vector store.

    Requires `pip install faiss-cpu`. Falls back gracefully by raising
    EmbeddingBackendError if faiss is not installed, so callers can
    catch that and use InMemoryVectorStore instead.
    """

    def __init__(self, dimension: int) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise EmbeddingBackendError(
                "faiss is not installed. Run `pip install faiss-cpu`."
            ) from exc
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dimension)
        self._id_to_doc: list[str] = []
        self._texts: dict[str, str] = {}

    def upsert(self, doc_id: str, vector: Vector, text: str) -> None:
        import numpy as np  # faiss depends on numpy already

        vec = np.array([vector], dtype="float32")
        self._faiss.normalize_L2(vec)
        self._index.add(vec)
        self._id_to_doc.append(doc_id)
        self._texts[doc_id] = text

    def delete(self, doc_id: str) -> None:
        # IndexFlatIP does not support in-place deletion; a production
        # setup would rebuild the index periodically. For a personal
        # vault this is an acceptable, documented limitation.
        logger.warning("FaissVectorStore.delete is a no-op placeholder for %s.", doc_id)

    def query(self, vector: Vector, top_k: int) -> list[tuple[str, float, str]]:
        import numpy as np

        vec = np.array([vector], dtype="float32")
        self._faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, top_k)
        results: list[tuple[str, float, str]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_to_doc):
                continue
            doc_id = self._id_to_doc[idx]
            results.append((doc_id, float(score), self._texts[doc_id]))
        return results


class ChromaVectorStore:
    """Optional Chroma-backed vector store. Requires `pip install chromadb`."""

    def __init__(self, collection_name: str = "assistant_vault") -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise EmbeddingBackendError(
                "chromadb is not installed. Run `pip install chromadb`."
            ) from exc
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert(self, doc_id: str, vector: Vector, text: str) -> None:
        self._collection.upsert(ids=[doc_id], embeddings=[vector], documents=[text])

    def delete(self, doc_id: str) -> None:
        self._collection.delete(ids=[doc_id])

    def query(self, vector: Vector, top_k: int) -> list[tuple[str, float, str]]:
        result = self._collection.query(query_embeddings=[vector], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        documents = result.get("documents", [[]])[0]
        # Chroma returns distances (lower = closer); convert to a
        # similarity-like score so callers can sort descending.
        return [
            (doc_id, -dist, doc)
            for doc_id, dist, doc in zip(ids, distances, documents)
        ]


def _cosine_similarity(a: Vector, b: Vector) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_vector_store(dimension: int) -> VectorStore:
    """Factory that builds the configured vector store backend,
    falling back to the in-memory store if the requested backend's
    dependency is unavailable.
    """
    backend = CONFIG.vector_store_backend
    try:
        if backend == "faiss":
            return FaissVectorStore(dimension)
        if backend == "chroma":
            return ChromaVectorStore()
    except EmbeddingBackendError as exc:
        logger.warning("Falling back to in-memory vector store: %s", exc)
    return InMemoryVectorStore()


class EmbeddingRetriever:
    """A Retriever (see memory.py's Protocol) backed by embeddings.

    This is a drop-in replacement for KeywordRetriever: construct it
    and pass it to `VaultMemory(retriever=EmbeddingRetriever(...))`.
    It is not wired in by default because it requires an embedding
    model to be pulled in Ollama first.
    """

    def __init__(
        self,
        vault_dir: Path,
        provider: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        dimension: int = 768,
    ) -> None:
        self._vault_dir = vault_dir
        self._provider = provider or OllamaEmbeddingProvider()
        self._store = store or build_vector_store(dimension)
        self._bootstrap_index()

    def _bootstrap_index(self) -> None:
        """Index every existing markdown file on startup."""
        for path in sorted(self._vault_dir.rglob("*.md")):
            relative = str(path.relative_to(self._vault_dir)).replace("\\", "/")
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                self.index_file(relative, content)
            except (OSError, EmbeddingBackendError) as exc:
                logger.warning("Could not index %s: %s", relative, exc)

    def index_file(self, relative_path: str, content: str) -> None:
        try:
            vector = self._provider.embed(content)
        except EmbeddingBackendError as exc:
            logger.warning("Skipping embedding for %s: %s", relative_path, exc)
            return
        self._store.upsert(relative_path, vector, content)

    def remove_file(self, relative_path: str) -> None:
        self._store.delete(relative_path)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        try:
            query_vector = self._provider.embed(query)
        except EmbeddingBackendError as exc:
            logger.warning("Embedding search unavailable, returning no hits: %s", exc)
            return []
        results = self._store.query(query_vector, top_k)
        return [
            SearchHit(path=doc_id, score=score, snippet=text[:160].replace("\n", " "))
            for doc_id, score, text in results
        ]
