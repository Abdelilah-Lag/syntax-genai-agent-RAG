"""Batching + L2-normalisation around the embedding endpoint."""
from __future__ import annotations

import numpy as np

from app.core.llm import ChatLLM


class Embedder:
    def __init__(self, llm: ChatLLM, batch_size: int = 64):
        self._llm = llm
        self._batch = batch_size
        self._query_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = [t if t.strip() else "." for t in texts[i : i + self._batch]]
            vectors.extend(self._llm.embed(batch))
        return self._normalise(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._query_cache:
            if len(self._query_cache) > 512:
                self._query_cache.clear()
            self._query_cache[text] = self._normalise(np.asarray(self._llm.embed([text]), dtype=np.float32))[0]
        return self._query_cache[text]
