"""Hybrid in-memory retrieval: dense cosine similarity + BM25, fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from threading import Lock

import numpy as np

from app.ingestion.chunker import Chunk
from app.ingestion.indexer import VectorIndex
from app.retrieval.embeddings import Embedder

log = logging.getLogger(__name__)

_STOP = frozenset(
    "a an the of in on at to for with by and or is are was were be been being it its this that these those as from "
    "what which who whom how when where why do does did can could should would may might will shall not no than "
    "then there their them they he she his her i you we our your me my about into over under between each other "
    "any all some such only also if so".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOP]


@dataclass
class Passage:
    chunk: Chunk
    score: float
    dense: float | None = None


class Retriever:
    def __init__(self, index: VectorIndex, embedder: Embedder | None):
        self.index = index
        self._embedder = embedder
        self._search_cache: dict[tuple[object, ...], tuple[Passage, ...]] = {}
        self._search_cache_lock = Lock()
        self._build_bm25()

    # ---- BM25 -------------------------------------------------------------------------------
    def _build_bm25(self, k1: float = 1.4, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths = np.zeros(len(self.index.chunks), dtype=np.float32)
        for i, chunk in enumerate(self.index.chunks):
            tokens = tokenize(chunk.embed_text)
            self._lengths[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                self._postings[term].append((i, tf))
        self._avgdl = float(self._lengths.mean()) if len(self._lengths) else 1.0
        n = len(self.index.chunks)
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()}

    def _bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.index.chunks), dtype=np.float32)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                denom = tf + self._k1 * (1 - self._b + self._b * self._lengths[i] / self._avgdl)
                scores[i] += idf * tf * (self._k1 + 1) / denom
        return scores

    # ---- search -----------------------------------------------------------------------------
    def _mask(self, drugs: list[str] | None, source_types: list[str] | None) -> np.ndarray:
        drug_set = {d.lower() for d in drugs} if drugs else None
        types = set(source_types) if source_types else None
        return np.array(
            [
                (drug_set is None or c.drug in drug_set) and (types is None or c.source_type in types)
                for c in self.index.chunks
            ],
            dtype=bool,
        )

    def _dense(self, query: str) -> np.ndarray | None:
        if self.index.matrix is None or self._embedder is None:
            return None
        try:
            return self.index.matrix @ self._embedder.embed_query(query)
        except Exception as exc:  # embedding endpoint down: degrade to BM25
            log.warning("Query embedding failed (%s); using BM25 only", exc)
            return None

    @staticmethod
    def _source_bias(query: str, chunk: Chunk) -> float:
        q = query.lower()
        label_terms = (
            "contraindication",
            "warning",
            "side effect",
            "adverse",
            "interaction",
            "dose",
            "dosage",
            "indication",
            "pregnancy",
            "allergy",
            "monitor",
            "renal",
            "bleeding",
        )
        general_terms = ("what is", "definition", "overview", "symptom", "mechanism")
        if any(term in q for term in label_terms) and chunk.source_type == "dailymed":
            return 0.25
        if any(term in q for term in general_terms) and chunk.source_type == "medlineplus":
            return 0.15
        if chunk.source_type == "glossary":
            return 0.1 if any(term in q for term in ("define", "what is", "meaning", "acronym", "abbreviation")) else 0.0
        return 0.0

    def search(
        self,
        query: str,
        top_k: int = 5,
        drugs: list[str] | None = None,
        source_types: list[str] | None = None,
    ) -> list[Passage]:
        drugs = drugs or self.index.catalog.detect(query)
        key = (
            query,
            top_k,
            tuple(sorted(drugs)) if drugs else (),
            tuple(sorted(source_types)) if source_types else (),
        )
        with self._search_cache_lock:
            cached = self._search_cache.get(key)
        if cached is not None:
            return [Passage(p.chunk, p.score, p.dense) for p in cached]

        mask = self._mask(drugs, source_types)
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        lexical = self._bm25(query)
        dense = self._dense(query)

        fused: dict[int, float] = defaultdict(float)
        if dense is not None:
            for rank, i in enumerate(candidates[np.argsort(-dense[candidates])][: top_k * 6]):
                fused[int(i)] += 1.0 / (RRF_K + rank)
        positive = candidates[lexical[candidates] > 0]
        for rank, i in enumerate(positive[np.argsort(-lexical[positive])][: top_k * 6]):
            fused[int(i)] += 1.0 / (RRF_K + rank)

        ranked: list[tuple[int, float]] = []
        seen: set[str] = set()
        reranked = sorted(
            (
                (i, score + self._source_bias(query, self.index.chunks[i]))
                for i, score in fused.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for i, score in reranked:
            chunk = self.index.chunks[i]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            ranked.append((i, score))
            if len(ranked) >= top_k:
                break

        passages = [
            Passage(chunk=self.index.chunks[i], score=score, dense=float(dense[i]) if dense is not None else None)
            for i, score in ranked
        ]
        with self._search_cache_lock:
            if len(self._search_cache) >= 256:
                self._search_cache.pop(next(iter(self._search_cache)))
            self._search_cache[key] = tuple(passages)
        return [Passage(p.chunk, p.score, p.dense) for p in passages]
