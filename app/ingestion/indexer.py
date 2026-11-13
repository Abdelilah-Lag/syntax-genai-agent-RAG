"""Build the in-memory vector index (embeddings cached on disk between restarts)."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

from app.core.config import Settings
from app.ingestion.chunker import Chunk, chunk_corpus
from app.ingestion.loader import DrugCatalog, Document, load_corpus
from app.retrieval.embeddings import Embedder

log = logging.getLogger(__name__)


@dataclass
class VectorIndex:
    documents: list[Document]
    chunks: list[Chunk]
    matrix: np.ndarray | None  # (n_chunks, dim) normalised, or None => lexical-only mode
    catalog: DrugCatalog
    by_doc: dict[str, list[Chunk]] = field(default_factory=dict)

    def __post_init__(self):
        for chunk in self.chunks:
            self.by_doc.setdefault(chunk.doc_id, []).append(chunk)

    def document(self, doc_id: str) -> Document | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)

    def chunks_for(self, drug: str, source_type: str | None = None) -> list[Chunk]:
        return [c for c in self.chunks if c.drug == drug and (source_type is None or c.source_type == source_type)]


def build_index(settings: Settings, embedder: Embedder | None) -> VectorIndex:
    docs = load_corpus(settings.corpus_dir)
    chunks = chunk_corpus(docs)
    catalog = DrugCatalog(d.drug for d in docs if d.drug)
    log.info("Corpus: %d documents, %d chunks, drugs=%s", len(docs), len(chunks), catalog.drugs)

    matrix = _embed_with_cache(settings, embedder, chunks) if embedder else None
    return VectorIndex(documents=docs, chunks=chunks, matrix=matrix, catalog=catalog)


def _embed_with_cache(settings: Settings, embedder: Embedder, chunks: list[Chunk]) -> np.ndarray | None:
    texts = [c.embed_text for c in chunks]
    key = hashlib.sha256((settings.embedding_model + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:20]
    cache_file = settings.cache_dir / f"emb-{key}.npy"
    try:
        if cache_file.exists():
            matrix = np.load(cache_file)
            if matrix.shape[0] == len(chunks):
                log.info("Loaded embeddings from cache %s", cache_file.name)
                return matrix
    except Exception:  # corrupted cache: just recompute
        log.warning("Ignoring unreadable embedding cache", exc_info=True)
    try:
        matrix = embedder.embed_documents(texts)
    except Exception as exc:
        log.error("Embedding failed (%s): falling back to lexical (BM25) retrieval only.", exc)
        return None
    try:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, matrix)
    except OSError:
        log.warning("Could not write embedding cache", exc_info=True)
    return matrix
