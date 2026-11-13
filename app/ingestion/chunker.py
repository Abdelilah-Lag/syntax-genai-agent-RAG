"""Section-aware chunking of markdown documents.

Every chunk keeps its heading path (e.g. "4 CONTRAINDICATIONS" or
"5 WARNINGS AND PRECAUTIONS > 5.3 Increased Risk of Bleeding") so that retrieval and citations
can point at a precise part of a label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from app.ingestion.loader import Document

MAX_CHARS = 1100
OVERLAP_CHARS = 160
MIN_CHARS = 150
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    drug: str | None
    source_type: str
    section: str  # full heading path
    section_root: str  # top-level (##) heading
    ordinal: int  # position inside the document
    text: str

    @property
    def embed_text(self) -> str:
        """Text that is embedded / BM25-indexed: the content plus its context header."""
        label = self.drug or "general"
        return f"{label} | {self.source_type} | {self.section}\n{self.text}"


def iter_sections(text: str, fallback_title: str) -> Iterator[tuple[list[str], str]]:
    """Yield (heading_path, body) for each heading that directly owns some body text."""
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def path() -> list[str]:
        titles = [t for level, t in stack if level >= 2]
        return titles or [fallback_title]

    def flush() -> tuple[list[str], str] | None:
        body = "\n".join(line for line in buf if not line.startswith("Source:")).strip()
        return (path(), body) if body else None

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            out = flush()
            if out:
                yield out
            buf = []
            level, title = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    out = flush()
    if out:
        yield out


def _hard_split(sentence: str) -> list[str]:
    words, parts, cur = sentence.split(), [], ""
    for word in words:
        if cur and len(cur) + 1 + len(word) > MAX_CHARS:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        parts.append(cur)
    return parts


def _split_long_line(line: str) -> list[str]:
    """Split an over-long paragraph into windows of whole sentences with a one-sentence overlap."""
    sentences: list[str] = []
    for s in _SENTENCE_RE.split(line):
        sentences.extend(_hard_split(s) if len(s) > MAX_CHARS else [s])
    windows: list[str] = []
    cur: list[str] = []
    size = 0
    for s in sentences:
        if cur and size + len(s) + 1 > MAX_CHARS:
            windows.append(" ".join(cur))
            keep = cur[-1] if len(cur[-1]) <= OVERLAP_CHARS * 2 else None
            cur, size = ([keep] if keep else []), (len(keep) if keep else 0)
        cur.append(s)
        size += len(s) + 1
    if cur:
        windows.append(" ".join(cur))
    return windows


def split_body(body: str) -> list[str]:
    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= MAX_CHARS:
            units.append(para)
            continue
        for line in para.split("\n"):  # keep bullet items intact when possible
            line = line.strip()
            if line:
                units.extend([line] if len(line) <= MAX_CHARS else _split_long_line(line))
    chunks: list[str] = []
    cur = ""
    for unit in units:
        if cur and len(cur) + 1 + len(unit) > MAX_CHARS:
            chunks.append(cur)
            cur = unit
        else:
            cur = f"{cur}\n{unit}" if cur else unit
    if cur:
        chunks.append(cur)
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHARS:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]}\n{tail}"
    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading_path, body in iter_sections(doc.text, fallback_title=doc.title):
        for piece in split_body(body):
            n = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{n}",
                    doc_id=doc.doc_id,
                    drug=doc.drug,
                    source_type=doc.source_type,
                    section=" > ".join(heading_path),
                    section_root=heading_path[0],
                    ordinal=n,
                    text=piece,
                )
            )
    return chunks


def chunk_corpus(docs: list[Document]) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc)]
