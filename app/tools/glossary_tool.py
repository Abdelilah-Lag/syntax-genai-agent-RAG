"""`glossary`: look a term up in GLOSSARY.md."""
from __future__ import annotations

import difflib
import re
from typing import Any

from app.ingestion.chunker import Chunk
from app.retrieval.retriever import Passage
from app.tools.base import ToolContext, ToolResult

_ENTRY_RE = re.compile(r"^- \*\*(?P<term>[^*]+)\*\*:\s*(?P<definition>.*)$")


def parse_glossary(text: str, doc_id: str) -> list[tuple[str, Chunk]]:
    entries: list[tuple[str, Chunk]] = []
    section, term, definition = "Glossary", None, []

    def flush():
        if term:
            body = f"{term}: {' '.join(definition).strip()}"
            chunk = Chunk(f"{doc_id}#{len(entries)}", doc_id, None, "glossary", section, section, len(entries), body)
            entries.append((term, chunk))

    for line in text.split("\n"):
        if line.startswith("## "):
            flush()
            term, definition, section = None, [], line[3:].strip()
        elif (m := _ENTRY_RE.match(line)):
            flush()
            term, definition = m["term"].strip(), [m["definition"].strip()]
        elif term and line.startswith("  ") and line.strip():
            definition.append(line.strip())
    flush()
    return entries


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class GlossaryTool:
    name = "glossary"
    description = "Define a pharmacology term from GLOSSARY.md."

    def __init__(self):
        self._entries: list[tuple[str, Chunk]] | None = None

    def _load(self, ctx: ToolContext):
        if self._entries is None:
            doc = next((d for d in ctx.index.documents if d.source_type == "glossary"), None)
            self._entries = parse_glossary(doc.text, doc.doc_id) if doc else []
        return self._entries

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        term = str(args.get("term") or "").strip()
        entries = self._load(ctx)
        if not term or not entries:
            return ToolResult(self.name, args, [], "no glossary available", primary=False)
        q = _norm(term)
        scored = []
        for name, chunk in entries:
            n = _norm(name)
            # the entry name or any of its parenthesised aliases ("Half-life (t½)", "INR (International ...)")
            variants = {n, *(_norm(v) for v in re.split(r"[()/]", name))} - {""}
            ratio = max(difflib.SequenceMatcher(None, q, v).ratio() for v in variants)
            if q in n or n in q or any(q == v for v in variants):
                ratio = max(ratio, 0.95)
            scored.append((ratio, chunk))
        scored.sort(key=lambda t: t[0], reverse=True)
        best = [c for r, c in scored[:2] if r >= 0.6]
        if not best:
            return ToolResult(self.name, {"term": term}, [], f"'{term}' is not defined in the glossary", primary=False)
        return ToolResult(self.name, {"term": term}, [Passage(chunk=c, score=1.0) for c in best], f"Glossary entry for '{term}'.")
