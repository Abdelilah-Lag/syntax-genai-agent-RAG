"""`compare`: side-by-side comparison of several drugs on one aspect (shared vs unique)."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.ingestion.chunker import Chunk
from app.retrieval.retriever import Passage
from app.tools.base import ToolContext, ToolResult
from app.tools.extract_tool import budgeted, collect_sections

_ASPECT_STOPWORDS = {"and", "the", "for", "from", "with", "when", "what", "how", "patient", "facing"}
_ASPECT_SYNONYMS = {"missed": {"forget", "forgot"}, "bleeding": {"bleed"}}


def _rank_for_aspect(chunks: list[Chunk], aspect: str) -> list[Chunk]:
    terms = {
        term for term in re.findall(r"[a-z0-9]+", aspect.casefold())
        if len(term) > 2 and term not in _ASPECT_STOPWORDS
    }

    def relevance(chunk: Chunk) -> int:
        section = set(re.findall(r"[a-z0-9]+", chunk.section.casefold()))
        content = set(re.findall(r"[a-z0-9]+", chunk.text.casefold()))
        score = 0
        for term in terms:
            variants = {term, *_ASPECT_SYNONYMS.get(term, set())}
            if any(variant in section for variant in variants):
                score += 10
            elif any(variant in content for variant in variants):
                score += 1
        return score

    return sorted(chunks, key=relevance, reverse=True)


class CompareTool:
    name = "compare"
    description = "Retrieve the same aspect from 2+ drug labels for grounded comparison during synthesis."

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        drugs = list(dict.fromkeys(d for d in (ctx.catalog.canonical(x) for x in args.get("drugs") or []) if d))
        aspect = str(args.get("aspect") or "contraindications").strip()
        if len(drugs) < 2:
            return ToolResult(self.name, args, [], "compare needs at least two known drugs", primary=False)

        def extract_for_drug(drug: str) -> tuple[str, list[Passage], list[Chunk]]:
            chunks_by_source = []
            for source_type in ("dailymed", "medlineplus"):
                sections = _rank_for_aspect(collect_sections(ctx, drug, aspect, source_type), aspect)
                chunks = budgeted(sections, limit=3000)
                if not chunks:
                    found = ctx.retriever.search(
                        f"{drug} {aspect}",
                        top_k=3,
                        drugs=[drug],
                        source_types=[source_type],
                    )
                    chunks_by_source.extend(p.chunk for p in found)
                else:
                    chunks_by_source.extend(chunks)
            passages = [Passage(chunk=chunk, score=1.0) for chunk in chunks_by_source]
            return drug, passages, chunks_by_source

        with ThreadPoolExecutor(max_workers=min(4, len(drugs))) as pool:
            extracted = list(pool.map(extract_for_drug, drugs))

        passages: list[Passage] = []
        for _, found, _ in extracted:
            passages.extend(found)

        notes = (
            f"Compare the labelled {aspect} passages for {', '.join(drugs)} directly. "
            "Report shared items only when each label states them; otherwise keep findings drug-specific."
        )
        return ToolResult(self.name, {"drugs": drugs, "aspect": aspect}, passages, notes)
