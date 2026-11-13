"""`extract`: pull the full label section(s) of one drug about a topic (precise, section-aware)."""
from __future__ import annotations

import re
from threading import Lock
from typing import Any

from app.ingestion.chunker import Chunk
from app.retrieval.retriever import Passage
from app.tools.base import ToolContext, ToolResult

# topic keyword -> regex applied to the chunk's section path
SECTION_PATTERNS: dict[str, str] = {
    "contraindication": r"contraindicat",
    "interaction": r"interaction",
    "dosage": r"dosage|dosing|how should this medicine|\bdose",
    "dose": r"dosage|dosing|how should this medicine|\bdose",
    "administration": r"dosage|administration|how should this medicine be used",
    "indication": r"\bindications?\b|why is this medication prescribed|^uses?$",
    "warning": r"warning|precaution",
    "precaution": r"warning|precaution",
    "patient": r"information for patients|patient counseling|special precautions|precaution|warning",
    "bleeding": r"bleed|serious|call|warning",
    "help": r"serious|call|emergency|warning",
    "adverse": r"adverse|side effect",
    "side effect": r"adverse|side effect",
    "mechanism": r"mechanism|pharmacodynamic|clinical pharmacology",
    "pharmacokinetic": r"pharmacokinetic|clinical pharmacology",
    "renal": r"renal|kidney",
    "kidney": r"renal|kidney",
    "hepatic": r"hepatic|liver",
    "pregnan": r"pregnan|lactation",
    "lactation": r"pregnan|lactation",
    "overdos": r"overdos",
    "monitor": r"monitor",
    "pediatric": r"pediatric",
    "geriatric": r"geriatric",
    "storage": r"storage|store",
    "missed": r"missed|forget",
}
MAX_CHARS = 8000


def sections_for_topic(topic: str) -> re.Pattern[str]:
    topic_l = topic.lower()
    # keys must start at a word boundary, so "indication" does not match inside "contraindications"
    parts = [rx for key, rx in SECTION_PATTERNS.items() if re.search(rf"\b{re.escape(key)}", topic_l)]
    if not parts:  # unknown topic: use its own words as the section filter
        words = [w for w in re.findall(r"[a-z]{4,}", topic_l)]
        parts = [re.escape(w) for w in words] or [r"."]
    return re.compile("|".join(dict.fromkeys(parts)), re.I)


def collect_sections(
    ctx: ToolContext, drug: str, topic: str, source_type: str | None = None
) -> list[Chunk]:
    """Chunks of `drug` whose section path matches the topic (DailyMed first, then MedlinePlus)."""
    rx = sections_for_topic(topic)
    source_types = (source_type,) if source_type else ("dailymed", "medlineplus")
    for selected_source in source_types:
        hits = [c for c in ctx.index.chunks_for(drug, selected_source) if rx.search(c.section)]
        if hits:
            return hits
    return []


def budgeted(chunks: list[Chunk], limit: int = MAX_CHARS) -> list[Chunk]:
    out, used = [], 0
    for c in chunks:
        if out and used + len(c.text) > limit:
            continue
        if not out and len(c.text) > limit:
            continue
        out.append(c)
        used += len(c.text)
    return out


class ExtractTool:
    name = "extract"
    description = "Read the label section(s) of one drug about a topic (contraindications, interactions, dosage...)."

    def __init__(self) -> None:
        self._section_cache: dict[tuple[str, str, tuple[str, ...]], tuple[Chunk, ...]] = {}
        self._cache_lock = Lock()

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        drug = ctx.catalog.canonical(args.get("drug"))
        topic = str(args.get("topic") or "").strip()
        requested_sources = args.get("source_types")
        source_types = tuple(
            source for source in ("dailymed", "medlineplus")
            if isinstance(requested_sources, list) and source in requested_sources
        )
        if not drug:
            return ToolResult(self.name, args, [], f"unknown drug: {args.get('drug')!r}", primary=False)
        key = (drug, topic.casefold(), source_types)
        with self._cache_lock:
            chunks = self._section_cache.get(key)
        if chunks is None:
            if topic and source_types:
                selected = [
                    chunk
                    for source_type in source_types
                    for chunk in budgeted(
                        collect_sections(ctx, drug, topic, source_type),
                        limit=MAX_CHARS // len(source_types),
                    )
                ]
                if not selected:
                    selected = [
                        passage.chunk
                        for source_type in source_types
                        for passage in ctx.retriever.search(
                            f"{drug} {topic}",
                            top_k=3,
                            drugs=[drug],
                            source_types=[source_type],
                        )
                    ]
                chunks = tuple(budgeted(selected))
            else:
                chunks = tuple(budgeted(collect_sections(ctx, drug, topic))) if topic else ()
            with self._cache_lock:
                if len(self._section_cache) >= 128:
                    self._section_cache.pop(next(iter(self._section_cache)))
                self._section_cache[key] = chunks
        if chunks:
            passages = [Passage(chunk=c, score=1.0) for c in chunks]
            note = f"Extracted {len(chunks)} section chunk(s) of {drug} matching '{topic}'."
        else:  # no matching heading: fall back to retrieval restricted to this drug
            passages = ctx.retriever.search(f"{drug} {topic}", top_k=4, drugs=[drug])
            note = f"No section heading matched '{topic}' for {drug}; used semantic search restricted to {drug}."
        return ToolResult(self.name, {"drug": drug, "topic": topic}, passages, note)
