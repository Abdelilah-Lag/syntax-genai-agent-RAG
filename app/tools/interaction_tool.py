"""`interaction`: find where each label mentions the other drug (or its class) in interaction/warning text."""
from __future__ import annotations

import re
from typing import Any

from app.retrieval.retriever import Passage
from app.tools.base import ToolContext, ToolResult

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_SECTION_BONUS = re.compile(r"interaction|bleeding|warning|precaution|monitor|counsel|important warning", re.I)


def _term_regex(terms: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b", re.I)


class InteractionTool:
    name = "interaction"
    description = "Locate drug-drug interaction statements between two drugs in both labels."

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        a = ctx.catalog.canonical(args.get("drug_a"))
        b = ctx.catalog.canonical(args.get("drug_b"))
        if not a or not b or a == b:
            return ToolResult(self.name, args, [], "interaction needs two different known drugs", primary=False)

        scored: list[tuple[float, Passage]] = []
        highlights: list[str] = []
        for source, other in ((a, b), (b, a)):
            named = _term_regex(ctx.catalog.search_terms(other))
            by_class = _term_regex(ctx.catalog.class_terms(other)) if ctx.catalog.class_terms(other) else None
            for chunk in ctx.index.chunks_for(source):
                direct = named.search(chunk.text)
                klass = by_class.search(chunk.text) if by_class else None
                if not (direct or klass):
                    continue
                score = (2.0 if direct else 1.0) + (1.0 if _SECTION_BONUS.search(chunk.section) else 0.0)
                score += 0.5 if chunk.source_type == "dailymed" else 0.0
                scored.append((score, Passage(chunk=chunk, score=score)))
                if direct:
                    sentence = next((s for s in _SENT_RE.split(chunk.text) if named.search(s)), "")
                    if sentence:
                        highlights.append(f"{chunk.doc_id} [{chunk.section}]: {sentence.strip()[:300]}")

        scored.sort(key=lambda t: t[0], reverse=True)
        passages = [p for _, p in scored[:8]]
        if passages:
            notes = f"Passages where the {a} / {b} labels mention each other or the other's drug class."
            if highlights:
                notes += "\nKey sentences:\n- " + "\n- ".join(dict.fromkeys(highlights[:5]))
        else:
            notes = f"Neither label explicitly mentions {a} with {b}: absence of a statement is NOT evidence of safety."
            passages = ctx.retriever.search(f"{a} {b} drug interaction", top_k=4, drugs=[a, b])
        return ToolResult(self.name, {"drug_a": a, "drug_b": b}, passages, notes)
