"""`search`: semantic + lexical retrieval. Mandatory for every plan step."""
from __future__ import annotations

from typing import Any

from app.retrieval.retriever import Passage
from app.tools.base import ToolContext, ToolResult

_LABEL_TERMS = (
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
    "precaution",
)


class SearchTool:
    name = "search"
    description = "Hybrid (embedding + BM25) search over the corpus; optionally restricted to given drugs."

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        drugs = [d for d in (args.get("drugs") or []) if d]
        if not drugs:
            drugs = ctx.catalog.detect(query)
        top_k = int(args.get("top_k") or ctx.settings.top_k)
        if not query:
            return ToolResult(self.name, args, [], "empty query", primary=False)

        if drugs:
            # Keep each drug represented without letting the same file appear multiple times.
            per_drug = max(2, min(top_k, max(3, top_k // max(1, len(drugs)))))
            by_doc: dict[str, Passage] = {}
            label_query = any(term in query.lower() for term in _LABEL_TERMS)
            for drug in drugs:
                hits = (
                    ctx.retriever.search(query, top_k=per_drug, drugs=[drug], source_types=["dailymed"])
                    if label_query
                    else []
                )
                if not hits:
                    hits = ctx.retriever.search(query, top_k=per_drug, drugs=[drug])
                for passage in hits:
                    key = passage.chunk.doc_id
                    if key not in by_doc or passage.score > by_doc[key].score:
                        by_doc[key] = passage
            passages = sorted(by_doc.values(), key=lambda p: p.score, reverse=True)[:top_k]
            present = {passage.chunk.doc_id for passage in passages}
            for drug in drugs:
                patient_hits = ctx.retriever.search(
                    query, top_k=1, drugs=[drug], source_types=["medlineplus"]
                )
                for passage in patient_hits:
                    if passage.chunk.doc_id not in present:
                        passages.append(passage)
                        present.add(passage.chunk.doc_id)
        else:
            passages = ctx.retriever.search(query, top_k=top_k + 2)
            seen: set[str] = set()
            deduped: list[Passage] = []
            for passage in passages:
                if passage.chunk.doc_id in seen:
                    continue
                seen.add(passage.chunk.doc_id)
                deduped.append(passage)
            passages = deduped[:top_k]
        return ToolResult(self.name, {"query": query, "drugs": drugs}, passages, primary=False)
