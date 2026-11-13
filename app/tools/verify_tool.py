"""`verify`: check a claim made by the user against the corpus."""
from __future__ import annotations

from typing import Any

from app.tools.base import ToolContext, ToolResult


class VerifyTool:
    name = "verify"
    description = "Retrieve evidence for a user-stated claim; the final synthesis evaluates it against those passages."

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        claim = str(args.get("claim") or "").strip()
        if not claim:
            return ToolResult(self.name, args, [], "empty claim", primary=False)
        drugs = [d for d in (ctx.catalog.canonical(x) for x in args.get("drugs") or []) if d] or ctx.catalog.detect(claim)
        passages = ctx.retriever.search(claim, top_k=6, drugs=drugs or None)
        if not passages:
            return ToolResult(self.name, {"claim": claim}, [], "no passage retrieved for this claim", primary=False)
        notes = (
            "Check the user's claim against these retrieved passages. State whether it is supported, contradicted, "
            "or not addressed; do not infer beyond the text."
        )
        return ToolResult(self.name, {"claim": claim}, passages, notes)
