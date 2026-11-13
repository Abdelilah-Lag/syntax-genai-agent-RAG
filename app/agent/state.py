"""Typed state shared by all graph nodes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retrieval.retriever import Passage


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    why: str = ""


@dataclass
class PlanStep:
    id: int
    sub_question: str
    search_query: str
    drugs: list[str]
    tools: list[ToolCall] = field(default_factory=list)  # optional tools chosen by the planner (search is implicit)


@dataclass
class Plan:
    intent: str
    steps: list[PlanStep]
    origin: str = "llm"  # "llm" or "fallback" (heuristic plan used when the LLM planner failed)


@dataclass
class ToolTrace:
    step: int
    tool: str
    args: dict[str, Any]
    n_passages: int
    ms: int
    error: str | None = None
    notes: str = ""


@dataclass
class EvidenceItem:
    eid: str
    passage: Passage
    tools: list[str]
    primary: bool


class EvidenceStore:
    """Deduplicated passages gathered by the tools; `freeze` assigns stable E1..En identifiers."""

    def __init__(self) -> None:
        self._raw: dict[str, dict[str, Any]] = {}
        self.items: list[EvidenceItem] = []

    def add(self, passage: Passage, tool: str, primary: bool) -> None:
        entry = self._raw.setdefault(
            passage.chunk.chunk_id, {"passage": passage, "tools": [], "primary": False, "score": 0.0, "order": len(self._raw)}
        )
        if tool not in entry["tools"]:
            entry["tools"].append(tool)
        entry["primary"] = entry["primary"] or primary
        entry["score"] = max(entry["score"], passage.score)

    def __len__(self) -> int:
        return len(self._raw)

    def freeze(self, char_budget: int) -> list[EvidenceItem]:
        """Keep primary evidence first, then the best search hits, within a character budget."""
        ranked = sorted(self._raw.values(), key=lambda e: (not e["primary"], e["order"] if e["primary"] else -e["score"]))
        kept, used = [], 0
        for entry in ranked:
            size = len(entry["passage"].chunk.text)
            if kept and used + size > char_budget:
                continue
            kept.append(entry)
            used += size
        kept.sort(key=lambda e: (e["passage"].chunk.doc_id, e["passage"].chunk.ordinal))
        self.items = [EvidenceItem(f"E{i}", e["passage"], e["tools"], e["primary"]) for i, e in enumerate(kept, 1)]
        return self.items

    def get(self, eid: str) -> EvidenceItem | None:
        return next((i for i in self.items if i.eid == eid), None)

    def render(self) -> str:
        return "\n\n".join(
            f"[{i.eid}] file: {i.passage.chunk.doc_id} | section: {i.passage.chunk.section}\n{i.passage.chunk.text}" for i in self.items
        )

    def full_text(self) -> str:
        return "\n".join(i.passage.chunk.text for i in self.items)


@dataclass
class ClaimCheck:
    claim: str
    status: str  # supported | unsupported | contradicted | ambiguous | misclassified | scope_mismatch
    evidence: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class CriticReport:
    verdict: str  # pass | revise | review_unavailable | not_required
    grounding_score: float | None
    claims: list[ClaimCheck] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)
    summary: str = ""
    llm_ok: bool | None = True
    invalid_critic_citations: list[str] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    review_attempts: int = 0

    @property
    def needs_revision(self) -> bool:
        return (
            self.verdict == "revise"
            or (self.grounding_score is not None and self.grounding_score < 0.9)
            or any(
                c.status in {
                    "unsupported", "contradicted", "generic_advice", "misclassified",
                    "scope_mismatch", "ambiguous",
                }
                for c in self.claims
            )
            or bool(
                self.missing or self.invalid_citations or self.ungrounded_numbers
                or self.invalid_critic_citations or self.uncited_claims
            )
        )

    @property
    def flags(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for c in self.claims:
            if c.status not in {"supported", "inferred"}:
                out.append({"type": c.status, "claim": c.claim, "note": c.note})
        out += [{"type": "missing", "claim": m, "note": "part of the question not answered"} for m in self.missing]
        out += [{"type": "invalid_citation", "claim": c, "note": "cites evidence that does not exist"} for c in self.invalid_citations]
        out += [
            {"type": "invalid_critic_citation", "claim": c, "note": "critic referenced evidence not provided for review"}
            for c in self.invalid_critic_citations
        ]
        for claim in self.uncited_claims:
            check = next((item for item in self.claims if item.claim == claim), None)
            evidence = f"; cite [{'], ['.join(check.evidence)}]" if check and check.evidence else ""
            out.append(
                {
                    "type": "uncited_claim",
                    "claim": claim,
                    "note": f"no supporting evidence ID is cited in the answer{evidence}",
                }
            )
        out += [{"type": "number_not_in_evidence", "claim": n, "note": "number absent from the evidence and the question"} for n in self.ungrounded_numbers]
        if self.llm_ok and not self.claims and self.verdict == "revise":
            out.append({"type": "review_incomplete", "claim": "", "note": "critic returned no verifiable claims"})
        if self.verdict == "review_unavailable":
            out.append({"type": "review_unavailable", "claim": "", "note": "automatic review could not run"})
        return out

    def feedback(self) -> str:
        return "\n".join(
            f"- [{f['type']}] {f['claim']} {('- ' + f['note']) if f['note'] else ''}".strip()
            for f in self.flags
        )


@dataclass
class AgentState:
    question: str
    plan: Plan | None = None
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    tool_notes: list[str] = field(default_factory=list)
    traces: list[ToolTrace] = field(default_factory=list)
    draft: str = ""
    critic: CriticReport | None = None
    critic_history: list[dict[str, Any]] = field(default_factory=list)
    revisions: int = 0
    critic_attempts: int = 0
    answer: dict[str, Any] | None = None
    timings: dict[str, int] = field(default_factory=dict)
