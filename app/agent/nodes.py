"""Graph nodes: plan -> execute (search + chosen tools) -> synthesize -> review -> (revise) -> finalize."""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from app.agent import prompts
from app.agent.planner import make_plan
from app.agent.state import AgentState, CriticReport, ToolTrace
from app.core.config import DISCLAIMER, MAX_REVIEW_TIMEOUT_SECONDS, Settings
from app.core.llm import ChatLLM
from app.evaluation.critic import review
from app.ingestion.indexer import VectorIndex
from app.retrieval.retriever import Retriever
from app.tools.base import Tool, ToolContext, ToolResult
from app.tools.registry import MANDATORY_TOOL

log = logging.getLogger(__name__)
_CITE_GROUP_RE = re.compile(r"\[(E\d+(?:\s*[,;]\s*E\d+)*)\]|\[(E\d+)\]")
_REFUSAL_RE = re.compile(
    r"(?is)^\s*(?:i (?:cannot|can't|could not|couldn't|do not|don't) "
    r"(?:answer|find|verify|provide|determine)|(?:the|this) information "
    r"(?:is|does) not (?:available|appear|exist)|no (?:relevant )?"
    r"(?:passages?|evidence|sources?) (?:were|was) found|je ne peux pas "
    r"(?:répondre|déterminer|vérifier)|je ne trouve pas|(?:the )?corpus "
    r"(?:does not|doesn't) (?:contain|include|provide|cover)|(?:this|that) "
    r"cannot be (?:determined|answered|verified))[^\n]*$"
)
_UNAVAILABLE_RE = re.compile(
    r"\b(?:does not|doesn't|do not|don't) (?:contain|include|provide|identify|establish|state|mention|list|give)\b"
    r"|\bnot (?:available|provided|found|included|in the corpus)\b"
    r"|\bcannot (?:determine|establish|answer)\b"
    r"|\babsent from (?:the )?(?:corpus|documents?|sources?)\b",
    re.I,
)
_MISSING_DOCUMENT_RE = re.compile(
    r"\b(?:the )?corpus (?:does not|doesn't) contain (?:a|an) ([a-z][a-z-]+) (?:drug )?(?:document|label)\b",
    re.I,
)
NO_EVIDENCE = (
    "I could not find passages in the provided corpus that answer this question, so I cannot answer it "
    "from the available sources."
)
OUT_OF_SCOPE_ANSWER = (
    "This request is outside the pharmacology topics covered by the provided corpus, "
    "so I cannot answer it from the available sources."
)
UNSUPPORTED_CORPUS_ANSWER = (
    "Current medicine prices and pharmacy price comparisons are not provided in the supplied corpus, "
    "so I cannot answer this from the available sources."
)


@dataclass
class AgentDeps:
    llm: ChatLLM
    index: VectorIndex
    retriever: Retriever
    tools: dict[str, Tool]
    settings: Settings

    @property
    def ctx(self) -> ToolContext:
        return ToolContext(self.index, self.retriever, self.llm, self.settings)


# ---- plan ----------------------------------------------------------------------------------------
def plan_node(state: AgentState, deps: AgentDeps) -> None:
    state.plan = make_plan(
        deps.llm,
        deps.index.catalog,
        state.question,
        deps.settings.max_plan_steps,
        model=deps.settings.fast_model or deps.settings.chat_model,
    )
    log.info("Plan (%s): %s", state.plan.origin, [(s.sub_question, [t.name for t in s.tools]) for s in state.plan.steps])


# ---- execute -------------------------------------------------------------------------------------
def _run_tool(deps: AgentDeps, step_id: int, name: str, args: dict[str, Any]) -> tuple[ToolResult | None, ToolTrace]:
    start = time.perf_counter()
    try:
        result = deps.tools[name].run(deps.ctx, args)
        trace = ToolTrace(step_id, name, result.args, len(result.passages), int((time.perf_counter() - start) * 1000), None, result.notes)
        return result, trace
    except Exception as exc:  # a failing tool must not break the whole answer
        log.exception("Tool %s failed", name)
        return None, ToolTrace(step_id, name, args, 0, int((time.perf_counter() - start) * 1000), str(exc))


def execute_node(state: AgentState, deps: AgentDeps) -> None:
    steps = state.plan.steps
    jobs = [
        (step.id, name, args)
        for step in steps
        for name, args in [
            (MANDATORY_TOOL, {"query": step.search_query, "drugs": step.drugs}),
            *((call.name, call.args) for call in step.tools),
        ]
    ]
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(jobs)))) as pool:
        results = list(pool.map(lambda job: _run_tool(deps, *job), jobs))
    per_step: dict[int, list[tuple[ToolResult | None, ToolTrace]]] = {step.id: [] for step in steps}
    for (step_id, _, _), result in zip(jobs, results):
        per_step[step_id].append(result)

    for step in steps:
        for result, trace in per_step[step.id]:
            state.traces.append(trace)
            if result is None:
                continue
            for passage in result.passages:
                state.evidence.add(passage, tool=result.tool, primary=result.primary)
            if result.tool != MANDATORY_TOOL and result.notes:
                state.tool_notes.append(f"[step {step.id}] {result.tool}: {result.notes}")


# ---- synthesize / revise -------------------------------------------------------------------------
def _write_answer(state: AgentState, deps: AgentDeps, feedback: str | None = None) -> str:
    plan_text = "\n".join(f"{s.id}. {s.sub_question}" for s in state.plan.steps)
    notes = "\n".join(state.tool_notes) or "(none)"
    user = (
        f"QUESTION:\n{state.question}\n\nPLAN:\n{plan_text}\n\nTOOL NOTES:\n{notes}\n\nEVIDENCE:\n{state.evidence.render()}\n"
    )
    if feedback:
        user += (
            f"\nPREVIOUS DRAFT:\n{state.draft}\n\nREVIEWER FEEDBACK (fix every point: remove or soften unsupported claims, "
            f"fix citations, answer missing parts, and only use numbers present in the evidence):\n{feedback}\n"
        )
    model = deps.settings.fast_model or deps.settings.chat_model
    if state.plan.intent in {"compare", "interaction", "multi_hop"}:
        model = deps.settings.chat_model
    return deps.llm.chat(
        [{"role": "system", "content": prompts.SYNTH_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        model=model,
    ).strip()


def synthesize_node(state: AgentState, deps: AgentDeps) -> None:
    state.evidence.freeze(deps.settings.evidence_char_budget)
    if state.plan and state.plan.intent == "out_of_scope":
        state.draft = OUT_OF_SCOPE_ANSWER
    elif state.plan and state.plan.intent == "unsupported_corpus":
        state.draft = UNSUPPORTED_CORPUS_ANSWER
    else:
        state.draft = _write_answer(state, deps) if state.evidence.items else NO_EVIDENCE


def revise_node(state: AgentState, deps: AgentDeps) -> None:
    state.revisions += 1
    state.draft = _write_answer(state, deps, feedback=state.critic.feedback() if state.critic else None)


# ---- review --------------------------------------------------------------------------------------
def review_node(state: AgentState, deps: AgentDeps) -> None:
    if not state.evidence.items or _is_refusal_only(state.draft, deps.index.catalog):
        state.critic = CriticReport(
            "not_required", None, llm_ok=None, summary="No answer claims require review."
        )
        return
    state.critic_attempts += 1
    started = time.perf_counter()
    state.critic = review(
        deps.llm,
        state.question,
        state.draft,
        state.evidence,
        model=deps.settings.critic_model or deps.settings.fast_model or deps.settings.chat_model,
        timeout=min(max(deps.settings.review_timeout, 0.1), MAX_REVIEW_TIMEOUT_SECONDS),
    )
    state.critic_history.append({
        "attempt": state.critic_attempts,
        "verdict": state.critic.verdict,
        "grounding_score": state.critic.grounding_score,
        "llm_ok": state.critic.llm_ok,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "flags": state.critic.flags,
        "claims": [
            {"claim": claim.claim, "status": claim.status, "evidence": claim.evidence, "note": claim.note}
            for claim in state.critic.claims
        ],
    })
    grounding = (
        f"{state.critic.grounding_score:.2f}"
        if state.critic.grounding_score is not None
        else "n/a"
    )
    log.info(
        "Critic: verdict=%s grounding=%s flags=%d",
        state.critic.verdict,
        grounding,
        len(state.critic.flags),
    )


# ---- finalize ------------------------------------------------------------------------------------
def _resolve_citations(text: str, state: AgentState) -> tuple[str, list[str]]:
    """Replace evidence IDs with document-and-section citations and collapse adjacent duplicates."""
    used: list[str] = []

    def repl(match: re.Match) -> str:
        ids = re.findall(r"E\d+", match.group(0))
        citations: list[str] = []
        for eid in ids:
            item = state.evidence.get(eid)
            if item:
                if eid not in used:
                    used.append(eid)
                chunk = item.passage.chunk
                citation = f"{chunk.doc_id} §{chunk.section}"
                if citation not in citations:
                    citations.append(citation)
        return "".join(f"[{citation}]" for citation in citations)

    resolved = _CITE_GROUP_RE.sub(repl, text)
    duplicate = re.compile(r"\[(?P<citation>[^\]]+)\](?:(?:\s*[,.;:]\s*|\s*)\[(?P=citation)\])+")
    while True:
        normalized = duplicate.sub(r"[\g<citation>]", resolved)
        if normalized == resolved:
            break
        resolved = normalized
    resolved = _dedupe_sentence_citations(resolved)
    return resolved, list(dict.fromkeys(used))


def _dedupe_sentence_citations(text: str) -> str:
    citation = re.compile(r"\[([^\]]+)\]")
    pieces = re.split(r"(?<=[.!?])(?=\s|$)", text)
    result: list[str] = []
    for piece in pieces:
        seen: set[str] = set()
        result.append(
            citation.sub(
                lambda match: "" if match.group(1) in seen else _remember_citation(match, seen),
                piece,
            )
        )
    return "".join(result)


def _remember_citation(match: re.Match, seen: set[str]) -> str:
    citation = match.group(1)
    seen.add(citation)
    return match.group(0)


def _is_refusal_only(text: str, catalog=None) -> bool:
    plain = re.sub(r"\[(?:E\d+|[^\]]+)\]", "", text).strip()
    missing_document = _MISSING_DOCUMENT_RE.search(plain)
    if missing_document and catalog and not catalog.canonical(missing_document.group(1)):
        return True

    plain = re.sub(r"(?i)^\s*(?:no[,.]?\s+|non[,.]?\s+)", "", plain)
    if not _REFUSAL_RE.match(plain) and not _UNAVAILABLE_RE.search(plain):
        return False
    sentences = [sentence.strip(" -*\t") for sentence in re.split(r"(?<=[.!?])\s+", plain) if sentence.strip()]
    substantive = [
        sentence for sentence in sentences
        if not _UNAVAILABLE_RE.search(sentence)
        and not re.match(r"(?i)^(?:therefore|thus|so|as a result)\b", sentence)
    ]
    return not substantive


def finalize_node(state: AgentState, deps: AgentDeps) -> None:
    critic = state.critic
    body, used = _resolve_citations(state.draft, state)
    refusal_only = _is_refusal_only(state.draft, deps.index.catalog)
    if refusal_only:
        body = re.sub(r"\s*\[[^\]]+\]", "", body).strip()
    items = [i for i in state.evidence.items if i.eid in used] if used and not refusal_only else []
    sources, seen = [], set()
    for item in items:
        c = item.passage.chunk
        if (c.doc_id, c.section) in seen:
            continue
        seen.add((c.doc_id, c.section))
        doc = deps.index.document(c.doc_id)
        sources.append(
            {
                "evidence_id": item.eid,
                "file": c.doc_id,
                "section": c.section,
                "source_type": c.source_type,
                "url": doc.url if doc else None,
                "snippet": (c.text[:280] + "...") if len(c.text) > 280 else c.text,
            }
        )

    parts = [body]
    flags = critic.flags if critic else []
    if flags:
        labels = {"unsupported": "Unsupported", "contradicted": "Contradicted by sources", "ambiguous": "Clinically ambiguous",
                  "misclassified": "Clinical category mismatch",
                  "scope_mismatch": "Evidence population or scope mismatch",
                  "generic_advice": "Unsupported general advice",
                  "missing": "Not answered", "invalid_citation": "Invalid citation", "number_not_in_evidence": "Number not found in sources",
                  "invalid_critic_citation": "Invalid reviewer evidence ID", "uncited_claim": "Claim without a citation",
                  "review_incomplete": "Review incomplete",
                  "review_unavailable": "Automatic review unavailable",
                  "review_incomplete": "Review incomplete"}
        notes = [f"- {labels.get(f['type'], f['type'])}: {f['claim']}".rstrip(": ") + (f" ({f['note']})" if f["note"] else "") for f in flags]
        parts.append("**Reviewer notes (please read before relying on this answer):**\n" + "\n".join(notes))
    files = list(dict.fromkeys(s["file"] for s in sources))
    if files:
        parts.append("**Sources:** " + ", ".join(files))
    parts.append(f"*{DISCLAIMER}*")

    plan = state.plan
    state.answer = {
        "question": state.question,
        "answer": "\n\n".join(parts),
        "answer_status": (
            "unsupported"
            if refusal_only or not used
            else "partial"
            if critic
            and (
                critic.llm_ok is False
                or critic.verdict == "revise"
                or critic.missing
                or any(claim.status == "generic_advice" for claim in critic.claims)
                or any(flag["type"] in {"unsupported", "contradicted", "ambiguous", "misclassified", "scope_mismatch"} for flag in flags)
            )
            else "supported"
        ),
        "sources": sources,
        "plan": {
            "intent": plan.intent,
            "origin": plan.origin,
            "steps": [
                {"id": s.id, "sub_question": s.sub_question, "drugs": s.drugs,
                 "tools": [MANDATORY_TOOL] + [t.name for t in s.tools],
                 "tool_reasons": {t.name: t.why for t in s.tools if t.why}}
                for s in plan.steps
            ],
        },
        "tools_used": list(dict.fromkeys(t.tool for t in state.traces)),
        "tool_trace": [
            {"step": t.step, "tool": t.tool, "args": t.args, "passages": t.n_passages, "ms": t.ms, "error": t.error} for t in state.traces
        ],
        "critic": {
            "verdict": critic.verdict if critic else "pass",
            "review_status": (
                "not_required"
                if not state.critic_attempts
                else "unavailable"
                if critic and critic.llm_ok is False
                else "completed"
            ),
            "grounding_score": critic.grounding_score if critic else 1.0,
            "revisions": state.revisions,
            "attempts": state.critic_attempts,
            "llm_ok": critic.llm_ok if critic else True,
            "flags": flags,
            "claims": [
                {"claim": claim.claim, "status": claim.status, "evidence": claim.evidence, "note": claim.note}
                for claim in critic.claims
            ] if critic else [],
            "reviews": state.critic_history,
            "summary": critic.summary if critic else "",
        },
        "evidence": [
            {
                "evidence_id": item.eid,
                "file": item.passage.chunk.doc_id,
                "section": item.passage.chunk.section,
                "source_type": item.passage.chunk.source_type,
                "snippet": item.passage.chunk.text,
                "score": item.passage.score,
                "tools": item.tools,
                "primary": item.primary,
                "cited": item.eid in used,
            }
            for item in state.evidence.items
        ],
        "disclaimer": DISCLAIMER,
    }
