"""Planning: decompose the question and let the LLM pick tools. Validated + heuristic fallback."""
from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.prompts import PLANNER_SYSTEM
from app.agent.state import Plan, PlanStep, ToolCall
from app.core.llm import ChatLLM
from app.ingestion.loader import DrugCatalog
from app.tools.extract_tool import SECTION_PATTERNS
from app.tools.registry import OPTIONAL_TOOLS

log = logging.getLogger(__name__)
MAX_TOOLS_PER_STEP = 3
_PHARMACOLOGY_TERMS_RE = re.compile(
    r"\b(?:drug|medication|medicine|pharmacology|pharmacokinetic|pharmacodynamic|"
    r"dose|dosage|warning|contraindication|interaction|side effect|adverse|"
    r"symptom|treatment|therapy|indication|mechanism|clinical|patient|medical|"
    r"pregnan\w*|renal|kidney|hepatic|liver|bleeding|allerg\w*|label|"
    r"serotonin|anticoagulant|antibiotic|pain|fever|inr|egfr)\b",
    re.I,
)
_OUT_OF_CORPUS_INFO_RE = re.compile(
    r"\b(?:price|cost|cheapest|affordability)\b"
    r"|\bpharmacies?\b.{0,60}\b(?:price|cost|cheapest)\b",
    re.I,
)


def scope_guard_reason(question: str, catalog: DrugCatalog) -> str | None:
    if _OUT_OF_CORPUS_INFO_RE.search(question):
        return "unsupported_corpus"
    if catalog.detect(question) or _PHARMACOLOGY_TERMS_RE.search(question):
        return None
    return "out_of_scope"


def make_plan(
    llm: ChatLLM,
    catalog: DrugCatalog,
    question: str,
    max_steps: int = 5,
    model: str | None = None,
) -> Plan:
    scope_reason = scope_guard_reason(question, catalog)
    if scope_reason:
        return Plan(intent=scope_reason, steps=[], origin="scope_guard")

    system = PLANNER_SYSTEM.format(max_steps=max_steps, drugs=", ".join(catalog.drugs))
    try:
        raw = llm.chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": question}],
            model=model,
        )
        plan = validate_plan(raw, catalog, question, max_steps)
        if plan.steps:
            return plan
        log.warning("Planner returned no usable step; using fallback plan")
    except Exception as exc:  # LLM down, invalid JSON...: degrade gracefully
        log.warning("Planner failed (%s); using fallback plan", exc)
    return fallback_plan(catalog, question)


def _drug_list(values: Any, catalog: DrugCatalog) -> list[str]:
    if isinstance(values, str):
        values = [values]
    out = [catalog.canonical(v) for v in values or [] if isinstance(v, str)]
    return list(dict.fromkeys(d for d in out if d))


def _clean_call(name: str, args: dict, why: str, step_drugs: list[str], sub_question: str, catalog: DrugCatalog) -> ToolCall | None:
    """Enforce each tool's usage rules in code, whatever the LLM proposed."""
    if name == "extract":
        drug = catalog.canonical(args.get("drug")) or (step_drugs[0] if len(step_drugs) == 1 else None)
        topic = str(args.get("topic") or "").strip()
        if not (drug and topic):
            return None
        source_types = [
            source_type
            for source_type, label in (("dailymed", "dailymed"), ("medlineplus", "medlineplus"))
            if re.search(rf"\b{label}\b", sub_question, re.I)
        ]
        call_args: dict[str, Any] = {"drug": drug, "topic": topic}
        if len(source_types) > 1:
            call_args["source_types"] = source_types
        return ToolCall("extract", call_args, why)
    if name == "compare":
        drugs = _drug_list(args.get("drugs"), catalog) or step_drugs
        if len(drugs) < 2:
            return None
        return ToolCall("compare", {"drugs": drugs, "aspect": str(args.get("aspect") or "contraindications")}, why)
    if name == "interaction":
        a, b = catalog.canonical(args.get("drug_a")), catalog.canonical(args.get("drug_b"))
        if not (a and b) and len(step_drugs) >= 2:
            a, b = step_drugs[:2]
        return ToolCall("interaction", {"drug_a": a, "drug_b": b}, why) if a and b and a != b else None
    if name == "glossary":
        term = str(args.get("term") or "").strip()
        return ToolCall("glossary", {"term": term}, why) if term else None
    if name == "verify":
        claim = str(args.get("claim") or sub_question).strip()
        return ToolCall("verify", {"claim": claim, "drugs": step_drugs}, why) if claim else None
    return None


def validate_plan(raw: dict, catalog: DrugCatalog, question: str, max_steps: int = 5) -> Plan:
    steps: list[PlanStep] = []
    for item in (raw.get("steps") or [])[:max_steps]:
        if not isinstance(item, dict):
            continue
        sub_q = str(item.get("sub_question") or "").strip()
        if not sub_q:
            continue
        drugs = _drug_list(item.get("drugs"), catalog) or catalog.detect(sub_q)
        calls: list[ToolCall] = []
        for t in item.get("tools") or []:
            if not isinstance(t, dict):
                continue
            name = str(t.get("name") or "").lower()
            if name not in OPTIONAL_TOOLS:  # `search` is implicit; unknown names are dropped
                continue
            args = t.get("args") if isinstance(t.get("args"), dict) else {}
            call = _clean_call(name, args, str(t.get("why") or ""), drugs, sub_q, catalog)
            if call and all((call.name, call.args) != (c.name, c.args) for c in calls):
                calls.append(call)
        steps.append(
            PlanStep(
                id=len(steps) + 1,
                sub_question=sub_q,
                search_query=str(item.get("search_query") or sub_q).strip(),
                drugs=drugs,
                tools=calls[:MAX_TOOLS_PER_STEP],
            )
        )
    return Plan(intent=str(raw.get("intent") or "lookup"), steps=steps, origin="llm")


def fallback_plan(catalog: DrugCatalog, question: str) -> Plan:
    """Keyword heuristics - only used when the LLM planner is unavailable."""
    q = question.lower()
    drugs = catalog.detect(question)
    calls: list[ToolCall] = []
    intent = "lookup"
    if len(drugs) >= 2 and re.search(r"compar|versus|\bvs\.?\b|differ|shared|unique|similar", q):
        calls.append(ToolCall("compare", {"drugs": drugs, "aspect": _topic(q) or "contraindications"}, "heuristic: comparison wording"))
        intent = "compare"
    elif len(drugs) >= 2 and re.search(r"interact|prescribed|together|combin|concomitant|co-?administ|taking|\bon\b", q):
        calls.append(ToolCall("interaction", {"drug_a": drugs[0], "drug_b": drugs[1]}, "heuristic: two drugs + interaction wording"))
        intent = "interaction"
    elif not drugs and re.match(r"\s*(what is|what does|what's|define|meaning of|explain)\b", q):
        term = re.sub(r"^(what is|what does|what's|define|meaning of|explain)\s+(an?\s+|the\s+)?|\s+mean\??$|\?$", "", question.strip(), flags=re.I)
        calls.append(ToolCall("glossary", {"term": term.strip()}, "heuristic: definition wording"))
        intent = "definition"
    elif drugs and _topic(q):
        calls.append(ToolCall("extract", {"drug": drugs[0], "topic": _topic(q)}, "heuristic: drug + topic keyword"))
    step = PlanStep(1, question.strip(), question.strip(), drugs, calls)
    return Plan(intent=intent, steps=[step], origin="fallback")


def _topic(q: str) -> str:
    for key in ("contraindication", "interaction", "dosage", "dose", "adverse", "side effect", "warning", "mechanism", "renal", "egfr", "indication", "pregnan"):
        if key in q:
            return "renal impairment contraindications" if key == "egfr" else key
    return ""
