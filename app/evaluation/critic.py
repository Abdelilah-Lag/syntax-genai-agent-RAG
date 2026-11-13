"""Critic: verifies grounding and flags unsupported / clinically ambiguous claims before the answer is emitted."""
from __future__ import annotations

import logging
import re

from app.agent.prompts import CRITIC_SYSTEM
from app.agent.state import ClaimCheck, CriticReport, EvidenceItem, EvidenceStore
from app.core.llm import ChatLLM

log = logging.getLogger(__name__)
_CITE_RE = re.compile(r"\[(E\d+(?:\s*[,;]\s*E\d+)*)\]")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_VALID_STATUS = {
    "supported", "inferred", "misclassified", "scope_mismatch", "generic_advice",
    "unsupported", "contradicted", "ambiguous",
}
_ACCEPTABLE_STATUS = {"supported", "inferred"}
_MAX_REVIEW_PASSAGES = 6
_MAX_REVIEW_CHARS = 9_000


def cited_ids(text: str) -> list[str]:
    return list(dict.fromkeys(
        eid
        for citation in _CITE_RE.findall(text)
        for eid in re.findall(r"E\d+", citation)
    ))


def _numbers(text: str) -> set[str]:
    text = _CITE_RE.sub(" ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s", " ", text)  # markdown list numbering
    return {n.replace(",", ".") for n in _NUMBER_RE.findall(text)}


def deterministic_checks(question: str, draft: str, evidence: EvidenceStore) -> tuple[list[str], list[str]]:
    valid = {i.eid for i in evidence.items}
    invalid = [c for c in cited_ids(draft) if c not in valid]
    allowed = _numbers(evidence.full_text()) | _numbers(question)
    ungrounded = sorted(n for n in _numbers(draft) if n not in allowed)
    return invalid, ungrounded


def _review_evidence(
    evidence: EvidenceStore, cited_ids_in_draft: list[str]
) -> list[tuple[EvidenceItem, str]]:
    by_id = {item.eid: item for item in evidence.items}
    cited = [by_id[eid] for eid in cited_ids_in_draft if eid in by_id]
    ordered = cited + [item for item in evidence.items if item.primary and item not in cited]
    ordered += [item for item in evidence.items if item not in ordered]
    selected: list[tuple[EvidenceItem, str]] = []
    used = 0
    for item in ordered:
        if len(selected) >= _MAX_REVIEW_PASSAGES:
            break
        remaining = _MAX_REVIEW_CHARS - used
        if remaining <= 0:
            break
        text = item.passage.chunk.text
        if len(text) > remaining:
            if item in cited:
                text = text[:remaining]
            else:
                continue
        selected.append((item, text))
        used += len(text)
    return selected


def _normalize_evidence_ids(value: object) -> list[str]:
    values = value if isinstance(value, list) else [value]
    normalized = []
    for item in values:
        match = re.fullmatch(r"\[?(E\d+)\]?", str(item).strip())
        normalized.append(match.group(1) if match else str(item).strip())
    return list(dict.fromkeys(eid for eid in normalized if eid))


def review(
    llm: ChatLLM,
    question: str,
    draft: str,
    evidence: EvidenceStore,
    model: str | None = None,
    timeout: float | None = None,
) -> CriticReport:
    invalid, ungrounded = deterministic_checks(question, draft, evidence)
    claims: list[ClaimCheck] = []
    missing: list[str] = []
    invalid_critic_citations: list[str] = []
    uncited_claims: list[str] = []
    summary, llm_ok = "", True
    draft_citations = cited_ids(draft)
    review_items = _review_evidence(evidence, draft_citations)
    review_ids = {item.eid for item, _ in review_items}
    evidence_text = "\n\n".join(
        f"[{item.eid}] file: {item.passage.chunk.doc_id} | section: {item.passage.chunk.section}\n"
        f"{text}"
        for item, text in review_items
    )
    try:
        data = llm.chat_json(
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": f"QUESTION:\n{question}\n\nDRAFT:\n{draft}\n\nEVIDENCE:\n{evidence_text}"},
            ],
            model=model,
            timeout=timeout,
        )
        for c in data.get("claims") or []:
            if not isinstance(c, dict) or not c.get("claim"):
                continue
            status = str(c.get("status") or "").lower()
            claim = str(c["claim"])
            claim_evidence = _normalize_evidence_ids(c.get("evidence") or [])
            unknown = [eid for eid in claim_evidence if eid not in review_ids]
            invalid_critic_citations.extend(eid for eid in unknown if eid not in invalid_critic_citations)
            note = str(c.get("note") or "")
            if unknown:
                status = "unsupported"
                note = (note + " " if note else "") + f"Critic referenced unavailable evidence: {', '.join(unknown)}."
            elif status in _ACCEPTABLE_STATUS and not claim_evidence:
                status = "unsupported"
                note = (note + " " if note else "") + "No supporting evidence ID was provided."
            elif status in _ACCEPTABLE_STATUS and not set(claim_evidence).intersection(draft_citations):
                uncited_claims.append(claim)
            claims.append(
                ClaimCheck(
                    claim=claim,
                    status=status if status in _VALID_STATUS else "unsupported",
                    evidence=claim_evidence,
                    note=note,
                )
            )
        missing = [str(m) for m in data.get("missing") or [] if str(m).strip()]
        summary = str(data.get("summary") or "")
    except Exception as exc:
        log.warning("Critic LLM failed: %s", exc)
        llm_ok = False

    acceptable = sum(c.status in _ACCEPTABLE_STATUS for c in claims)
    score = round(acceptable / len(claims), 3) if claims else None
    bad = any(
        c.status not in _ACCEPTABLE_STATUS
        for c in claims
    )
    deterministic_issues = bool(invalid or ungrounded)
    verdict = (
        "review_unavailable"
        if not llm_ok and not deterministic_issues
        else "revise"
        if bad or deterministic_issues or missing or invalid_critic_citations or uncited_claims or not claims
        else "pass"
    )
    return CriticReport(
        verdict,
        score,
        claims,
        missing,
        invalid,
        ungrounded,
        summary,
        llm_ok,
        invalid_critic_citations,
        uncited_claims,
        1,
    )
