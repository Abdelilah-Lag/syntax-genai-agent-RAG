"""Pydantic models of the public API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

API_CONTRACT_VERSION = 4


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["What are the main contraindications for warfarin?"])

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 characters")
        return v


class Source(BaseModel):
    evidence_id: str
    file: str
    section: str
    source_type: str
    url: str | None = None
    snippet: str


class PlanStepOut(BaseModel):
    id: int
    sub_question: str
    drugs: list[str]
    tools: list[str]
    tool_reasons: dict[str, str] = {}


class PlanOut(BaseModel):
    intent: str
    origin: str
    steps: list[PlanStepOut]


class ToolTraceOut(BaseModel):
    step: int
    tool: str
    args: dict[str, Any]
    passages: int
    ms: int
    error: str | None = None


class CriticOut(BaseModel):
    verdict: str
    review_status: str = "completed"
    grounding_score: float | None
    revisions: int
    flags: list[dict[str, str]]
    attempts: int = 0
    llm_ok: bool | None = True
    claims: list[dict[str, Any]] = Field(default_factory=list)
    reviews: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class EvidenceOut(BaseModel):
    evidence_id: str
    file: str
    section: str
    source_type: str
    snippet: str
    score: float
    tools: list[str]
    primary: bool
    cited: bool


class AskResponse(BaseModel):
    request_id: str | None = None
    timestamp: str | None = None
    question: str
    answer: str
    answer_status: str
    cache_hit: bool = False
    sources: list[Source]
    evidence: list[EvidenceOut] = Field(default_factory=list)
    plan: PlanOut
    tools_used: list[str]
    tool_trace: list[ToolTraceOut]
    critic: CriticOut
    disclaimer: str
    latency_ms: int
    timings_ms: dict[str, int] = {}
