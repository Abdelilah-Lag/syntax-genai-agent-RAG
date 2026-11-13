from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.core.llm import LLMError
from app.schemas.response import API_CONTRACT_VERSION, AskRequest, AskResponse
from app.tools.registry import MANDATORY_TOOL, OPTIONAL_TOOLS

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", tags=["meta"])
def health(request: Request) -> dict:
    """Liveness probe: always 200 once the server is up (indexing may still be running)."""
    runtime = request.app.state.runtime
    return {
        "status": "ok",
        "api_contract_version": API_CONTRACT_VERSION,
        "index_ready": runtime.ready,
        "index_error": runtime.error,
        "chat_model": runtime.settings.chat_model,
        "fast_model": runtime.settings.fast_model,
        "critic_model": (
            runtime.settings.critic_model
            or runtime.settings.fast_model
            or runtime.settings.chat_model
        ),
    }


@router.get("/metrics", tags=["meta"])
def metrics(request: Request) -> dict:
    snapshot = request.app.state.runtime.metrics.snapshot()
    snapshot["registered_tools"] = sorted({MANDATORY_TOOL, *OPTIONAL_TOOLS})
    return snapshot


@router.post("/ask", response_model=AskResponse, tags=["agent"])
def ask(payload: AskRequest, request: Request) -> dict:
    runtime = request.app.state.runtime
    started = time.perf_counter()
    try:
        agent = runtime.get_agent()  # waits for the index if it is still being built
        bypass_cache = "no-cache" in request.headers.get("cache-control", "").lower()
        output = agent.ask(payload.question, use_cache=not bypass_cache)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        record = runtime.metrics.record_success(output, elapsed_ms)
        output["request_id"] = record["id"]
        output["timestamp"] = record["timestamp"]
        return output
    except LLMError as exc:
        runtime.metrics.record_failure(int((time.perf_counter() - started) * 1000), type(exc).__name__)
        log.error("LLM backend error: %s", exc)
        raise HTTPException(status_code=503, detail=f"LLM backend unavailable: {exc}") from exc
    except Exception as exc:
        runtime.metrics.record_failure(int((time.perf_counter() - started) * 1000), type(exc).__name__)
        log.exception("Unhandled error while answering")
        raise HTTPException(status_code=500, detail="Internal error while answering the question") from exc
