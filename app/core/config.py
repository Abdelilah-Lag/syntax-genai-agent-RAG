"""Application settings, read from the environment (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
MAX_REVIEW_TIMEOUT_SECONDS = 20.0

DISCLAIMER = (
    "Disclaimer: this answer is generated automatically from a small set of public FDA drug labels "
    "(DailyMed) and consumer medication monographs for a technical exercise. It is NOT medical advice. "
    "Always consult a qualified healthcare professional before making any medication decision."
)


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str | None
    chat_model: str
    embedding_model: str
    corpus_dir: Path
    cache_dir: Path
    frontend_dir: Path
    top_k: int = 5
    max_revisions: int = 1
    max_plan_steps: int = 5
    evidence_char_budget: int = 24000
    llm_timeout: float = 30.0
    log_level: str = "INFO"
    fast_model: str | None = None
    critic_model: str | None = None
    review_timeout: float = 14.0


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Real environment variables win over .env (the graders inject their own values).
    load_dotenv(ROOT / ".env", override=False)
    return Settings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        fast_model=os.getenv("OPENAI_FAST_MODEL") or None,
        critic_model=os.getenv("OPENAI_CRITIC_MODEL") or None,
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        corpus_dir=Path(os.getenv("CORPUS_DIR", ROOT / "corpus")),
        cache_dir=Path(os.getenv("CACHE_DIR", ROOT / ".cache")),
        frontend_dir=ROOT / "frontend",
        top_k=_int("TOP_K", 5),
        max_revisions=_int("MAX_REVISIONS", 1),
        llm_timeout=_float("LLM_TIMEOUT_SECONDS", 30.0),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        review_timeout=min(
            max(_float("REVIEW_TIMEOUT_SECONDS", 14.0), 0.1),
            MAX_REVIEW_TIMEOUT_SECONDS,
        ),
    )
