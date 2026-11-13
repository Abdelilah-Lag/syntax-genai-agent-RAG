"""Bounded, process-local operational metrics without retaining user questions or answer text."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from statistics import mean
from threading import Lock
from typing import Any
from uuid import uuid4


class RuntimeMetrics:
    def __init__(self, history_limit: int = 500) -> None:
        self._records: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._lock = Lock()

    @staticmethod
    def _record(
        *,
        success: bool,
        latency_ms: int,
        timings_ms: dict[str, Any] | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
        critic: dict[str, Any] | None = None,
        cache_hit: bool = False,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        critic = critic or {}
        return {
            "id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "latency_ms": max(0, latency_ms),
            "timings_ms": {
                str(name): max(0, int(duration))
                for name, duration in (timings_ms or {}).items()
                if isinstance(duration, (int, float))
            },
            "tool_runs": [
                {
                    "name": str(item.get("tool", "unknown")),
                    "duration_ms": max(0, int(item.get("ms", 0) or 0)),
                    "passages": max(0, int(item.get("passages", 0) or 0)),
                    "success": not bool(item.get("error")),
                }
                for item in (tool_trace or [])
                if isinstance(item, dict)
            ],
            "critic": {
                "verdict": critic.get("verdict"),
                "review_status": critic.get("review_status"),
                "grounding_score": critic.get("grounding_score"),
                "attempts": max(0, int(critic.get("attempts", 0) or 0)),
                "llm_ok": critic.get("llm_ok"),
                "revisions": max(0, int(critic.get("revisions", 0) or 0)),
            },
            "cache_hit": cache_hit,
            "error_type": error_type,
        }

    def record_success(self, output: dict[str, Any], latency_ms: int) -> dict[str, Any]:
        record = self._record(
            success=True,
            latency_ms=latency_ms,
            timings_ms=output.get("timings_ms"),
            tool_trace=output.get("tool_trace"),
            critic=output.get("critic"),
            cache_hit=bool(output.get("cache_hit", False)),
        )
        with self._lock:
            self._records.append(record)
        return record

    def record_failure(self, latency_ms: int, error_type: str) -> dict[str, Any]:
        record = self._record(
            success=False,
            latency_ms=latency_ms,
            error_type=error_type,
        )
        with self._lock:
            self._records.append(record)
        return record

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)

        successful = [item for item in records if item["success"]]
        component_times: dict[str, list[int]] = defaultdict(list)
        tool_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        critic_verdicts: Counter[str] = Counter()
        critic_review_statuses: Counter[str] = Counter()
        critic_scores: list[float] = []
        critic_attempts: list[int] = []
        revision_counts: list[int] = []

        for item in records:
            for name, duration in item["timings_ms"].items():
                component_times[name].append(duration)
            for run in item["tool_runs"]:
                tool_runs[run["name"]].append(run)
            critic = item["critic"]
            if critic["verdict"]:
                critic_verdicts[str(critic["verdict"])] += 1
            if critic["review_status"]:
                critic_review_statuses[str(critic["review_status"])] += 1
            if isinstance(critic["grounding_score"], (int, float)):
                critic_scores.append(float(critic["grounding_score"]))
            if critic["llm_ok"] is not None:
                critic_attempts.append(critic["attempts"])
            revision_counts.append(critic["revisions"])

        tool_summary = {}
        for name, runs in sorted(tool_runs.items()):
            durations = [item["duration_ms"] for item in runs]
            successes = sum(item["success"] for item in runs)
            tool_summary[name] = {
                "calls": len(runs),
                "average_ms": round(mean(durations), 1),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "successes": successes,
                "failures": len(runs) - successes,
                "success_rate": successes / len(runs),
                "last_used_at": next(
                    (
                        record["timestamp"]
                        for record in reversed(records)
                        if any(run["name"] == name for run in record["tool_runs"])
                    ),
                    None,
                ),
            }

        return {
            "scope": "current process; bounded in-memory history; resets on restart",
            "history_limit": self._records.maxlen,
            "total_requests": len(records),
            "successful_requests": len(successful),
            "failed_requests": len(records) - len(successful),
            "average_latency_ms": round(mean(item["latency_ms"] for item in successful), 1) if successful else None,
            "average_tools_per_request": round(
                mean(len(item["tool_runs"]) for item in successful), 2
            ) if successful else None,
            "average_revisions": round(mean(revision_counts), 2) if revision_counts else None,
            "critic_verdicts": dict(critic_verdicts),
            "critic_review_statuses": dict(critic_review_statuses),
            "critic_review_available_rate": (
                sum(item["critic"]["llm_ok"] is True for item in successful)
                / sum(item["critic"]["llm_ok"] is not None for item in successful)
                if any(item["critic"]["llm_ok"] is not None for item in successful)
                else None
            ),
            "average_grounding_score": round(mean(critic_scores), 3) if critic_scores else None,
            "average_critic_attempts": round(mean(critic_attempts), 2) if critic_attempts else None,
            "average_component_ms": {
                name: round(mean(values), 1) for name, values in sorted(component_times.items())
            },
            "tools": tool_summary,
            "history": [
                {
                    **item,
                    "tool_runs": [
                        {key: run[key] for key in ("name", "duration_ms", "passages", "success")}
                        for run in item["tool_runs"]
                    ],
                }
                for item in records[-100:]
            ],
        }
