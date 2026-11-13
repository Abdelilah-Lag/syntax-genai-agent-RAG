"""Evaluate the Syntax Agentic RAG against the public corpus test set.

Run locally:
    python -m app.run_eval
    python -m app.run_eval --url http://localhost:8000

The dataset is read from app/public_eval_set.json. Reports are written to app/results/.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

from app.evaluation.metrics import has_abstention_justification, required_term_present
from app.schemas.response import API_CONTRACT_VERSION

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "public_eval_set.json"
RESULTS_DIR = BASE_DIR / "results"
LATENCY_BUDGET_MS = 30_000
LATENCY_PROFILES = {
    "simple": {"target_ms": 8_000, "max_ms": 15_000},
    "multi_document": {"target_ms": 15_000, "max_ms": 25_000},
    "complex": {"target_ms": 25_000, "max_ms": 40_000},
    "multi_hop": {"target_ms": 30_000, "max_ms": 45_000},
}


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile_rank / 100 * len(ordered)) - 1))
    return ordered[index]


def source_recall(expected: list[str], actual: list[str]) -> float:
    expected_set, actual_set = set(expected), set(actual)
    return len(expected_set & actual_set) / len(expected_set) if expected_set else 0.0


def source_precision(expected: list[str], actual: list[str]) -> float:
    expected_set, actual_set = set(expected), set(actual)
    if not actual_set:
        return 0.0
    return len(expected_set & actual_set) / len(actual_set)


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def load_dataset() -> list[dict]:
    with DATASET_FILE.open(encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, list):
        raise ValueError("Evaluation dataset must be a JSON list.")
    if len(dataset) != 40:
        print(f"WARNING: expected 40 questions, found {len(dataset)}.")
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Question at index {index} must be a JSON object.")
        for field in ("id", "question", "expected_sources"):
            if field not in item:
                raise ValueError(f"Question at index {index} is missing '{field}'.")
        item.setdefault("source_policy", "required" if item["expected_sources"] else "none")
        item.setdefault(
            "expected_status",
            "unsupported" if item["source_policy"] == "none" and item.get("category") == "abstention" else "supported",
        )
        if item["source_policy"] not in {"required", "optional", "none"}:
            raise ValueError(f"Question {item['id']} has invalid source_policy {item['source_policy']!r}.")
        if item["expected_status"] not in {"supported", "partial", "unsupported"}:
            raise ValueError(f"Question {item['id']} has invalid expected_status {item['expected_status']!r}.")
    corpus_dir = BASE_DIR.parent / "corpus"
    available_sources = {path.name for path in corpus_dir.glob("*.md")}
    for item in dataset:
        missing_sources = set(item["expected_sources"]) - available_sources
        if missing_sources:
            raise ValueError(
                f"Question {item['id']} expects source file(s) missing from corpus: {', '.join(sorted(missing_sources))}"
            )
    return dataset


def ask_http(url: str, question: str) -> dict:
    request = Request(
        url.rstrip("/") + "/ask",
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def validate_api_contract(health: dict) -> None:
    version = health.get("api_contract_version")
    if version != API_CONTRACT_VERSION:
        raise RuntimeError(
            f"API contract mismatch: evaluator requires version {API_CONTRACT_VERSION}, "
            f"server reports {version!r}. Restart the API (or rebuild/restart its container) before evaluation."
        )


def check_api_contract(url: str) -> None:
    request = Request(url.rstrip("/") + "/health", headers={"Accept": "application/json"})
    with urlopen(request, timeout=10) as response:
        validate_api_contract(json.load(response))


def get_agent():
    from app.core.config import get_settings
    from app.main import AgentRuntime
    return AgentRuntime(get_settings()).get_agent()


def extract_source_files(sources: list) -> list[str]:
    """Accept source records containing file, path, or source_file."""
    files = []
    for source in sources or []:
        if isinstance(source, str):
            candidate = source
        elif isinstance(source, dict):
            candidate = (
                source.get("file")
                or source.get("source_file")
                or source.get("path")
            )
        else:
            candidate = None
        if candidate:
            # Normalize paths such as corpus/dailymed-warfarin.md.
            files.append(Path(str(candidate).replace("\\", "/")).name)
    return list(dict.fromkeys(files))


def normalize_tool_names(value) -> list[str]:
    """Normalize tools_used whether the API returns strings or tool objects."""
    names = []
    for item in value or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("tool") or item.get("tool_name")
            if name:
                names.append(str(name))
    return list(dict.fromkeys(names))


def count_duplicate_citations(answer: str) -> int:
    total = 0
    for sentence in re.split(r"(?<=[.!?])(?=\s|$)", answer or ""):
        citations = re.findall(r"\[([^\]]+)\]", sentence)
        total += len(citations) - len(set(citations))
    return total


def latency_profile(item: dict) -> tuple[str, dict[str, int]]:
    category = str(item.get("category", "")).casefold()
    question = str(item.get("question", "")).casefold()
    if category == "multi_hop" or category == "glossary_reasoning":
        name = "multi_hop"
    elif re.search(r"\b(compare|comparison|versus|vs\.?|differences?|shared|unique)\b", question):
        name = "complex"
    elif re.search(r"\b(and also|across the relevant|both .+ and|in addition)\b", question):
        name = "multi_hop"
    elif category == "single_document":
        name = "simple"
    else:
        name = "multi_document"
    return name, LATENCY_PROFILES[name]


def contains_medical_disclaimer(answer: str) -> bool:
    text = (answer or "").casefold()
    indicators = (
        "not medical advice",
        "not a substitute for medical advice",
        "does not constitute medical advice",
        "not intended as medical advice",
        "ne constitue pas un avis médical",
    )
    return any(term in text for term in indicators)


def effective_answer_status(answer_status: str | None, answer: str) -> tuple[str | None, str]:
    if answer_status in {"supported", "partial", "unsupported"}:
        return answer_status, "api"
    if has_abstention_justification(answer):
        return "unsupported", "inferred_from_explicit_abstention"
    return None, "unavailable"


def evaluate_question(item: dict, ask) -> dict:
    question = item["question"]
    expected = item.get("expected_sources", [])
    result = {
        "id": item.get("id"),
        "category": item.get("category", "uncategorized"),
        "question": question,
        "expected_sources": expected,
        "expected_status": item.get("expected_status"),
    }

    try:
        output = ask(question) or {}
        answer = output.get("answer", "") or ""
        sources = output.get("sources", []) or []
        actual = extract_source_files(sources)
        critic = output.get("critic") or {}
        tools = normalize_tool_names(output.get("tools_used", []))
        trace = output.get("tool_trace", []) or []
        plan = output.get("plan") or {}
        plan_steps = plan.get("steps", []) if isinstance(plan, dict) else []
        valid_plan_steps = [step for step in plan_steps if isinstance(step, dict)]
        planned_tools = list(dict.fromkeys(
            tool
            for step in valid_plan_steps
            for tool in normalize_tool_names(step.get("tools", []))
        ))
        plan_valid = bool(plan_steps) and len(valid_plan_steps) == len(plan_steps) and all(
            "search" in normalize_tool_names(step.get("tools", []))
            for step in valid_plan_steps
        )
        planned_tool_pairs = [
            (step.get("id"), tool)
            for step in valid_plan_steps
            for tool in normalize_tool_names(step.get("tools", []))
        ]
        executed_tool_pairs = {
            (entry.get("step"), entry.get("tool"))
            for entry in trace if isinstance(entry, dict)
        } if isinstance(trace, list) else set()
        executed_planned_tool_pairs = [pair for pair in planned_tool_pairs if pair in executed_tool_pairs]
        answer_status = output.get("answer_status")
        expected_status = item.get("expected_status")
        explicit_abstention = has_abstention_justification(answer)
        effective_status, status_source = effective_answer_status(answer_status, answer)
        retrieved_count = sum(
            int(entry.get("passages", 0) or 0)
            for entry in trace if isinstance(entry, dict)
        ) if isinstance(trace, list) else 0
        if retrieved_count == 0:
            retrieval_state = "no_documents_retrieved"
        elif effective_status == "unsupported" or (expected_status == "unsupported" and explicit_abstention):
            retrieval_state = "retrieved_but_answer_unsupported"
        elif not actual:
            retrieval_state = "retrieved_without_cited_evidence"
        else:
            retrieval_state = "cited_evidence"
        latency_class, profile = latency_profile(item)
        latency_ms = output.get("latency_ms")

        result.update({
            "status": "success",
            "answer": answer,
            "answer_status": answer_status,
            "effective_answer_status": effective_status,
            "answer_status_source": status_source,
            "answer_status_correct": effective_status == expected_status if expected_status and effective_status else None,
            "tools_used": tools,
            "tool_call_count": len(trace) if isinstance(trace, list) else 0,
            "tool_trace": trace,
            "plan_intent": plan.get("intent") if isinstance(plan, dict) else None,
            "plan_origin": plan.get("origin") if isinstance(plan, dict) else None,
            "plan_steps": plan_steps,
            "planned_tools": planned_tools,
            "plan_valid": plan_valid,
            "planned_tools_executed": [tool for _, tool in executed_planned_tool_pairs],
            "planned_tool_execution_rate": (
                len(executed_planned_tool_pairs) / len(planned_tool_pairs)
                if planned_tool_pairs else None
            ),
            "retrieval_state": retrieval_state,
            "latency_class": latency_class,
            "latency_target_ms": profile["target_ms"],
            "latency_max_ms": profile["max_ms"],
            "latency_target_met": isinstance(latency_ms, (int, float)) and latency_ms <= profile["target_ms"],
            "latency_max_met": isinstance(latency_ms, (int, float)) and latency_ms <= profile["max_ms"],
            "critic": {
                "verdict": critic.get("verdict"),
                "review_status": critic.get("review_status"),
                "grounding_score": critic.get("grounding_score"),
                "revisions": critic.get("revisions", 0),
                "attempts": critic.get("attempts", 0),
                "llm_ok": critic.get("llm_ok"),
                "flags": critic.get("flags", []),
                "claims": critic.get("claims", []),
            },
            "latency_ms": latency_ms,
            "timings_ms": output.get("timings_ms", {}),
            "actual_sources": actual,
            "duplicate_citations": count_duplicate_citations(answer),
            "has_medical_disclaimer": contains_medical_disclaimer(answer),
            "answer_nonempty": bool(answer.strip()),
        })

        # Source metrics are meaningful only when the question has a known
        # expected-source answer key. Do not inflate averages with abstention
        # questions whose expected_sources list is intentionally empty.
        if expected:
            precision = source_precision(expected, actual)
            recall = source_recall(expected, actual)
            result.update({
                "source_recall": recall,
                "source_precision": precision,
                "source_f1": f1_score(precision, recall),
            })
        else:
            result["expected_abstention"] = True
            result["abstained_without_sources"] = retrieval_state == "no_documents_retrieved"
            result["source_precision"] = None
            result["correct_abstention"] = (
                item.get("expected_status") == "unsupported"
                and explicit_abstention
            )
        result["source_policy"] = item.get("source_policy", "required" if expected else "none")
        result["source_policy_satisfied"] = (
            set(expected).issubset(actual)
            if result["source_policy"] == "required"
            else not actual
            if result["source_policy"] == "none"
            else True
        )

        required_terms = [
            str(term).strip()
            for term in item.get("required_terms", [])
            if str(term).strip()
        ]
        if required_terms:
            alternatives = item.get("required_term_alternatives", {})
            matched = [
                term for term in required_terms
                if required_term_present(answer, term, alternatives.get(term, []))
            ]
            result["required_term_recall"] = len(matched) / len(required_terms)
            result["matched_required_terms"] = matched
            result["missing_required_terms"] = [
                term for term in required_terms if term not in matched
            ]

    except Exception as error:
        result.update({"status": "error", "error": str(error)})

    return result


def build_summary(results: list[dict]) -> dict:
    successful = [r for r in results if r["status"] == "success"]
    expected_source_results = [r for r in successful if r.get("expected_sources")]
    abstention_results = [r for r in successful if not r.get("expected_sources")]
    unsupported_cases = [r for r in successful if r.get("expected_status") == "unsupported"]
    supported_cases = [r for r in successful if r.get("expected_status") == "supported"]
    status_keyed_cases = [r for r in successful if r.get("expected_status") in {"supported", "partial", "unsupported"}]
    status_observed_cases = [
        r for r in status_keyed_cases if r.get("effective_answer_status") in {"supported", "partial", "unsupported"}
    ]

    latencies = [
        r["latency_ms"] for r in successful
        if isinstance(r.get("latency_ms"), (int, float))
    ]
    critic_scores = [
        r["critic"]["grounding_score"] for r in successful
        if isinstance(r.get("critic", {}).get("grounding_score"), (int, float))
    ]
    verdicts = [
        str(r.get("critic", {}).get("verdict", "")).casefold()
        for r in successful
    ]
    critic_review_status = [
        r["critic"]["llm_ok"] for r in successful
        if isinstance(r.get("critic", {}).get("llm_ok"), bool)
    ]
    critic_reviews_attempted = [
        r for r in successful if (r.get("critic", {}).get("attempts", 0) or 0) > 0
    ]
    critic_reviews_completed = [
        r for r in critic_reviews_attempted
        if r.get("critic", {}).get("llm_ok") is True
        and r.get("critic", {}).get("verdict") in {"pass", "revise"}
    ]
    tools_counter = Counter(
        tool for r in successful for tool in r.get("tools_used", [])
    )
    term_scores = [
        r["required_term_recall"] for r in successful
        if "required_term_recall" in r
    ]
    revisions = sum(
        (r.get("critic", {}).get("revisions", 0) or 0)
        for r in successful
    )
    duplicate_citations = sum(r.get("duplicate_citations", 0) for r in successful)
    plan_results = [r for r in successful if r.get("plan_steps")]
    tool_plan_results = [
        r for r in plan_results if isinstance(r.get("planned_tool_execution_rate"), (int, float))
    ]
    latency_by_class = {}
    for name in LATENCY_PROFILES:
        group = [r for r in successful if r.get("latency_class") == name and isinstance(r.get("latency_ms"), (int, float))]
        latency_by_class[name] = {
            "questions": len(group),
            "average_ms": average([r["latency_ms"] for r in group]),
            "max_ms": max((r["latency_ms"] for r in group), default=0),
            "target_ms": LATENCY_PROFILES[name]["target_ms"],
            "target_met_rate": average([float(r["latency_ms"] <= LATENCY_PROFILES[name]["target_ms"]) for r in group]),
            "max_ms_limit": LATENCY_PROFILES[name]["max_ms"],
            "max_limit_met_rate": average([float(r["latency_ms"] <= LATENCY_PROFILES[name]["max_ms"]) for r in group]),
        }

    return {
        "total_questions": len(results),
        "successful_executions": len(successful),
        "failed_executions": len(results) - len(successful),
        "execution_success_rate": len(successful) / len(results) if results else 0.0,

        # Retrieval quality, calculated only on questions with a source answer key.
        "questions_with_expected_sources": len(expected_source_results),
        "average_source_recall": average([r["source_recall"] for r in expected_source_results]),
        "average_source_precision": average([r["source_precision"] for r in expected_source_results]),
        "average_source_precision_with_expected": average([r["source_precision"] for r in expected_source_results]),
        "average_source_f1": average([r["source_f1"] for r in expected_source_results]),

        # Abstention behavior, calculated only on deliberately unsupported questions.
        "questions_without_expected_sources": len(abstention_results),
        "abstention_without_retrieved_sources_rate": average([
            float(r.get("abstained_without_sources", False)) for r in unsupported_cases
        ]),
        "correct_abstention_rate": average([float(r.get("correct_abstention", False)) for r in unsupported_cases]),
        "correct_abstention_cases": sum(bool(r.get("correct_abstention", False)) for r in unsupported_cases),
        "unsupported_cases_evaluated": len(unsupported_cases),
        "retrieval_state_counts": dict(Counter(r.get("retrieval_state", "unknown") for r in unsupported_cases)),
        "abstention_with_retrieval_but_unsupported_rate": average([
            float(r.get("retrieval_state") == "retrieved_but_answer_unsupported")
            for r in unsupported_cases
        ]),
        "empty_expected_source_free_rate": average([
            float(not r.get("actual_sources")) for r in abstention_results
            if r.get("source_policy") == "none"
        ]),
        "false_refusal_rate": average([float(r.get("effective_answer_status") == "unsupported") for r in supported_cases]),
        "unsupported_answer_rate": average([
            float(r.get("effective_answer_status") != "unsupported") for r in unsupported_cases
            if r.get("effective_answer_status") in {"supported", "partial", "unsupported"}
        ]),
        "source_policy_compliance_rate": average([
            float(r.get("source_policy_satisfied", False)) for r in successful
        ]),
        "answer_status_accuracy": average([
            float(r.get("answer_status_correct", False)) for r in status_observed_cases
        ]),
        "answer_status_coverage_rate": len(status_observed_cases) / len(status_keyed_cases) if status_keyed_cases else 0.0,

        # Additional diagnostic indicators; not substitutes for human review.
        "average_required_term_recall": average(term_scores),
        "questions_with_required_terms": len(term_scores),
        "critic_pass_rate": average([
            float(r["critic"]["verdict"] == "pass") for r in critic_reviews_completed
        ]),
        "critic_review_available_rate": average([float(available) for available in critic_review_status]),
        "questions_with_critic_review_status": len(critic_review_status),
        "critic_reviews_attempted": len(critic_reviews_attempted),
        "critic_reviews_completed": len(critic_reviews_completed),
        "average_grounding_score": average(critic_scores),
        "answers_with_medical_disclaimer_rate": average([
            float(r.get("has_medical_disclaimer", False)) for r in successful
        ]),
        "nonempty_answer_rate": average([
            float(r.get("answer_nonempty", False)) for r in successful
        ]),
        "unique_tools_used": sorted(tools_counter),
        "tool_usage_counts": dict(tools_counter),
        "questions_using_at_least_one_tool": sum(
            bool(r.get("tools_used") or r.get("tool_trace")) for r in successful
        ),
        "total_revisions": revisions,
        "duplicate_citations": duplicate_citations,
        "planner_plan_valid_rate": average([float(r.get("plan_valid", False)) for r in plan_results]),
        "planner_plan_count": len(plan_results),
        "planner_fallback_rate": average([float(r.get("plan_origin") == "fallback") for r in plan_results]),
        "planned_tool_execution_rate": average([
            r["planned_tool_execution_rate"] for r in tool_plan_results
        ]),

        "average_latency_ms": average(latencies),
        "median_latency_ms": statistics.median(latencies) if latencies else 0,
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "max_latency_ms": max(latencies) if latencies else 0,
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "requests_over_latency_budget": sum(x > LATENCY_BUDGET_MS for x in latencies),
        "latency_budget_met_rate": average([
            float(x <= LATENCY_BUDGET_MS) for x in latencies
        ]),
        "latency_profiles": LATENCY_PROFILES,
        "latency_by_class": latency_by_class,
        "latency_class_max_limit_met_rate": average([
            float(r.get("latency_max_met", False)) for r in successful if isinstance(r.get("latency_ms"), (int, float))
        ]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Base URL, e.g. http://localhost:8000")
    args = parser.parse_args()

    dataset = load_dataset()
    print(f"\\nLoaded {len(dataset)} evaluation questions.\\n")

    if args.url:
        check_api_contract(args.url)
        ask = lambda question: ask_http(args.url, question)
    else:
        agent = get_agent()
        ask = agent.ask

    results = []
    for index, item in enumerate(dataset, start=1):
        print("=" * 90)
        print(f"[{index}/{len(dataset)}] {item['question']}")
        result = evaluate_question(item, ask)
        results.append(result)

        if result["status"] == "success":
            print(
                f"Status=success | "
                f"Recall={result.get('source_recall', 'N/A')} | "
                f"Precision={result.get('source_precision', 'N/A')} | "
                f"Latency={result.get('latency_ms')}ms"
            )
            print("Tools:", result.get("tools_used", []))
            print("Sources:", result.get("actual_sources", []))
            print("Critic:", result.get("critic", {}).get("verdict"))
        else:
            print("ERROR:", result["error"])

    summary = build_summary(results)
    report = {
        "project": "Syntax Agentic RAG Evaluation",
        "evaluation_date": datetime.now().astimezone().isoformat(),
        "dataset": DATASET_FILE.name,
        "summary": summary,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("evaluation_%Y%m%d_%H%M%S.json")
    report_path = RESULTS_DIR / filename
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\\n" + "=" * 90)
    print("GLOBAL EVALUATION SUMMARY")
    print("=" * 90)
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\\nJSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
