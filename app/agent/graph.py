"""A tiny explicit state graph (LangGraph-style) - no heavy dependency, fully testable.

    plan -> execute -> synthesize -> review --(needs revision & budget left)--> revise -> review
                                        \\--(otherwise)--> finalize -> END
"""
from __future__ import annotations

import copy
import hashlib
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Callable

from app.agent import nodes
from app.agent.nodes import AgentDeps
from app.agent.state import AgentState

log = logging.getLogger(__name__)
END = "__end__"
NodeFn = Callable[[AgentState], None]


class Graph:
    def __init__(self, entry: str, max_steps: int = 20):
        self.entry = entry
        self.max_steps = max_steps
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, str] = {}
        self.routers: dict[str, Callable[[AgentState], str]] = {}

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, router: Callable[[AgentState], str]) -> None:
        self.routers[src] = router

    def run(self, state: AgentState) -> AgentState:
        current = self.entry
        for _ in range(self.max_steps):
            start = time.perf_counter()
            self.nodes[current](state)
            state.timings[current] = state.timings.get(current, 0) + int((time.perf_counter() - start) * 1000)
            current = self.routers[current](state) if current in self.routers else self.edges.get(current, END)
            if current == END:
                return state
        raise RuntimeError("Graph exceeded its step limit")


def build_graph(deps: AgentDeps) -> Graph:
    g = Graph(entry="plan")
    for name, fn in [
        ("plan", nodes.plan_node),
        ("execute", nodes.execute_node),
        ("synthesize", nodes.synthesize_node),
        ("review", nodes.review_node),
        ("revise", nodes.revise_node),
        ("finalize", nodes.finalize_node),
    ]:
        g.add_node(name, lambda s, fn=fn: fn(s, deps))
    g.add_edge("plan", "execute")
    g.add_edge("execute", "synthesize")
    g.add_edge("synthesize", "review")
    g.add_conditional_edges(
        "revise",
        lambda s: "review" if s.critic and s.critic.llm_ok else "finalize",
    )
    g.add_conditional_edges(
        "review",
        lambda s: (
            "revise"
            if (
                s.critic
                and s.critic.needs_revision
                and s.revisions < min(1, max(0, deps.settings.max_revisions))
                and s.plan
                and s.plan.intent not in {"definition", "verification", "out_of_scope"}
            )
            else "finalize"
        ),
    )
    g.add_edge("finalize", END)
    return g


class ResearchAgent:
    def __init__(self, deps: AgentDeps):
        self.deps = deps
        self.graph = build_graph(deps)
        corpus_material = "\0".join(
            f"{document.doc_id}\0{document.text}" for document in deps.index.documents
        ).encode("utf-8")
        self._corpus_fingerprint = hashlib.sha256(corpus_material).hexdigest()
        self._answer_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._answer_cache_lock = Lock()

    def ask(self, question: str, *, use_cache: bool = True) -> dict:
        start = time.perf_counter()
        normalized = " ".join(question.split()).casefold()
        key = (self._corpus_fingerprint, normalized)
        if use_cache:
            with self._answer_cache_lock:
                cached = self._answer_cache.get(key)
                if cached is not None:
                    self._answer_cache.move_to_end(key)
                    answer = copy.deepcopy(cached)
                    answer["latency_ms"] = int((time.perf_counter() - start) * 1000)
                    answer["timings_ms"] = {"cache": answer["latency_ms"]}
                    answer["cache_hit"] = True
                    return answer

        state = self.graph.run(AgentState(question=question.strip()))
        answer = dict(state.answer or {})
        answer["latency_ms"] = int((time.perf_counter() - start) * 1000)
        answer["timings_ms"] = state.timings
        answer["cache_hit"] = False
        if use_cache:
            with self._answer_cache_lock:
                self._answer_cache[key] = copy.deepcopy(answer)
                self._answer_cache.move_to_end(key)
                while len(self._answer_cache) > 128:
                    self._answer_cache.popitem(last=False)
        return answer
