"""Tool contract shared by the six agent tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.config import Settings
from app.core.llm import ChatLLM
from app.ingestion.indexer import VectorIndex
from app.retrieval.retriever import Passage, Retriever


@dataclass
class ToolContext:
    index: VectorIndex
    retriever: Retriever
    llm: ChatLLM
    settings: Settings

    @property
    def catalog(self):
        return self.index.catalog


@dataclass
class ToolResult:
    tool: str
    args: dict[str, Any]
    passages: list[Passage] = field(default_factory=list)
    notes: str = ""  # machine-generated hint for the answer writer (never cited directly)
    primary: bool = True  # primary evidence gets priority when the prompt budget is tight


class Tool(Protocol):
    name: str
    description: str

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult: ...
