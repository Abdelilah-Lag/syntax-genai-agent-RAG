"""Registry of the six tools. `search` is mandatory; the planner chooses among the others."""
from __future__ import annotations

from app.tools.base import Tool
from app.tools.compare_tool import CompareTool
from app.tools.extract_tool import ExtractTool
from app.tools.glossary_tool import GlossaryTool
from app.tools.interaction_tool import InteractionTool
from app.tools.search_tool import SearchTool
from app.tools.verify_tool import VerifyTool

MANDATORY_TOOL = "search"


def build_tools() -> dict[str, Tool]:
    tools: list[Tool] = [SearchTool(), ExtractTool(), CompareTool(), InteractionTool(), GlossaryTool(), VerifyTool()]
    return {t.name: t for t in tools}


OPTIONAL_TOOLS = ("extract", "compare", "interaction", "glossary", "verify")
