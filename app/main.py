"""FastAPI entry point: `uvicorn app.main:app`."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import ResearchAgent
from app.agent.nodes import AgentDeps
from app.api.metrics import RuntimeMetrics
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.llm import ChatLLM, LLMClient
from app.ingestion.indexer import build_index
from app.retrieval.embeddings import Embedder
from app.retrieval.retriever import Retriever
from app.tools.registry import build_tools

log = logging.getLogger("app")


class AgentRuntime:
    """Builds the index/agent once (in the background at startup) and hands it to request handlers."""

    def __init__(self, settings: Settings, llm: ChatLLM | None = None):
        self.settings = settings
        self.llm: ChatLLM = llm or LLMClient(settings)
        self._agent: ResearchAgent | None = None
        self._lock = threading.Lock()
        self.error: str | None = None
        self.metrics = RuntimeMetrics()

    @property
    def ready(self) -> bool:
        return self._agent is not None

    def get_agent(self) -> ResearchAgent:
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    self._agent = self._build()
        return self._agent

    def _build(self) -> ResearchAgent:
        try:
            embedder = Embedder(self.llm)
            index = build_index(self.settings, embedder)
            retriever = Retriever(index, embedder)
            agent = ResearchAgent(AgentDeps(self.llm, index, retriever, build_tools(), self.settings))
            self.error = None
            log.info("Agent ready (%d chunks, dense=%s)", len(index.chunks), index.matrix is not None)
            return agent
        except Exception as exc:
            self.error = str(exc)
            log.exception("Index build failed")
            raise

    def warm_up(self) -> None:
        try:
            self.get_agent()
        except Exception:
            pass  # error is exposed on /health; the next /ask retries


def create_app(settings: Settings | None = None, llm: ChatLLM | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runtime = AgentRuntime(settings, llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=runtime.warm_up, daemon=True, name="index-warmup").start()
        yield

    app = FastAPI(title="Pharmacology research agent", version="1.0.0", lifespan=lifespan,
                  description="Agentic RAG over FDA drug labels. Technical exercise - not medical advice.")
    app.state.runtime = runtime
    app.include_router(router)
    index_file = settings.frontend_dir / "index.html"
    if index_file.is_file():
        app.add_api_route(
            "/",
            lambda: RedirectResponse("/research", status_code=307),
            include_in_schema=False,
            name="frontend-root",
        )
        for page in ("research", "performance", "architecture", "insights"):
            app.add_api_route(
                f"/{page}",
                lambda index_file=index_file: FileResponse(index_file),
                include_in_schema=False,
                name=f"frontend-{page}",
            )
    if settings.frontend_dir.exists():  # mounted last so API routes keep priority
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


app = create_app()
