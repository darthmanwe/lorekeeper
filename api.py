"""FastAPI wrapper for the Lorekeeper pipeline.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  POST /generate       — run one generation cycle
  GET  /session        — get current session state
  POST /session/reset  — reset the session to seed state
  GET  /graph/stats    — node and relationship counts
  GET  /graph/facts    — structured facts for current branch
  GET  /health         — service health check
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from src.tracing import init_tracing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    player_action: str = Field(
        ..., min_length=1, max_length=500,
        description="The player's action or dialogue input.",
    )
    mode: str | None = Field(
        default=None,
        description="Override session mode: 'nkge' or 'baseline'.",
    )


class ViolationResponse(BaseModel):
    check_name: str
    violation_message: str
    severity: str


class ExtractionSummary(BaseModel):
    proposed: int
    approved: int
    flagged: int
    committed: int


class GenerateResponse(BaseModel):
    seq_id: int
    branch_id: str
    generated_text: str
    graph_context_tokens: int
    vector_context_tokens: int
    violations: list[ViolationResponse]
    extraction: ExtractionSummary | None
    mode: str


class SessionResponse(BaseModel):
    session_id: str
    story_seed: str
    active_branch_id: str
    current_location: str
    present_characters: list[str]
    last_segment_seq_id: int
    mode: str


class GraphStatsResponse(BaseModel):
    node_counts: dict[str, int]
    relationship_counts: dict[str, int]
    total_nodes: int
    total_relationships: int


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    service: str


# ---------------------------------------------------------------------------
# App state — initialized on startup, shared across requests
# ---------------------------------------------------------------------------


class AppState:
    """Holds pipeline components initialized once at startup."""

    def __init__(self) -> None:
        from langchain_anthropic import ChatAnthropic

        from src.extraction import ExtractionPipeline
        from src.graph_client import GraphClient
        from src.guard import BranchManager, ContradictionGuard
        from src.persona import PersonaStore
        from src.pipeline import StoryPipeline
        from src.retrieval import CypherRetriever, VectorRetriever
        from src.schema import SessionState

        self.gc = GraphClient(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", ""),
        )
        self.gc.verify_connectivity()

        llm = ChatAnthropic(
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=0.7,
            max_tokens=1024,
        )

        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store/")
        self.persona_store = PersonaStore(persist_dir=chroma_dir)
        guard = ContradictionGuard(
            self.gc, mode=os.getenv("GUARD_MODE", "permissive")
        )
        branch_mgr = BranchManager(self.gc)

        self.pipeline = StoryPipeline(
            graph_client=self.gc,
            cypher_retriever=CypherRetriever(self.gc, token_budget=2000),
            vector_retriever=VectorRetriever(persist_dir=chroma_dir),
            extraction=ExtractionPipeline(llm=llm, graph_client=self.gc),
            llm=llm,
            guard=guard,
            branch_manager=branch_mgr,
            persona_store=self.persona_store,
        )

        self.session = SessionState(
            session_id=str(uuid.uuid4()),
            story_seed="The Iron Tavern",
            active_branch_id="main",
            current_location="Iron Tavern",
            present_characters=["Kael", "Elara"],
            last_segment_seq_id=3,
            last_segment_text="",
            mode="nkge",
        )

        logger.info("AppState initialized: session=%s", self.session.session_id)


_state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Initialize components on startup, clean up on shutdown."""
    global _state  # noqa: PLW0603
    init_tracing()
    _state = AppState()
    logger.info("Lorekeeper API ready")
    yield
    if _state is not None:
        _state.gc.close()
        logger.info("Neo4j connection closed")


def _get_state() -> AppState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _state


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Lorekeeper API",
    description=(
        "REST API for the Lorekeeper Narrative Knowledge Graph Engine. "
        "Generates story segments grounded in a Neo4j knowledge graph with "
        "dual RAG retrieval, contradiction guard, and persona-consistent "
        "character voices."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Run one segment generation cycle.

    Executes the full pipeline: retrieve -> guard -> generate -> extract.
    """
    s = _get_state()

    if req.mode and req.mode in ("nkge", "baseline"):
        s.session.mode = req.mode

    result = s.pipeline.run(s.session, req.player_action)

    generated = result.get("generated_text", "")
    violations = result.get("violations", [])
    ext = result.get("extraction_result")

    s.session.last_segment_seq_id += 1
    s.session.last_segment_text = generated

    _refresh_present_characters(s)

    violation_list = []
    for v in violations:
        if hasattr(v, "check_name"):
            violation_list.append(ViolationResponse(
                check_name=v.check_name,
                violation_message=v.violation_message,
                severity=v.severity,
            ))

    extraction_summary = None
    if ext is not None:
        extraction_summary = ExtractionSummary(
            proposed=len(ext.proposals),
            approved=len(ext.approved),
            flagged=len(ext.flagged),
            committed=ext.committed_count,
        )

    return GenerateResponse(
        seq_id=s.session.last_segment_seq_id,
        branch_id=s.session.active_branch_id,
        generated_text=generated,
        graph_context_tokens=result.get("graph_context_tokens", 0),
        vector_context_tokens=result.get("vector_context_tokens", 0),
        violations=violation_list,
        extraction=extraction_summary,
        mode=s.session.mode,
    )


@app.get("/session", response_model=SessionResponse)
async def get_session() -> SessionResponse:
    """Return the current session state."""
    s = _get_state()
    return SessionResponse(
        session_id=s.session.session_id,
        story_seed=s.session.story_seed,
        active_branch_id=s.session.active_branch_id,
        current_location=s.session.current_location,
        present_characters=s.session.present_characters,
        last_segment_seq_id=s.session.last_segment_seq_id,
        mode=s.session.mode,
    )


@app.post("/session/reset", response_model=SessionResponse)
async def reset_session() -> SessionResponse:
    """Reset the session to the initial seed state."""
    s = _get_state()
    from src.schema import SessionState

    s.session = SessionState(
        session_id=str(uuid.uuid4()),
        story_seed="The Iron Tavern",
        active_branch_id="main",
        current_location="Iron Tavern",
        present_characters=["Kael", "Elara"],
        last_segment_seq_id=3,
        last_segment_text="",
        mode="nkge",
    )
    return await get_session()


@app.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats() -> GraphStatsResponse:
    """Return node and relationship counts from Neo4j."""
    s = _get_state()
    nodes = s.gc.get_node_counts()
    rels = s.gc.get_relationship_counts()
    return GraphStatsResponse(
        node_counts=nodes,
        relationship_counts=rels,
        total_nodes=sum(nodes.values()),
        total_relationships=sum(rels.values()),
    )


@app.get("/graph/facts")
async def graph_facts() -> dict[str, Any]:
    """Return structured facts for the current branch."""
    s = _get_state()
    facts = s.gc.get_graph_summary_facts(s.session.active_branch_id)
    return {"branch_id": s.session.active_branch_id, "facts": facts}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health check including Neo4j connectivity."""
    s = _get_state()
    try:
        s.gc.verify_connectivity()
        neo4j_status = "connected"
    except Exception:
        neo4j_status = "disconnected"
    return HealthResponse(
        status="ok" if neo4j_status == "connected" else "degraded",
        neo4j=neo4j_status,
        service="lorekeeper",
    )


def _refresh_present_characters(s: AppState) -> None:
    """Update present_characters from the graph after generation."""
    try:
        chars = s.gc.get_characters_at_location(
            s.session.current_location, s.session.active_branch_id
        )
        s.session.present_characters = sorted(c.name for c in chars)
    except Exception:
        logger.warning("Failed to refresh present characters")
