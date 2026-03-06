"""LangGraph StateGraph orchestration for the Lorekeeper segment generation cycle.

This file is a flagged deviation from the design document's repo structure
(Section 6.2) — see Design Decisions Log in study_packet.md. Without it,
the generation loop would be duplicated across notebooks, app.py, api.py,
and eval.py.

The pipeline implements the full read-write-verify loop:
  retrieve_graph -> retrieve_vector -> assemble_context -> run_guard ->
  generate -> extract_and_commit

Baseline mode skips graph retrieval (uses rolling text summary), guard,
and extraction, producing comparable output without the knowledge graph.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.extraction import ExtractionPipeline
from src.guard import BranchManager, ContradictionGuard
from src.prompts import get_prompt
from src.retrieval import ContextAssembler, CypherRetriever, VectorRetriever, count_tokens
from src.schema import (
    ConstraintViolation,
    ExtractionResult,
    Segment,
    SessionState,
)

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    """Typed state for the LangGraph segment generation pipeline.

    All fields are optional (total=False) because different nodes
    populate different fields during the pipeline's execution.
    """

    session: SessionState
    player_action: str
    graph_context: str
    graph_context_tokens: int
    vector_context: str
    vector_context_tokens: int
    violations: list[ConstraintViolation]
    persona_docs: list[str]
    assembled_prompt: dict[str, str]
    generated_text: str
    extraction_result: ExtractionResult | None
    retry_count: int
    error: str | None


class StoryPipeline:
    """Orchestrates the full segment generation cycle as a LangGraph StateGraph.

    Args:
        graph_client: Neo4j GraphClient instance.
        cypher_retriever: Tiered Cypher retrieval engine.
        vector_retriever: ChromaDB-backed vector retrieval.
        extraction: Two-stage extraction pipeline.
        llm: ChatAnthropic instance for generation.
        guard: ContradictionGuard instance (injected in P4, optional until then).
    """

    def __init__(
        self,
        graph_client: Any,
        cypher_retriever: CypherRetriever,
        vector_retriever: VectorRetriever,
        extraction: ExtractionPipeline,
        llm: ChatAnthropic,
        guard: ContradictionGuard | None = None,
        branch_manager: BranchManager | None = None,
    ) -> None:
        self._gc = graph_client
        self._cypher = cypher_retriever
        self._vector = vector_retriever
        self._extraction = extraction
        self._llm = llm
        self._guard = guard
        self._branch_mgr = branch_manager
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph StateGraph with conditional routing.

        The graph includes a post-generation guard re-check node for strict
        mode. If blocking violations persist after generation and retries
        are exhausted, a branch is created to isolate the divergent state.
        """
        builder = StateGraph(PipelineState)

        builder.add_node("retrieve_graph", self._node_retrieve_graph)
        builder.add_node("retrieve_vector", self._node_retrieve_vector)
        builder.add_node("assemble_context", self._node_assemble_context)
        builder.add_node("run_guard", self._node_run_guard)
        builder.add_node("generate", self._node_generate)
        builder.add_node("post_generate_check", self._node_post_generate_check)
        builder.add_node("extract_and_commit", self._node_extract_and_commit)

        builder.set_entry_point("retrieve_graph")
        builder.add_edge("retrieve_graph", "retrieve_vector")
        builder.add_edge("retrieve_vector", "assemble_context")
        builder.add_edge("assemble_context", "run_guard")
        builder.add_edge("run_guard", "generate")
        builder.add_edge("generate", "post_generate_check")
        builder.add_conditional_edges(
            "post_generate_check",
            self._route_after_generate,
            {
                "extract": "extract_and_commit",
                "retry": "generate",
                "end": END,
            },
        )
        builder.add_edge("extract_and_commit", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Pipeline nodes
    # ------------------------------------------------------------------

    def _node_retrieve_graph(self, state: PipelineState) -> dict[str, Any]:
        """Retrieve graph context: tiered Cypher for NKGE, rolling summary for baseline."""
        session: SessionState = state["session"]

        if session.mode == "baseline":
            summary = self._build_rolling_summary(session)
            tokens = count_tokens(summary)
            logger.info("Baseline mode: rolling summary (%d tokens)", tokens)
            return {
                "graph_context": summary,
                "graph_context_tokens": tokens,
            }

        context, tokens = self._cypher.retrieve(session)
        logger.info("NKGE graph retrieval: %d tokens", tokens)
        return {
            "graph_context": context,
            "graph_context_tokens": tokens,
        }

    def _node_retrieve_vector(self, state: PipelineState) -> dict[str, Any]:
        """Retrieve tonal anchors from ChromaDB with anti-parroting filter."""
        session: SessionState = state["session"]
        last_texts = [session.last_segment_text] if session.last_segment_text else []

        context, tokens = self._vector.retrieve(
            last_texts,
            token_budget=1000,
            exclude_text=session.last_segment_text or None,
        )
        logger.info("Vector retrieval: %d tokens, %d anchors", tokens, context.count("[Tonal anchor"))
        return {
            "vector_context": context,
            "vector_context_tokens": tokens,
        }

    def _node_assemble_context(self, state: PipelineState) -> dict[str, Any]:
        """Merge all context sources into structured prompt sections."""
        session: SessionState = state["session"]
        sections = ContextAssembler.assemble(
            graph_context=state.get("graph_context", ""),
            vector_context=state.get("vector_context", ""),
            violations=state.get("violations"),
            persona_docs=state.get("persona_docs"),
            session=session,
        )
        return {"assembled_prompt": sections}

    def _node_run_guard(self, state: PipelineState) -> dict[str, Any]:
        """Run contradiction guard checks and inject violations into context.

        In permissive mode, violations are injected into the prompt but
        never block generation. In strict mode, blocking violations
        (Critical/Major) will trigger retries in the generate node.
        """
        if self._guard is None:
            return {"violations": []}

        session: SessionState = state["session"]

        if session.mode == "baseline":
            return {"violations": []}

        violations = self._guard.run_all_checks(session)
        logger.info("Guard check: %d violations found", len(violations))

        if violations:
            sections = ContextAssembler.assemble(
                graph_context=state.get("graph_context", ""),
                vector_context=state.get("vector_context", ""),
                violations=violations,
                persona_docs=state.get("persona_docs"),
                session=session,
            )
            return {"violations": violations, "assembled_prompt": sections}

        return {"violations": violations}

    def _node_generate(self, state: PipelineState) -> dict[str, Any]:
        """Generate the next story segment via Claude."""
        prompt = get_prompt("generation_v1")
        sections = state.get("assembled_prompt", {})
        player_action = state.get("player_action", "")

        system_msg = prompt.format_system(
            known_facts=sections.get("known_facts", ""),
            constraints=sections.get("constraints", ""),
            character_voices=sections.get("character_voices", ""),
            tonal_context=sections.get("tonal_context", ""),
        )
        user_msg = prompt.format_user(
            previous_segment=sections.get("previous_segment", ""),
            player_action=player_action,
        )

        response = self._llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ])

        generated = response.content
        logger.info("Generated segment: %d chars, ~%d tokens", len(generated), count_tokens(generated))
        return {"generated_text": generated}

    def _node_extract_and_commit(self, state: PipelineState) -> dict[str, Any]:
        """Run extraction pipeline on the generated text and commit to graph."""
        session: SessionState = state["session"]

        if session.mode == "baseline":
            return {"extraction_result": None}

        generated = state.get("generated_text", "")
        next_seq = session.last_segment_seq_id + 1

        seg = Segment(
            text=generated,
            seq_id=next_seq,
            branch_id=session.active_branch_id,
        )
        self._gc.merge_segment(seg)

        self._vector.add_segment(
            generated,
            f"{session.active_branch_id}_seq_{next_seq}",
        )

        result = self._extraction.run(
            segment_text=generated,
            branch_id=session.active_branch_id,
            seq_id=next_seq,
            auto_approve=True,
        )
        logger.info(
            "Extraction: %d proposed, %d approved, %d flagged, %d committed",
            len(result.proposals),
            len(result.approved),
            len(result.flagged),
            result.committed_count,
        )
        return {"extraction_result": result}

    _MAX_STRICT_RETRIES = 2

    def _node_post_generate_check(self, state: PipelineState) -> dict[str, Any]:
        """Post-generation guard re-check for strict mode retry logic.

        In strict mode, if the generated text still triggers blocking
        violations (Critical/Major), increment retry_count and signal
        a retry. After _MAX_STRICT_RETRIES, create a branch to isolate
        the divergent state and proceed with the generated text as-is.

        In permissive mode or baseline mode, this is a pass-through.
        """
        session: SessionState = state["session"]

        if session.mode == "baseline" or self._guard is None:
            return {}

        if self._guard.mode != "strict":
            return {}

        violations = state.get("violations", [])
        if not self._guard.has_blocking_violations(violations):
            return {}

        retry_count = state.get("retry_count", 0)
        if retry_count >= self._MAX_STRICT_RETRIES:
            if self._branch_mgr:
                blocking = [v for v in violations if v.severity in ("critical", "major")]
                reason = f"Unresolved after {retry_count} retries: {blocking[0].violation_message}"
                new_branch = self._branch_mgr.create_branch(session, reason)
                logger.warning(
                    "Strict mode: max retries exhausted, branched to %s", new_branch
                )
            else:
                logger.warning(
                    "Strict mode: max retries exhausted, no branch manager — proceeding with violations"
                )
            return {}

        logger.info(
            "Strict mode: blocking violations detected, retry %d/%d",
            retry_count + 1, self._MAX_STRICT_RETRIES,
        )
        return {"retry_count": retry_count + 1}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_generate(self, state: PipelineState) -> str:
        """Route after post-generate check: extract, retry, or end."""
        session: SessionState = state["session"]

        if session.mode == "baseline":
            return "end"

        if (
            self._guard is not None
            and self._guard.mode == "strict"
            and self._guard.has_blocking_violations(state.get("violations", []))
            and state.get("retry_count", 0) < self._MAX_STRICT_RETRIES
        ):
            return "retry"

        return "extract"

    # ------------------------------------------------------------------
    # Baseline helper
    # ------------------------------------------------------------------

    def _build_rolling_summary(self, session: SessionState) -> str:
        """Build a rolling text summary of the last 3 segments for baseline mode.

        Baseline mode replaces graph context with raw concatenation of recent
        segment texts. This is the realistic alternative to building a graph —
        not trivially weak, but lacking structured memory.

        Args:
            session: Current session state.

        Returns:
            Concatenated text of the last 3 segments.
        """
        query = """
        MATCH (s:Segment {branch_id: $branch_id})
        WHERE s.seq_id <= $max_seq AND s.seq_id > $min_seq
        RETURN s.text AS text
        ORDER BY s.seq_id DESC
        LIMIT 3
        """
        min_seq = max(0, session.last_segment_seq_id - 3)
        with self._gc._driver.session(database=self._gc._database) as db_session:
            result = db_session.run(query, {
                "branch_id": session.active_branch_id,
                "max_seq": session.last_segment_seq_id,
                "min_seq": min_seq,
            })
            texts = [r["text"] for r in result]

        if not texts:
            return "(no previous segments)"

        return "RECENT STORY CONTEXT (last segments):\n\n" + "\n\n---\n\n".join(texts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, session: SessionState, player_action: str) -> PipelineState:
        """Execute the full segment generation pipeline.

        Args:
            session: Current SessionState with location, branch, characters.
            player_action: The player's input action/choice.

        Returns:
            Completed PipelineState with generated text and extraction results.
        """
        initial_state: PipelineState = {
            "session": session,
            "player_action": player_action,
            "retry_count": 0,
            "violations": [],
            "persona_docs": [],
        }

        result = self._graph.invoke(initial_state)
        return result
