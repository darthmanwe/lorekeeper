"""Unit tests for pipeline.py — routing logic, baseline mode, and strict retry.

These test the pipeline's internal routing decisions and mode branching
without invoking real LLM or database calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.guard import ContradictionGuard
from src.pipeline import PipelineState, StoryPipeline
from src.schema import ConstraintViolation, SessionState


def _make_pipeline(guard: ContradictionGuard | None = None) -> StoryPipeline:
    """Create a StoryPipeline with all dependencies mocked."""
    return StoryPipeline(
        graph_client=MagicMock(),
        cypher_retriever=MagicMock(),
        vector_retriever=MagicMock(),
        extraction=MagicMock(),
        llm=MagicMock(),
        guard=guard,
    )


class TestRouteAfterGenerate:
    """Verify the conditional routing logic after generation."""

    def test_nkge_mode_routes_to_extract(self) -> None:
        pipeline = _make_pipeline()
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="nkge"
            ),
            "violations": [],
        }
        assert pipeline._route_after_generate(state) == "extract"

    def test_baseline_mode_routes_to_end(self) -> None:
        pipeline = _make_pipeline()
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="baseline"
            ),
        }
        assert pipeline._route_after_generate(state) == "end"

    def test_strict_mode_with_blocking_violations_routes_to_retry(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="strict")
        pipeline = _make_pipeline(guard=guard)
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="nkge"
            ),
            "violations": [
                ConstraintViolation(
                    check_name="dead_character_active",
                    violation_message="Brann is dead",
                    severity="critical",
                ),
            ],
            "retry_count": 0,
        }
        assert pipeline._route_after_generate(state) == "retry"

    def test_strict_mode_retries_exhausted_routes_to_extract(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="strict")
        pipeline = _make_pipeline(guard=guard)
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="nkge"
            ),
            "violations": [
                ConstraintViolation(
                    check_name="dead_character_active",
                    violation_message="Brann is dead",
                    severity="critical",
                ),
            ],
            "retry_count": 2,
        }
        assert pipeline._route_after_generate(state) == "extract"

    def test_permissive_mode_with_violations_routes_to_extract(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="permissive")
        pipeline = _make_pipeline(guard=guard)
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="nkge"
            ),
            "violations": [
                ConstraintViolation(
                    check_name="dead_character_active",
                    violation_message="Brann is dead",
                    severity="critical",
                ),
            ],
            "retry_count": 0,
        }
        assert pipeline._route_after_generate(state) == "extract"


class TestPostGenerateCheck:
    """Verify post-generate check handles strict mode retries and branching."""

    def test_permissive_mode_passthrough(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="permissive")
        pipeline = _make_pipeline(guard=guard)
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="nkge"
            ),
            "violations": [
                ConstraintViolation(
                    check_name="x", violation_message="y", severity="critical"
                ),
            ],
        }
        result = pipeline._node_post_generate_check(state)
        assert result == {}

    def test_strict_mode_increments_retry(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="strict")
        pipeline = _make_pipeline(guard=guard)
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="nkge"
            ),
            "violations": [
                ConstraintViolation(
                    check_name="x", violation_message="y", severity="critical"
                ),
            ],
            "retry_count": 0,
        }
        result = pipeline._node_post_generate_check(state)
        assert result["retry_count"] == 1

    def test_baseline_mode_passthrough(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="strict")
        pipeline = _make_pipeline(guard=guard)
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="baseline"
            ),
            "violations": [],
        }
        result = pipeline._node_post_generate_check(state)
        assert result == {}


class TestPipelineStateTyping:
    """Verify PipelineState TypedDict structure."""

    def test_minimal_state_construction(self) -> None:
        state: PipelineState = {
            "session": SessionState(session_id="s1", story_seed="seed"),
            "player_action": "open the door",
        }
        assert state["player_action"] == "open the door"

    def test_full_state_construction(self) -> None:
        state: PipelineState = {
            "session": SessionState(session_id="s1", story_seed="seed"),
            "player_action": "attack",
            "graph_context": "facts",
            "graph_context_tokens": 100,
            "vector_context": "tone",
            "vector_context_tokens": 50,
            "violations": [],
            "persona_docs": [],
            "assembled_prompt": {"known_facts": "f"},
            "generated_text": "The story continues.",
            "extraction_result": None,
            "retry_count": 0,
            "error": None,
        }
        assert state["graph_context_tokens"] == 100
        assert state["generated_text"] == "The story continues."
