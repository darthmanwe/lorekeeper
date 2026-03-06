"""Unit tests for pipeline.py — routing logic and baseline mode.

These test the pipeline's internal routing decisions and mode branching
without invoking real LLM or database calls.
"""

from __future__ import annotations

import pytest

from src.pipeline import PipelineState, StoryPipeline
from src.schema import SessionState


class TestRouteAfterGenerate:
    """Verify the conditional routing logic after generation."""

    def test_nkge_mode_routes_to_extract(self) -> None:
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="nkge"
            ),
        }
        assert StoryPipeline._route_after_generate(state) == "extract"

    def test_baseline_mode_routes_to_end(self) -> None:
        state: PipelineState = {
            "session": SessionState(
                session_id="test", story_seed="test", mode="baseline"
            ),
        }
        assert StoryPipeline._route_after_generate(state) == "end"


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
