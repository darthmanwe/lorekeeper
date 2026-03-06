"""Unit tests for retrieval.py — token counting, tier formatting, context assembly."""

from __future__ import annotations

import pytest

from src.retrieval import ContextAssembler, _text_overlap_ratio, count_tokens
from src.schema import ConstraintViolation, SessionState


class TestCountTokens:
    def test_empty_string_returns_zero(self) -> None:
        assert count_tokens("") == 0

    def test_nonempty_string_returns_positive(self) -> None:
        assert count_tokens("Hello, world!") > 0

    def test_longer_text_has_more_tokens(self) -> None:
        short = count_tokens("hello")
        long = count_tokens("hello world this is a much longer sentence with many words")
        assert long > short

    def test_none_coercion(self) -> None:
        assert count_tokens("") == 0


class TestContextAssembler:
    @pytest.fixture()
    def session(self) -> SessionState:
        return SessionState(
            session_id="test-sess",
            story_seed="test",
            current_location="Castle",
            present_characters=["Aria", "Kael"],
            last_segment_text="The sun set over the castle walls.",
            last_segment_seq_id=3,
        )

    def test_assemble_with_all_sections(self, session: SessionState) -> None:
        result = ContextAssembler.assemble(
            graph_context="## Active Scene: Castle\n- Aria (alive)",
            vector_context="[Tonal anchor 1]: The wind howled.",
            violations=[
                ConstraintViolation(
                    check_name="dead_character_active",
                    violation_message="Brann is dead",
                    severity="critical",
                )
            ],
            persona_docs=["Aria speaks in clipped sentences."],
            session=session,
        )
        assert "Active Scene" in result["known_facts"]
        assert "Tonal anchor" in result["tonal_context"]
        assert "CRITICAL" in result["constraints"]
        assert "Aria speaks" in result["character_voices"]
        assert "sun set" in result["previous_segment"]

    def test_assemble_with_no_context(self) -> None:
        result = ContextAssembler.assemble(
            graph_context="",
            vector_context="",
        )
        assert "no graph context" in result["known_facts"]
        assert "no tonal anchors" in result["tonal_context"]
        assert "no constraint violations" in result["constraints"]
        assert "no persona documents" in result["character_voices"]
        assert "story beginning" in result["previous_segment"]

    def test_assemble_preserves_violation_severity_ordering(self) -> None:
        violations = [
            ConstraintViolation(
                check_name="dead_check", violation_message="X is dead", severity="critical"
            ),
            ConstraintViolation(
                check_name="location_check", violation_message="Y locked", severity="minor"
            ),
        ]
        result = ContextAssembler.assemble(
            graph_context="facts",
            vector_context="tone",
            violations=violations,
        )
        assert "CRITICAL" in result["constraints"]
        assert "MINOR" in result["constraints"]
        lines = result["constraints"].split("\n")
        assert len(lines) == 2

    def test_multiple_persona_docs_joined(self) -> None:
        result = ContextAssembler.assemble(
            graph_context="facts",
            vector_context="tone",
            persona_docs=["Voice A.", "Voice B."],
        )
        assert "Voice A" in result["character_voices"]
        assert "Voice B" in result["character_voices"]


class TestTextOverlapRatio:
    def test_identical_texts_returns_one(self) -> None:
        assert _text_overlap_ratio("hello world", "hello world") == 1.0

    def test_disjoint_texts_returns_zero(self) -> None:
        assert _text_overlap_ratio("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        ratio = _text_overlap_ratio("the quick brown fox", "the slow brown cat")
        assert 0.2 < ratio < 0.8

    def test_empty_text_returns_zero(self) -> None:
        assert _text_overlap_ratio("", "hello") == 0.0
        assert _text_overlap_ratio("hello", "") == 0.0
