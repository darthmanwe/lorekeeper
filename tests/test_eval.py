"""Unit tests for eval.py — LLM judge parsing, metrics, comparison."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.eval import (
    SEVERITY_WEIGHTS,
    EvalRunResult,
    _extract_entity_mentions,
    _extract_keywords,
    _parse_json_safe,
    _parse_judge_response,
    compute_improvement,
    compute_retrieval_precision,
    format_comparison_table,
)
from src.schema import EvalRunOutput, EvalRunSummary, SegmentEvalRecord


# ---------------------------------------------------------------------------
# _parse_judge_response
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_valid_array(self) -> None:
        content = json.dumps([{
            "contradiction_text": "Kael smiled",
            "conflicting_fact": "Kael is dead",
            "severity": "critical",
            "reasoning": "Dead characters cannot smile",
        }])
        result = _parse_judge_response(content)
        assert len(result) == 1
        assert result[0]["severity"] == "critical"

    def test_empty_array(self) -> None:
        result = _parse_judge_response("[]")
        assert result == []

    def test_markdown_fenced(self) -> None:
        content = '```json\n[{"contradiction_text": "x", "conflicting_fact": "y", "severity": "minor", "reasoning": "z"}]\n```'
        result = _parse_judge_response(content)
        assert len(result) == 1

    def test_single_object_wrapped_in_array(self) -> None:
        content = '{"contradiction_text": "x", "conflicting_fact": "y", "severity": "major", "reasoning": "z"}'
        result = _parse_judge_response(content)
        assert len(result) == 1
        assert result[0]["severity"] == "major"

    def test_preamble_before_json(self) -> None:
        content = 'I found the following contradictions:\n[{"contradiction_text": "x", "conflicting_fact": "y", "severity": "soft", "reasoning": "z"}]'
        result = _parse_judge_response(content)
        assert len(result) == 1

    def test_malformed_returns_empty(self) -> None:
        result = _parse_judge_response("This is not JSON at all.")
        assert result == []

    def test_trailing_comma_handled(self) -> None:
        content = '[{"contradiction_text": "x", "conflicting_fact": "y", "severity": "minor", "reasoning": "z",}]'
        result = _parse_judge_response(content)
        assert len(result) == 1

    def test_truncated_json_recovered(self) -> None:
        content = '[{"contradiction_text": "x", "conflicting_fact": "y", "severity": "minor", "reasoning": "z"}, {"contradiction_text": "a", "conflicting_fact": "b", "severity": "major", "reason'
        result = _parse_judge_response(content)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# _parse_json_safe
# ---------------------------------------------------------------------------


class TestParseJsonSafe:
    def test_valid_json(self) -> None:
        result = _parse_json_safe('{"score": 4, "reasoning": "good flow"}')
        assert result["score"] == 4

    def test_markdown_fenced(self) -> None:
        result = _parse_json_safe('```json\n{"score": 3, "reasoning": "ok"}\n```')
        assert result["score"] == 3

    def test_preamble_before_object(self) -> None:
        result = _parse_json_safe('Here is my assessment: {"score": 5, "reasoning": "excellent"}')
        assert result["score"] == 5

    def test_invalid_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_json_safe("not json")


# ---------------------------------------------------------------------------
# _extract_entity_mentions
# ---------------------------------------------------------------------------


class TestExtractEntityMentions:
    def test_finds_known_entities(self) -> None:
        text = "Kael walked into the Iron Tavern and saw Maren."
        known = {"Kael", "Maren", "Elara", "Iron Tavern"}
        found = _extract_entity_mentions(text, known)
        assert found == {"Kael", "Maren", "Iron Tavern"}

    def test_case_insensitive(self) -> None:
        text = "KAEL shouted at MAREN"
        known = {"Kael", "Maren"}
        found = _extract_entity_mentions(text, known)
        assert found == {"Kael", "Maren"}

    def test_no_matches(self) -> None:
        text = "The wind howled through the empty streets."
        known = {"Kael", "Maren"}
        assert _extract_entity_mentions(text, known) == set()

    def test_empty_known(self) -> None:
        assert _extract_entity_mentions("any text", set()) == set()


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_strips_stop_words(self) -> None:
        keywords = _extract_keywords("Character 'Kael' is alive and located at Iron Tavern")
        assert "Kael" in keywords
        assert "alive" in keywords
        assert "Iron" in keywords
        assert "Tavern" in keywords
        assert "is" not in keywords
        assert "at" not in keywords

    def test_short_words_excluded(self) -> None:
        keywords = _extract_keywords("a of to")
        assert keywords == []


# ---------------------------------------------------------------------------
# compute_retrieval_precision
# ---------------------------------------------------------------------------


class TestRetrievalPrecision:
    def test_full_precision(self) -> None:
        graph_context = "=== Characters ===\n- Kael is alive\n- Maren is alive"
        generated = "Kael and Maren stood in the tavern."
        precision = compute_retrieval_precision(graph_context, generated)
        assert precision == 1.0

    def test_partial_precision(self) -> None:
        graph_context = "=== Characters ===\n- Kael is alive\n- Elara is dead"
        generated = "Kael entered the room alone."
        precision = compute_retrieval_precision(graph_context, generated)
        assert 0.0 < precision < 1.0

    def test_no_context_returns_zero(self) -> None:
        assert compute_retrieval_precision("", "any text") == 0.0
        assert compute_retrieval_precision("(no graph context)", "any text") == 0.0

    def test_no_text_returns_zero(self) -> None:
        precision = compute_retrieval_precision("Kael is alive", "")
        assert precision == 0.0


# ---------------------------------------------------------------------------
# Severity weights
# ---------------------------------------------------------------------------


class TestSeverityWeights:
    def test_weights_match_design_doc(self) -> None:
        assert SEVERITY_WEIGHTS["critical"] == 3.0
        assert SEVERITY_WEIGHTS["major"] == 2.0
        assert SEVERITY_WEIGHTS["minor"] == 1.0
        assert SEVERITY_WEIGHTS["soft"] == 0.5


# ---------------------------------------------------------------------------
# compute_improvement / format_comparison_table
# ---------------------------------------------------------------------------


class TestComparison:
    def _make_result(
        self,
        contradiction: float = 0.0,
        coherence: float = 3.0,
        coverage: float = 0.5,
        precision: float = 0.5,
        critical: int = 0,
        major: int = 0,
        minor: int = 0,
        soft: int = 0,
    ) -> EvalRunResult:
        summary = EvalRunSummary(
            mean_contradiction_score=contradiction,
            mean_coherence_score=coherence,
            mean_graph_coverage=coverage,
            mean_retrieval_precision=precision,
            critical_contradictions_total=critical,
            major_contradictions_total=major,
            minor_contradictions_total=minor,
            soft_contradictions_total=soft,
        )
        output = EvalRunOutput(
            run_id="test",
            mode="nkge",
            story_seed="test",
            summary=summary,
        )
        return EvalRunResult(output=output, output_path="test")

    def test_improvement_calculation(self) -> None:
        nkge = self._make_result(contradiction=1.0, critical=0, minor=1)
        baseline = self._make_result(contradiction=4.0, critical=2, minor=2)

        result = compute_improvement(nkge, baseline)
        assert result["improvement_pct"] == 75.0
        assert result["nkge_contradiction_score"] == 1.0
        assert result["baseline_contradiction_score"] == 4.0

    def test_improvement_with_zero_baseline(self) -> None:
        nkge = self._make_result(contradiction=0.0)
        baseline = self._make_result(contradiction=0.0)

        result = compute_improvement(nkge, baseline)
        assert result["improvement_pct"] == 0.0

    def test_nkge_worse_than_baseline(self) -> None:
        nkge = self._make_result(contradiction=5.0)
        baseline = self._make_result(contradiction=2.0)

        result = compute_improvement(nkge, baseline)
        assert result["improvement_pct"] < 0

    def test_format_table_contains_all_metrics(self) -> None:
        nkge = self._make_result(contradiction=1.0, coherence=4.0)
        baseline = self._make_result(contradiction=3.0, coherence=3.0)

        comparison = compute_improvement(nkge, baseline)
        table = format_comparison_table(comparison)

        assert "Contradiction Score" in table
        assert "Coherence" in table
        assert "Graph Coverage" in table
        assert "NKGE" in table
        assert "Baseline" in table

    def test_severity_breakdown_in_comparison(self) -> None:
        nkge = self._make_result(critical=1, major=2, minor=0, soft=3)
        baseline = self._make_result(critical=5, major=3, minor=2, soft=1)

        result = compute_improvement(nkge, baseline)
        assert result["severity_breakdown"]["nkge"]["critical"] == 1
        assert result["severity_breakdown"]["baseline"]["critical"] == 5


# ---------------------------------------------------------------------------
# EvalRunResult
# ---------------------------------------------------------------------------


class TestEvalRunResult:
    def test_headline_score(self) -> None:
        summary = EvalRunSummary(mean_contradiction_score=2.5)
        output = EvalRunOutput(
            run_id="test", mode="nkge", story_seed="test", summary=summary,
        )
        result = EvalRunResult(output=output, output_path="test")
        assert result.headline_score() == 2.5

    def test_worst_segments(self) -> None:
        segments = [
            SegmentEvalRecord(seq_id=1, player_action="a", generated_text="t", contradiction_score=0.5),
            SegmentEvalRecord(seq_id=2, player_action="b", generated_text="t", contradiction_score=3.0),
            SegmentEvalRecord(seq_id=3, player_action="c", generated_text="t", contradiction_score=1.0),
        ]
        output = EvalRunOutput(
            run_id="test", mode="nkge", story_seed="test", segments=segments,
        )
        result = EvalRunResult(output=output, output_path="test")
        worst = result.worst_segments(2)
        assert len(worst) == 2
        assert worst[0].seq_id == 2
        assert worst[1].seq_id == 3
