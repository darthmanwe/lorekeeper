"""Unit tests for the extraction pipeline.

Tests cover:
1. Happy path: proposals with valid confidence are approved
2. Confidence below threshold: proposal is flagged, not committed
3. Name collision: fuzzy match resolves near-duplicate to existing name
4. Status consistency: dead character set to alive is rejected
5. Empty proposals: returns empty result
6. Relationship directionality: CAUSED_BY requires both events to exist
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.extraction import (
    CONFIDENCE_THRESHOLD,
    ExtractionPipeline,
    NameResolver,
    StatusValidator,
)
from src.schema import (
    ConstraintViolation,
    ExtractionProposal,
    ExtractionResult,
)


# ---------------------------------------------------------------------------
# NameResolver tests
# ---------------------------------------------------------------------------


class TestNameResolver:
    def test_fuzzy_match_finds_close_name(self) -> None:
        existing = ["Elara", "Kael", "Maren"]
        result = NameResolver.fuzzy_match("Ellara", existing, threshold=80)
        assert result == "Elara"

    def test_fuzzy_match_returns_none_for_exact(self) -> None:
        existing = ["Elara", "Kael"]
        result = NameResolver.fuzzy_match("Elara", existing)
        assert result is None

    def test_fuzzy_match_returns_none_below_threshold(self) -> None:
        existing = ["Elara", "Kael"]
        result = NameResolver.fuzzy_match("Xyzzy", existing)
        assert result is None

    def test_fuzzy_match_empty_existing(self) -> None:
        result = NameResolver.fuzzy_match("Elara", [])
        assert result is None


# ---------------------------------------------------------------------------
# StatusValidator tests
# ---------------------------------------------------------------------------


class TestStatusValidator:
    def _make_mock_gc(self, dead: bool) -> MagicMock:
        gc = MagicMock()
        session = MagicMock()
        result = MagicMock()
        record = {"name": "Kael"} if dead else None
        result.single.return_value = record
        session.run.return_value = result
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        gc._driver.session.return_value = session
        gc._database = "neo4j"
        return gc

    def test_dead_character_set_alive_is_rejected(self) -> None:
        gc = self._make_mock_gc(dead=True)
        proposal = ExtractionProposal(
            entity_type="Character",
            entity_name="Kael",
            action="update",
            confidence=0.9,
            properties={"status": "alive"},
        )
        violation = StatusValidator.check_status_consistency(proposal, gc, "main")
        assert violation is not None
        assert violation.severity == "critical"
        assert "dead" in violation.violation_message.lower()

    def test_alive_character_stays_alive_ok(self) -> None:
        gc = self._make_mock_gc(dead=False)
        proposal = ExtractionProposal(
            entity_type="Character",
            entity_name="Kael",
            action="update",
            confidence=0.9,
            properties={"status": "alive"},
        )
        violation = StatusValidator.check_status_consistency(proposal, gc, "main")
        assert violation is None

    def test_non_character_is_always_ok(self) -> None:
        gc = self._make_mock_gc(dead=True)
        proposal = ExtractionProposal(
            entity_type="Location",
            entity_name="Iron Tavern",
            confidence=0.9,
            properties={"type": "tavern"},
        )
        violation = StatusValidator.check_status_consistency(proposal, gc, "main")
        assert violation is None

    def test_character_set_to_dead_is_ok(self) -> None:
        gc = self._make_mock_gc(dead=False)
        proposal = ExtractionProposal(
            entity_type="Character",
            entity_name="Kael",
            action="update",
            confidence=0.9,
            properties={"status": "dead"},
        )
        violation = StatusValidator.check_status_consistency(proposal, gc, "main")
        assert violation is None


# ---------------------------------------------------------------------------
# ExtractionPipeline.validate() tests
# ---------------------------------------------------------------------------


class TestValidation:
    def _make_pipeline(self, existing_names: list[str] | None = None, dead_chars: list[str] | None = None) -> ExtractionPipeline:
        llm = MagicMock()
        gc = MagicMock()
        gc.get_all_entity_names.return_value = existing_names or []
        gc._database = "neo4j"

        dead_chars = dead_chars or []

        session = MagicMock()
        result = MagicMock()

        def run_side_effect(query, params):
            mock_result = MagicMock()
            name = params.get("name", "")
            if name in dead_chars:
                mock_result.single.return_value = {"name": name}
            else:
                mock_result.single.return_value = None
            return mock_result

        session.run.side_effect = run_side_effect
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        gc._driver.session.return_value = session

        return ExtractionPipeline(llm, gc)

    def test_happy_path_approved(self) -> None:
        pipeline = self._make_pipeline(existing_names=["Kael", "Elara"])
        proposals = [
            ExtractionProposal(
                entity_type="Character",
                entity_name="Maren",
                confidence=0.85,
                properties={"status": "alive"},
            ),
            ExtractionProposal(
                entity_type="Location",
                entity_name="Dark Forest",
                confidence=0.90,
                properties={"type": "forest"},
            ),
        ]
        approved, flagged = pipeline.validate(proposals, "main")
        assert len(approved) == 2
        assert len(flagged) == 0

    def test_low_confidence_flagged(self) -> None:
        pipeline = self._make_pipeline()
        proposals = [
            ExtractionProposal(
                entity_type="Character",
                entity_name="Mysterious Figure",
                confidence=0.40,
                properties={"status": "unknown"},
            ),
        ]
        approved, flagged = pipeline.validate(proposals, "main")
        assert len(approved) == 0
        assert len(flagged) == 1

    def test_fuzzy_name_collision_resolves(self) -> None:
        pipeline = self._make_pipeline(existing_names=["Maren", "Kael"])
        proposals = [
            ExtractionProposal(
                entity_type="Character",
                entity_name="Maaren",
                confidence=0.90,
                properties={"status": "alive"},
            ),
        ]
        approved, flagged = pipeline.validate(proposals, "main")
        assert len(approved) == 1
        assert approved[0].entity_name == "Maren"
        assert approved[0].action == "update"

    def test_dead_character_alive_rejected(self) -> None:
        pipeline = self._make_pipeline(
            existing_names=["Kael"],
            dead_chars=["Kael"],
        )
        proposals = [
            ExtractionProposal(
                entity_type="Character",
                entity_name="Kael",
                confidence=0.95,
                properties={"status": "alive"},
            ),
        ]
        approved, flagged = pipeline.validate(proposals, "main")
        assert len(approved) == 0
        assert len(flagged) == 1

    def test_empty_proposals(self) -> None:
        pipeline = self._make_pipeline()
        approved, flagged = pipeline.validate([], "main")
        assert len(approved) == 0
        assert len(flagged) == 0

    def test_caused_by_missing_events_flagged(self) -> None:
        pipeline = self._make_pipeline()

        events_query_result = MagicMock()
        events_query_result.single.return_value = {
            "src_exists": False,
            "tgt_exists": False,
        }
        session = MagicMock()
        session.run.return_value = events_query_result
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        pipeline._gc._driver.session.return_value = session

        proposals = [
            ExtractionProposal(
                entity_type="Relationship",
                entity_name="causal_link",
                confidence=0.90,
                properties={
                    "rel_type": "CAUSED_BY",
                    "source": "nonexistent_event_1",
                    "target": "nonexistent_event_2",
                },
            ),
        ]
        approved, flagged = pipeline.validate(proposals, "main")
        assert len(approved) == 0
        assert len(flagged) == 1


# ---------------------------------------------------------------------------
# ExtractionPipeline._parse_proposals() tests
# ---------------------------------------------------------------------------


class TestParseProposals:
    def _make_pipeline(self) -> ExtractionPipeline:
        return ExtractionPipeline(MagicMock(), MagicMock())

    def test_parse_valid_json_array(self) -> None:
        pipeline = self._make_pipeline()
        content = '[{"entity_type": "Character", "entity_name": "Kael", "confidence": 0.9, "properties": {}}]'
        proposals = pipeline._parse_proposals(content)
        assert len(proposals) == 1
        assert proposals[0].entity_name == "Kael"

    def test_parse_markdown_fenced_json(self) -> None:
        pipeline = self._make_pipeline()
        content = '```json\n[{"entity_type": "Location", "entity_name": "Cave", "confidence": 0.8, "properties": {}}]\n```'
        proposals = pipeline._parse_proposals(content)
        assert len(proposals) == 1
        assert proposals[0].entity_name == "Cave"

    def test_parse_empty_array(self) -> None:
        pipeline = self._make_pipeline()
        proposals = pipeline._parse_proposals("[]")
        assert len(proposals) == 0

    def test_parse_malformed_item_skipped(self) -> None:
        pipeline = self._make_pipeline()
        content = '[{"entity_type": "Character", "entity_name": "Kael", "confidence": 0.9, "properties": {}}, {"bad": "item"}]'
        proposals = pipeline._parse_proposals(content)
        assert len(proposals) == 1

    def test_parse_truncated_json_repaired(self) -> None:
        """LLM output truncated mid-string — parser should recover partial items."""
        pipeline = self._make_pipeline()
        content = (
            '[{"entity_type": "Character", "entity_name": "Kael", '
            '"confidence": 0.9, "properties": {}}, '
            '{"entity_type": "Location", "entity_name": "Dark For'
        )
        proposals = pipeline._parse_proposals(content)
        assert len(proposals) >= 1
        assert proposals[0].entity_name == "Kael"

    def test_parse_trailing_comma_handled(self) -> None:
        pipeline = self._make_pipeline()
        content = '[{"entity_type": "Character", "entity_name": "Aria", "confidence": 0.85, "properties": {}},]'
        proposals = pipeline._parse_proposals(content)
        assert len(proposals) == 1

    def test_parse_no_json_returns_empty(self) -> None:
        pipeline = self._make_pipeline()
        content = "I could not find any entities in this text."
        proposals = pipeline._parse_proposals(content)
        assert len(proposals) == 0

    def test_parse_json_with_preamble(self) -> None:
        """LLM wraps JSON in explanatory text — parser should extract the array."""
        pipeline = self._make_pipeline()
        content = (
            'Here are the extracted entities:\n\n'
            '[{"entity_type": "Character", "entity_name": "Brann", '
            '"confidence": 0.92, "properties": {"status": "alive"}}]'
        )
        proposals = pipeline._parse_proposals(content)
        assert len(proposals) == 1
        assert proposals[0].entity_name == "Brann"
