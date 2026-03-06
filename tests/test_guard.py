"""Unit tests for guard.py — ContradictionGuard checks and BranchManager logic.

Tests use mocked Neo4j sessions to avoid requiring a running database.
Each guard check is tested for both positive (violation found) and
negative (no violation) cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.guard import BranchManager, ContradictionGuard
from src.schema import ConstraintViolation, SessionState


def _mock_gc_with_results(results: list[dict]) -> MagicMock:
    """Create a mock GraphClient whose session.run() returns the given records."""
    gc = MagicMock()
    session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter(results))
    mock_result.single.return_value = results[0] if results else None
    session.run.return_value = mock_result
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    gc._driver.session.return_value = session
    gc._database = "neo4j"
    return gc


def _mock_gc_with_multi_query(query_results: list[list[dict]]) -> MagicMock:
    """Mock GraphClient where each successive session.run() returns a different result set."""
    gc = MagicMock()
    session = MagicMock()

    results_iter = iter(query_results)

    def run_side_effect(*args, **kwargs):
        try:
            data = next(results_iter)
        except StopIteration:
            data = []
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter(data))
        mock_result.single.return_value = data[0] if data else None
        return mock_result

    session.run.side_effect = run_side_effect
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    gc._driver.session.return_value = session
    gc._database = "neo4j"
    return gc


# ---------------------------------------------------------------------------
# Dead Character Active
# ---------------------------------------------------------------------------


class TestDeadCharacterActive:
    def test_dead_character_detected(self) -> None:
        gc = _mock_gc_with_results([{"name": "Brann", "last_event": 3}])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_dead_character_active(["Brann", "Aria"], "main")
        assert len(violations) == 1
        assert violations[0].severity == "critical"
        assert "Brann" in violations[0].violation_message
        assert "dead" in violations[0].violation_message

    def test_no_dead_characters(self) -> None:
        gc = _mock_gc_with_results([])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_dead_character_active(["Aria", "Kael"], "main")
        assert len(violations) == 0

    def test_empty_present_chars(self) -> None:
        gc = MagicMock()
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_dead_character_active([], "main")
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Location Inaccessible
# ---------------------------------------------------------------------------


class TestLocationInaccessible:
    def test_inaccessible_location_detected(self) -> None:
        gc = _mock_gc_with_results([{
            "name": "Sealed Chamber",
            "description": "collapsed entrance",
        }])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_location_inaccessible("Sealed Chamber")
        assert len(violations) == 1
        assert violations[0].severity == "major"
        assert "inaccessible" in violations[0].violation_message

    def test_accessible_location_ok(self) -> None:
        gc = _mock_gc_with_results([])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_location_inaccessible("Iron Tavern")
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Hostile Co-presence
# ---------------------------------------------------------------------------


class TestHostileCopresence:
    def test_hostile_pair_detected(self) -> None:
        gc = _mock_gc_with_results([{
            "a_name": "Aria",
            "b_name": "Maren",
            "a_last_event": 5,
        }])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_hostile_copresence(["Aria", "Maren"], "main")
        assert len(violations) == 1
        assert violations[0].severity == "minor"
        assert "hostile" in violations[0].violation_message

    def test_no_hostile_pairs(self) -> None:
        gc = _mock_gc_with_results([])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_hostile_copresence(["Aria", "Kael"], "main")
        assert len(violations) == 0

    def test_single_character_skips_check(self) -> None:
        gc = MagicMock()
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_hostile_copresence(["Aria"], "main")
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Object Ownership
# ---------------------------------------------------------------------------


class TestObjectOwnership:
    def test_multi_owner_conflict_detected(self) -> None:
        gc = _mock_gc_with_multi_query([
            [{"object_name": "Crown", "owners": ["Aria", "Kael"]}],
            [],
        ])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_object_ownership(["Aria", "Kael"], "main")
        assert len(violations) == 1
        assert violations[0].severity == "major"
        assert "Crown" in violations[0].violation_message

    def test_dead_owner_flagged(self) -> None:
        gc = _mock_gc_with_multi_query([
            [],
            [{"dead_owner": "Brann", "object_name": "Sword"}],
        ])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_object_ownership(["Aria"], "main")
        assert len(violations) == 1
        assert violations[0].severity == "soft"
        assert "dead" in violations[0].violation_message.lower()

    def test_clean_ownership(self) -> None:
        gc = _mock_gc_with_multi_query([[], []])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_object_ownership(["Aria"], "main")
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Knowledge Boundary
# ---------------------------------------------------------------------------


class TestKnowledgeBoundary:
    def test_unknown_event_detected(self) -> None:
        gc = _mock_gc_with_results([{
            "seq_id": 4,
            "description": "Secret meeting in the forest",
        }])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_knowledge_boundary(["Aria"], "main")
        assert len(violations) == 1
        assert violations[0].severity == "soft"
        assert "Aria" in violations[0].violation_message

    def test_no_knowledge_gaps(self) -> None:
        gc = _mock_gc_with_results([])
        guard = ContradictionGuard(gc, mode="permissive")
        violations = guard.check_knowledge_boundary(["Kael"], "main")
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    def test_aggregates_all_checks(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="permissive")
        guard.check_dead_character_active = MagicMock(return_value=[
            ConstraintViolation(
                check_name="dead_character_active",
                violation_message="Brann is dead",
                severity="critical",
            ),
        ])
        guard.check_location_inaccessible = MagicMock(return_value=[])
        guard.check_object_ownership = MagicMock(return_value=[])
        guard.check_hostile_copresence = MagicMock(return_value=[
            ConstraintViolation(
                check_name="hostile_copresence",
                violation_message="A and B hostile",
                severity="minor",
            ),
        ])
        guard.check_knowledge_boundary = MagicMock(return_value=[])

        session = SessionState(
            session_id="test",
            story_seed="test",
            current_location="Castle",
            present_characters=["Brann", "Aria"],
        )
        violations = guard.run_all_checks(session)
        assert len(violations) == 2
        assert violations[0].severity == "critical"
        assert violations[1].severity == "minor"


# ---------------------------------------------------------------------------
# has_blocking_violations
# ---------------------------------------------------------------------------


class TestHasBlockingViolations:
    def test_strict_mode_critical_is_blocking(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="strict")
        violations = [
            ConstraintViolation(
                check_name="test", violation_message="x", severity="critical"
            ),
        ]
        assert guard.has_blocking_violations(violations) is True

    def test_strict_mode_major_is_blocking(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="strict")
        violations = [
            ConstraintViolation(
                check_name="test", violation_message="x", severity="major"
            ),
        ]
        assert guard.has_blocking_violations(violations) is True

    def test_strict_mode_minor_is_not_blocking(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="strict")
        violations = [
            ConstraintViolation(
                check_name="test", violation_message="x", severity="minor"
            ),
        ]
        assert guard.has_blocking_violations(violations) is False

    def test_permissive_mode_never_blocking(self) -> None:
        guard = ContradictionGuard(MagicMock(), mode="permissive")
        violations = [
            ConstraintViolation(
                check_name="test", violation_message="x", severity="critical"
            ),
        ]
        assert guard.has_blocking_violations(violations) is False


# ---------------------------------------------------------------------------
# BranchManager
# ---------------------------------------------------------------------------


class TestBranchManager:
    def test_create_branch_returns_new_id(self) -> None:
        gc = _mock_gc_with_multi_query([
            [],  # MERGE for branch event
            [],  # get_active_branches query
        ])
        bm = BranchManager(gc, max_branches=5)
        session = SessionState(
            session_id="s1",
            story_seed="test",
            active_branch_id="main",
            last_segment_seq_id=5,
        )
        new_id = bm.create_branch(session, "test reason")
        assert new_id == "main_b5"
        assert session.active_branch_id == "main_b5"
        assert "main_b5" in session.branch_ancestry

    def test_is_branch_archived_false_when_not_found(self) -> None:
        gc = _mock_gc_with_results([])
        bm = BranchManager(gc)
        assert bm.is_branch_archived("nonexistent") is False

    def test_is_branch_archived_true(self) -> None:
        gc = _mock_gc_with_results([{"archived": True}])
        bm = BranchManager(gc)
        assert bm.is_branch_archived("old_branch") is True
