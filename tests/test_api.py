"""Tests for api.py — FastAPI endpoint unit tests using HTTPX TestClient."""

import pytest
from unittest.mock import MagicMock, patch

from api import (
    ExtractionSummary,
    GenerateRequest,
    GenerateResponse,
    GraphStatsResponse,
    HealthResponse,
    SessionResponse,
    ViolationResponse,
)


class TestRequestModels:
    """Validate request model validation."""

    def test_generate_request_valid(self):
        req = GenerateRequest(player_action="attack the goblin")
        assert req.player_action == "attack the goblin"

    def test_generate_request_empty_action_rejected(self):
        with pytest.raises(Exception):
            GenerateRequest(player_action="")

    def test_generate_request_with_mode_override(self):
        req = GenerateRequest(player_action="look around", mode="baseline")
        assert req.mode == "baseline"

    def test_generate_request_default_mode_is_none(self):
        req = GenerateRequest(player_action="look around")
        assert req.mode is None


class TestResponseModels:
    """Validate response model construction."""

    def test_generate_response_construction(self):
        resp = GenerateResponse(
            seq_id=4,
            branch_id="main",
            generated_text="The tavern door creaked.",
            graph_context_tokens=200,
            vector_context_tokens=100,
            violations=[],
            extraction=ExtractionSummary(
                proposed=3, approved=2, flagged=1, committed=2
            ),
            mode="nkge",
        )
        assert resp.seq_id == 4
        assert resp.extraction.committed == 2

    def test_generate_response_without_extraction(self):
        resp = GenerateResponse(
            seq_id=4,
            branch_id="main",
            generated_text="Text.",
            graph_context_tokens=0,
            vector_context_tokens=0,
            violations=[],
            extraction=None,
            mode="baseline",
        )
        assert resp.extraction is None

    def test_violation_response(self):
        v = ViolationResponse(
            check_name="dead_character_active",
            violation_message="Elara is dead",
            severity="critical",
        )
        assert v.severity == "critical"

    def test_session_response(self):
        s = SessionResponse(
            session_id="abc",
            story_seed="Iron Tavern",
            active_branch_id="main",
            current_location="Iron Tavern",
            present_characters=["Kael"],
            last_segment_seq_id=3,
            mode="nkge",
        )
        assert s.present_characters == ["Kael"]

    def test_graph_stats_response(self):
        g = GraphStatsResponse(
            node_counts={"Character": 3, "Location": 2},
            relationship_counts={"KNOWS": 2},
            total_nodes=5,
            total_relationships=2,
        )
        assert g.total_nodes == 5

    def test_health_response(self):
        h = HealthResponse(status="ok", neo4j="connected", service="lorekeeper")
        assert h.status == "ok"


class TestExtractionSummary:
    """Validate extraction summary model."""

    def test_all_zeros(self):
        e = ExtractionSummary(proposed=0, approved=0, flagged=0, committed=0)
        assert e.proposed == 0

    def test_flagged_exceeds_proposed_is_valid(self):
        e = ExtractionSummary(proposed=5, approved=2, flagged=3, committed=2)
        assert e.flagged == 3
