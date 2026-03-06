"""Pydantic v2 models for the Lorekeeper narrative knowledge graph ontology.

Defines all node types, relationship types, session state, extraction proposals,
constraint violations, contradiction results, and the evaluation run output schema.
All models use Pydantic v2 syntax exclusively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------


class Character(BaseModel):
    """A named character in the story graph."""

    name: str
    status: Literal["alive", "dead", "unknown"] = "alive"
    current_location_id: str | None = None
    alignment: str | None = None
    traits: list[str] = Field(default_factory=list)
    persona_doc_id: str | None = None

    @field_validator("name")
    @classmethod
    def name_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Character name must not be empty")
        return v.strip()


class Location(BaseModel):
    """A named location in the story world."""

    name: str
    type: str = "generic"
    accessible: bool = True
    description_summary: str = ""

    @field_validator("name")
    @classmethod
    def name_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Location name must not be empty")
        return v.strip()


class Event(BaseModel):
    """A discrete story event anchored to a sequence position and branch."""

    description: str
    seq_id: int
    branch_id: str
    outcome: str | None = None
    timestamp: str | None = None

    @field_validator("seq_id")
    @classmethod
    def seq_id_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seq_id must be non-negative")
        return v


class StoryObject(BaseModel):
    """A trackable object with ownership provenance.

    Named StoryObject to avoid shadowing Python's built-in object.
    Maps to the Neo4j :Object label.
    """

    name: str
    current_owner_id: str | None = None
    significance: str = ""
    last_seen_location_id: str | None = None


class Faction(BaseModel):
    """A group of characters with shared goals."""

    name: str
    goals: list[str] = Field(default_factory=list)
    member_ids: list[str] = Field(default_factory=list)


class Segment(BaseModel):
    """A generated story segment linked to graph state."""

    text: str
    seq_id: int
    branch_id: str
    embedding_id: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Segment text must not be empty")
        return v


# ---------------------------------------------------------------------------
# Relationship model
# ---------------------------------------------------------------------------


class Relationship(BaseModel):
    """A typed, directed relationship between two named entities."""

    type: str
    source: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: str) -> str:
        valid_types = {
            "KNOWS",
            "LOCATED_AT",
            "PARTICIPATED_IN",
            "CAUSED_BY",
            "OWNS",
            "VISITED",
            "MEMBER_OF",
            "REFERENCES_GRAPH_STATE",
        }
        if v not in valid_types:
            raise ValueError(f"Relationship type must be one of {valid_types}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """Mutable session state for a single story playthrough.

    Tracks active branch, location, present characters, and last generated
    segment. Used by the pipeline, retrieval, and eval systems.
    """

    session_id: str
    story_seed: str
    active_branch_id: str = "main"
    branch_ancestry: list[str] = field(default_factory=lambda: ["main"])
    current_location: str = ""
    present_characters: list[str] = field(default_factory=list)
    last_segment_seq_id: int = 0
    last_segment_text: str = ""
    mode: Literal["nkge", "baseline"] = "nkge"


# ---------------------------------------------------------------------------
# Extraction pipeline models
# ---------------------------------------------------------------------------


class ExtractionProposal(BaseModel):
    """A single entity or relationship proposed by the extraction LLM."""

    entity_type: str
    entity_name: str
    action: Literal["create", "update"] = "create"
    confidence: float
    supporting_quote: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

    @field_validator("entity_type")
    @classmethod
    def entity_type_must_be_valid(cls, v: str) -> str:
        valid_types = {
            "Character",
            "Location",
            "Event",
            "Object",
            "Faction",
            "Relationship",
        }
        if v not in valid_types:
            raise ValueError(f"entity_type must be one of {valid_types}, got '{v}'")
        return v


class ExtractionResult(BaseModel):
    """Aggregate result of a full extraction pipeline run."""

    proposals: list[ExtractionProposal] = Field(default_factory=list)
    approved: list[ExtractionProposal] = Field(default_factory=list)
    flagged: list[ExtractionProposal] = Field(default_factory=list)
    committed_count: int = 0


# ---------------------------------------------------------------------------
# Contradiction guard models
# ---------------------------------------------------------------------------


class ConstraintViolation(BaseModel):
    """A guard-detected constraint violation to inject into the generation prompt."""

    check_name: str
    violation_message: str
    severity: Literal["critical", "major", "minor", "soft"]


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------


class ContradictionResult(BaseModel):
    """A single contradiction found by the LLM judge."""

    contradiction_text: str
    conflicting_fact: str
    severity: Literal["critical", "major", "minor", "soft"]
    reasoning: str
    weighted_score: float = 0.0

    SEVERITY_WEIGHTS: dict[str, float] = Field(
        default={"critical": 3.0, "major": 2.0, "minor": 1.0, "soft": 0.5},
        exclude=True,
    )

    @model_validator(mode="after")
    def compute_weighted_score(self) -> ContradictionResult:
        self.weighted_score = self.SEVERITY_WEIGHTS.get(self.severity, 1.0)
        return self


class SegmentEvalRecord(BaseModel):
    """Evaluation record for a single generated segment (Appendix B)."""

    seq_id: int
    player_action: str
    generated_text: str
    graph_context_tokens: int = 0
    vector_context_tokens: int = 0
    guard_violations: list[ConstraintViolation] = Field(default_factory=list)
    extraction_proposals: list[ExtractionProposal] = Field(default_factory=list)
    contradictions_found: list[ContradictionResult] = Field(default_factory=list)
    contradiction_score: float = 0.0
    coherence_score: float = 0.0
    graph_coverage_rate: float = 0.0
    retrieval_precision: float = 0.0


class EvalRunSummary(BaseModel):
    """Aggregate summary metrics for an evaluation run (Appendix B)."""

    mean_contradiction_score: float = 0.0
    mean_coherence_score: float = 0.0
    mean_graph_coverage: float = 0.0
    mean_retrieval_precision: float = 0.0
    total_nodes_created: int = 0
    total_relationships_created: int = 0
    critical_contradictions_total: int = 0
    major_contradictions_total: int = 0
    minor_contradictions_total: int = 0
    soft_contradictions_total: int = 0


class EvalRunOutput(BaseModel):
    """Complete evaluation run output matching Appendix B schema."""

    run_id: str
    mode: Literal["nkge", "baseline"]
    story_seed: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    segments: list[SegmentEvalRecord] = Field(default_factory=list)
    summary: EvalRunSummary = Field(default_factory=EvalRunSummary)
