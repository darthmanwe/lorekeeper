"""All LLM prompt templates with version tracking.

Every prompt used in the system is defined here and registered in
prompts_registry.json. No inline prompt strings elsewhere in the codebase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """A versioned prompt template with system and user sections."""

    version: str
    description: str
    system: str
    user: str

    def format_system(self, **kwargs: Any) -> str:
        """Interpolate the system prompt with provided context variables.

        Args:
            **kwargs: Template variables to substitute.

        Returns:
            Formatted system prompt string.
        """
        return self.system.format(**kwargs)

    def format_user(self, **kwargs: Any) -> str:
        """Interpolate the user prompt with provided context variables.

        Args:
            **kwargs: Template variables to substitute.

        Returns:
            Formatted user prompt string.
        """
        return self.user.format(**kwargs)


# ---------------------------------------------------------------------------
# Extraction prompt — used by ExtractionPipeline.propose()
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT_V1 = PromptTemplate(
    version="extraction_v1",
    description="Structured entity/relationship extraction from a generated story segment.",
    system="""\
You are a narrative knowledge graph extraction engine. You will be given a story \
segment and a list of entities already known to the graph.

Your task: identify ALL entities and relationships mentioned in the segment. For each, \
return a JSON object with the fields specified below.

ENTITY TYPES: Character, Location, Event, Object, Faction, Relationship

For each extraction, provide:
- entity_type: one of the types above
- entity_name: the canonical name (match existing names exactly when possible)
- action: "create" if this is a new entity, "update" if modifying an existing one
- confidence: 0.0-1.0 how certain you are this entity/relationship exists in the text
- supporting_quote: the exact phrase from the segment that supports this extraction
- properties: a dict of type-specific properties:
  - Character: status, current_location_id, alignment, traits (list)
  - Location: type, accessible (bool), description_summary
  - Event: description, outcome
  - Object: current_owner_id, significance, last_seen_location_id
  - Faction: goals (list), member_ids (list)
  - Relationship: rel_type (KNOWS/LOCATED_AT/PARTICIPATED_IN/CAUSED_BY/OWNS/VISITED/MEMBER_OF), \
source, target, and any relationship properties (sentiment, role, etc.)

EXISTING ENTITIES IN GRAPH:
{existing_entities}

Return a JSON array of extraction objects. If no new entities or changes are found, \
return an empty array [].
Do not invent entities not supported by the text. Only extract what is explicitly \
stated or directly implied.""",
    user="""\
STORY SEGMENT:
{segment_text}

Extract all entities and relationships from this segment as a JSON array.""",
)


# ---------------------------------------------------------------------------
# Reclassification prompt — used by ExtractionPipeline.reclassify()
# ---------------------------------------------------------------------------

RECLASSIFICATION_PROMPT_V1 = PromptTemplate(
    version="reclassification_v1",
    description="Schema reclassification pass for character trait/alignment/faction updates.",
    system="""\
You are a narrative analysis engine. You will be given the current state of all \
characters in a story graph. Your task is to identify any characters whose traits, \
alignment, or faction membership should be updated based on the story progression.

Return a JSON array of update objects. Each object must have:
- character_name: the name of the character to update
- updates: a dict of fields to change (traits, alignment, faction)
- reasoning: one sentence explaining why this update is warranted

Only propose updates that are clearly supported by the character's event participation \
and relationship changes. If no updates are needed, return an empty array [].""",
    user="""\
CURRENT CHARACTER STATE:
{character_state}

RECENT EVENTS (last 10):
{recent_events}

Analyze and propose any character reclassifications as a JSON array.""",
)


# ---------------------------------------------------------------------------
# Prompt registry helpers
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path(__file__).parent.parent / "prompts_registry.json"

_PROMPT_MAP: dict[str, PromptTemplate] = {
    "extraction_v1": EXTRACTION_PROMPT_V1,
    "reclassification_v1": RECLASSIFICATION_PROMPT_V1,
}


def get_prompt(prompt_id: str) -> PromptTemplate:
    """Look up a prompt template by its version ID.

    Args:
        prompt_id: The version string (e.g. "extraction_v1").

    Returns:
        The matching PromptTemplate instance.

    Raises:
        KeyError: If the prompt_id is not found.
    """
    if prompt_id not in _PROMPT_MAP:
        raise KeyError(
            f"Unknown prompt_id '{prompt_id}'. "
            f"Available: {list(_PROMPT_MAP.keys())}"
        )
    return _PROMPT_MAP[prompt_id]


def register_prompt(prompt: PromptTemplate) -> None:
    """Register a new prompt version in the in-memory map.

    Args:
        prompt: The PromptTemplate to register.
    """
    _PROMPT_MAP[prompt.version] = prompt


def list_prompts() -> list[str]:
    """Return all registered prompt version IDs.

    Returns:
        List of version ID strings.
    """
    return list(_PROMPT_MAP.keys())
