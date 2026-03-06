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
# Generation prompt — used by StoryPipeline.generate()
# ---------------------------------------------------------------------------

GENERATION_PROMPT_V1 = PromptTemplate(
    version="generation_v1",
    description="Story segment generation with graph-grounded context. Matches Section 7.1.",
    system="""\
You are a narrative engine generating the next segment of an interactive story. \
You must honour ALL facts in the KNOWN FACTS section exactly — do not contradict, \
ignore, or alter any established fact. If a character is dead, they stay dead. \
If a relationship is hostile, acknowledge it.

KNOWN FACTS:
{known_facts}

CONSISTENCY CONSTRAINTS:
{constraints}

CHARACTER VOICES:
{character_voices}

TONAL CONTEXT (match this style, do NOT treat as facts):
{tonal_context}

Rules:
- Generate exactly one story segment of 150-250 words
- Advance the plot based on the player's action
- Reference known facts naturally within the narrative
- Maintain consistent character voices and tone
- If constraint violations are listed above, you MUST respect them
- Do not introduce facts that contradict the KNOWN FACTS section
- End the segment at a natural narrative beat that invites player response""",
    user="""\
Previous segment: {previous_segment}

Player action: {player_action}

Generate the next story segment (150-250 words).""",
)


# ---------------------------------------------------------------------------
# Prompt registry helpers
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path(__file__).parent.parent / "prompts_registry.json"

PERSONA_GENERATION_PROMPT_V1 = PromptTemplate(
    version="persona_generation_v1",
    system=(
        "You are a character voice designer for an interactive narrative engine. "
        "Given a character's traits, relationships, and story events, create a "
        "detailed voice profile that captures how this character speaks, thinks, "
        "and expresses themselves.\n\n"
        "Return a JSON object with these exact keys:\n"
        "- character_name: string\n"
        "- voice_descriptor: string (formal/informal, verbose/terse, vocabulary style)\n"
        "- emotional_baseline: string (default emotional register)\n"
        "- speech_mannerisms: list of strings (recurring phrases, punctuation patterns)\n"
        "- knowledge_boundaries: list of strings (what this character knows/doesn't know)\n"
        "- alignment_notes: string (how alignment influences speech)\n\n"
        "Base your analysis on the character data provided. Be specific and actionable — "
        "a writer should be able to read this profile and immediately write dialogue "
        "that sounds like this character."
    ),
    user="Character data:\n{character_context}\n\nGenerate the voice profile as a JSON object.",
    description="Synthesize a structured persona document from graph-derived character data.",
)

_PROMPT_MAP: dict[str, PromptTemplate] = {
    "extraction_v1": EXTRACTION_PROMPT_V1,
    "reclassification_v1": RECLASSIFICATION_PROMPT_V1,
    "generation_v1": GENERATION_PROMPT_V1,
    "persona_generation_v1": PERSONA_GENERATION_PROMPT_V1,
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
