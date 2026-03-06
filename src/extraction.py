"""Two-stage extraction pipeline: propose (LLM) -> validate (deterministic) -> commit (Neo4j).

Stage 1 — Propose: LLM extracts entities and relationships with confidence scores.
Stage 2 — Validate: deterministic checks (confidence threshold, name resolution,
  status consistency, relationship directionality) split proposals into approved/flagged.
Commit: approved proposals are MERGEd to Neo4j with REFERENCES_GRAPH_STATE links.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from thefuzz import fuzz

from src.graph_client import GraphClient
from src.prompts import get_prompt
from src.schema import (
    Character,
    ConstraintViolation,
    Event,
    ExtractionProposal,
    ExtractionResult,
    Location,
    Relationship,
    Segment,
    StoryObject,
)

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.65
FUZZY_MATCH_THRESHOLD = 85
RECLASSIFICATION_INTERVAL = 10


class NameResolver:
    """Fuzzy name matching against existing graph entities to prevent duplicates."""

    @staticmethod
    def fuzzy_match(
        proposed_name: str,
        existing_names: list[str],
        threshold: int = FUZZY_MATCH_THRESHOLD,
    ) -> str | None:
        """Find the best fuzzy match for a proposed name among existing entities.

        Args:
            proposed_name: The name proposed by the extraction LLM.
            existing_names: All entity names currently in the graph.
            threshold: Minimum fuzz ratio (0-100) to consider a match.

        Returns:
            The matching existing name if found above threshold, else None.
        """
        if not existing_names:
            return None

        best_match = None
        best_score = 0

        for name in existing_names:
            score = fuzz.ratio(proposed_name.lower(), name.lower())
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= threshold and best_match != proposed_name:
            logger.info(
                "Fuzzy matched '%s' -> '%s' (score=%d)",
                proposed_name,
                best_match,
                best_score,
            )
            return best_match
        return None


class StatusValidator:
    """Validates entity status transitions against graph state."""

    @staticmethod
    def check_status_consistency(
        proposal: ExtractionProposal,
        graph_client: GraphClient,
        branch_id: str,
    ) -> ConstraintViolation | None:
        """Check if a proposed status change violates graph constraints.

        A dead character cannot be set to alive without an explicit
        resurrection event in the same extraction batch.

        Args:
            proposal: The extraction proposal to validate.
            graph_client: Active graph client for state lookups.
            branch_id: Active branch ID.

        Returns:
            A ConstraintViolation if the change is invalid, else None.
        """
        if proposal.entity_type != "Character":
            return None

        new_status = proposal.properties.get("status")
        if new_status != "alive":
            return None

        query = """
        MATCH (c:Character {name: $name})
        WHERE c.status = 'dead'
        RETURN c.name AS name
        """
        with graph_client._driver.session(database=graph_client._database) as session:
            result = session.run(query, {"name": proposal.entity_name})
            record = result.single()

        if record is not None:
            return ConstraintViolation(
                check_name="status_consistency",
                violation_message=(
                    f"Cannot set '{proposal.entity_name}' to alive — "
                    f"character is dead. Requires explicit resurrection event."
                ),
                severity="critical",
            )
        return None


class ExtractionPipeline:
    """Orchestrates the two-stage extraction pipeline.

    Args:
        llm: ChatAnthropic instance for LLM calls.
        graph_client: GraphClient for graph reads and writes.
    """

    def __init__(self, llm: ChatAnthropic, graph_client: GraphClient) -> None:
        self._llm = llm
        self._gc = graph_client

    def propose(
        self,
        segment_text: str,
        existing_entities: list[str],
    ) -> list[ExtractionProposal]:
        """Stage 1: LLM extracts entities and relationships with confidence scores.

        Args:
            segment_text: The generated story segment to extract from.
            existing_entities: Names of all entities currently in the graph.

        Returns:
            List of ExtractionProposal instances.
        """
        prompt = get_prompt("extraction_v1")
        system_msg = prompt.format_system(
            existing_entities=", ".join(existing_entities) if existing_entities else "(none)"
        )
        user_msg = prompt.format_user(segment_text=segment_text)

        response = self._llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ])

        return self._parse_proposals(response.content)

    def validate(
        self,
        proposals: list[ExtractionProposal],
        branch_id: str,
    ) -> tuple[list[ExtractionProposal], list[ExtractionProposal]]:
        """Stage 2: deterministic validation splits proposals into approved/flagged.

        Checks applied:
        1. Confidence threshold (below 0.65 -> flagged)
        2. Fuzzy name resolution (near-matches resolve to existing names)
        3. Status consistency (dead chars can't become alive without resurrection)
        4. Relationship directionality (CAUSED_BY requires both events to exist)

        Args:
            proposals: Raw proposals from Stage 1.
            branch_id: Active branch ID for graph lookups.

        Returns:
            Tuple of (approved, flagged) proposal lists.
        """
        existing_names = self._gc.get_all_entity_names(branch_id)
        approved: list[ExtractionProposal] = []
        flagged: list[ExtractionProposal] = []

        for proposal in proposals:
            flag_reasons: list[str] = []

            if proposal.confidence < CONFIDENCE_THRESHOLD:
                flag_reasons.append(
                    f"confidence {proposal.confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
                )

            matched_name = NameResolver.fuzzy_match(
                proposal.entity_name, existing_names
            )
            if matched_name is not None:
                proposal.entity_name = matched_name
                proposal.action = "update"

            violation = StatusValidator.check_status_consistency(
                proposal, self._gc, branch_id
            )
            if violation is not None:
                flag_reasons.append(violation.violation_message)

            if (
                proposal.entity_type == "Relationship"
                and proposal.properties.get("rel_type") == "CAUSED_BY"
            ):
                source_event = proposal.properties.get("source")
                target_event = proposal.properties.get("target")
                if not self._events_exist(source_event, target_event, branch_id):
                    flag_reasons.append(
                        f"CAUSED_BY requires both events to exist: "
                        f"'{source_event}' -> '{target_event}'"
                    )

            if flag_reasons:
                logger.info(
                    "Flagged '%s' (%s): %s",
                    proposal.entity_name,
                    proposal.entity_type,
                    "; ".join(flag_reasons),
                )
                flagged.append(proposal)
            else:
                approved.append(proposal)

        return approved, flagged

    def commit(
        self,
        approved: list[ExtractionProposal],
        branch_id: str,
        seq_id: int,
    ) -> int:
        """Write approved proposals to Neo4j via MERGE, then create REFERENCES_GRAPH_STATE links.

        After committing entity nodes, performs deterministic event linking:
        - PARTICIPATED_IN: characters mentioned in an event description
        - CAUSED_BY: chain to the previous event in the same branch

        Args:
            approved: Validated proposals to commit.
            branch_id: Active branch ID.
            seq_id: Current segment sequence ID.

        Returns:
            Number of proposals successfully committed.
        """
        committed = 0
        referenced_names: list[str] = []
        committed_events: list[tuple[int, str]] = []

        for proposal in approved:
            try:
                self._commit_single(proposal, branch_id, seq_id)
                committed += 1
                if proposal.entity_type == "Event":
                    desc = proposal.properties.get(
                        "description", proposal.entity_name
                    )
                    committed_events.append((seq_id, desc))
                if proposal.entity_type != "Relationship":
                    referenced_names.append(proposal.entity_name)
                else:
                    src = proposal.properties.get("source", "")
                    tgt = proposal.properties.get("target", "")
                    if src:
                        referenced_names.append(src)
                    if tgt:
                        referenced_names.append(tgt)
            except Exception:
                logger.exception(
                    "Failed to commit %s '%s'",
                    proposal.entity_type,
                    proposal.entity_name,
                )

        if referenced_names:
            try:
                self._gc.create_references_graph_state(
                    seq_id, branch_id, referenced_names
                )
            except Exception:
                logger.exception("Failed to create REFERENCES_GRAPH_STATE links")

        self._auto_link_all(committed_events, approved, branch_id)

        return committed

    def _auto_link_all(
        self,
        committed_events: list[tuple[int, str]],
        all_approved: list[ExtractionProposal],
        branch_id: str,
    ) -> None:
        """Deterministic post-commit structural edge creation.

        Runs after all proposals in a batch are committed so target nodes
        already exist. Creates edges that mirror property values:
        - Event: PARTICIPATED_IN (characters mentioned in description) + CAUSED_BY
        - Character: LOCATED_AT (from current_location_id property)
        - Object: OWNS (from current_owner_id property)
        """
        all_char_names = self._gc.get_all_character_names(branch_id)
        batch_char_names = {
            p.entity_name for p in all_approved if p.entity_type == "Character"
        }
        all_char_names = list(set(all_char_names) | batch_char_names)

        for evt_seq_id, evt_desc in committed_events:
            participants = self._find_mentioned_characters(
                evt_desc, all_char_names
            )
            if participants:
                self._gc.link_event_participants(
                    evt_seq_id, branch_id, participants
                )
            self._gc.link_event_causality(evt_seq_id, branch_id)

        for p in all_approved:
            if p.entity_type == "Character":
                loc = p.properties.get("current_location_id")
                if loc:
                    rel = Relationship(
                        type="LOCATED_AT",
                        source=p.entity_name,
                        target=loc,
                    )
                    try:
                        self._gc.merge_relationship(rel, branch_id)
                        logger.info(
                            "Auto-linked %s -[:LOCATED_AT]-> %s",
                            p.entity_name, loc,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to auto-link LOCATED_AT: %s -> %s",
                            p.entity_name, loc,
                        )

            elif p.entity_type == "Object":
                owner = p.properties.get("current_owner_id")
                if owner:
                    rel = Relationship(
                        type="OWNS",
                        source=owner,
                        target=p.entity_name,
                    )
                    try:
                        self._gc.merge_relationship(rel, branch_id)
                        logger.info(
                            "Auto-linked %s -[:OWNS]-> %s",
                            owner, p.entity_name,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to auto-link OWNS: %s -> %s",
                            owner, p.entity_name,
                        )

    @staticmethod
    def _find_mentioned_characters(
        text: str, character_names: list[str]
    ) -> list[str]:
        """Return character names that appear in the text (case-insensitive)."""
        text_lower = text.lower()
        return [
            name for name in character_names
            if name.lower() in text_lower
        ]

    def run(
        self,
        segment_text: str,
        branch_id: str,
        seq_id: int,
        auto_approve: bool = True,
    ) -> ExtractionResult:
        """Execute the full extraction pipeline: propose -> validate -> commit.

        Args:
            segment_text: The generated story segment to extract from.
            branch_id: Active branch ID.
            seq_id: Current segment sequence ID.
            auto_approve: If True, commit all approved proposals without
                interactive review. Set to False for notebook HITL mode.

        Returns:
            ExtractionResult with proposals, approved, flagged, and committed count.
        """
        existing = self._gc.get_all_entity_names(branch_id)
        proposals = self.propose(segment_text, existing)
        approved, flagged = self.validate(proposals, branch_id)

        committed_count = 0
        if auto_approve and approved:
            committed_count = self.commit(approved, branch_id, seq_id)

        return ExtractionResult(
            proposals=proposals,
            approved=approved,
            flagged=flagged,
            committed_count=committed_count,
        )

    def reclassify(self, branch_id: str) -> list[dict[str, Any]]:
        """Run a schema reclassification pass on all characters.

        Triggered every RECLASSIFICATION_INTERVAL segments. Sends the current
        character list to the LLM and returns a structured diff of trait updates,
        alignment changes, and faction reassignments.

        Args:
            branch_id: Active branch ID.

        Returns:
            List of update dicts with keys: character_name, updates, reasoning.
        """
        char_query = """
        MATCH (c:Character)
        WHERE c.branch_id = $branch_id OR c.branch_id IS NULL
        RETURN c.name AS name, c.status AS status, c.alignment AS alignment,
               c.traits AS traits
        """
        event_query = """
        MATCH (e:Event {branch_id: $branch_id})
        RETURN e.description AS desc, e.seq_id AS seq_id
        ORDER BY e.seq_id DESC LIMIT 10
        """
        with self._gc._driver.session(database=self._gc._database) as session:
            chars = [
                {k: v for k, v in dict(r).items()}
                for r in session.run(char_query, {"branch_id": branch_id})
            ]
            events = [
                {k: v for k, v in dict(r).items()}
                for r in session.run(event_query, {"branch_id": branch_id})
            ]

        prompt = get_prompt("reclassification_v1")
        system_msg = prompt.format_system()
        user_msg = prompt.format_user(
            character_state=json.dumps(chars, indent=2),
            recent_events=json.dumps(events, indent=2),
        )

        response = self._llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ])

        try:
            updates = self._parse_json_response(response.content)
            if not isinstance(updates, list):
                return []
            return updates
        except Exception:
            logger.exception("Failed to parse reclassification response")
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_proposals(self, content: str) -> list[ExtractionProposal]:
        """Parse LLM response content into ExtractionProposal instances."""
        raw = self._parse_json_response(content)
        if not isinstance(raw, list):
            logger.warning("LLM extraction returned non-list: %s", type(raw))
            return []

        proposals: list[ExtractionProposal] = []
        for item in raw:
            try:
                proposals.append(ExtractionProposal(**item))
            except Exception as exc:
                logger.warning("Skipping malformed proposal: %s — %s", item, exc)
        return proposals

    @staticmethod
    def _parse_json_response(content: str) -> Any:
        """Extract and parse JSON from an LLM response with robust error recovery.

        Handles: markdown fences, truncated strings, trailing commas,
        partial arrays, and other common LLM output malformations.
        """
        text = content.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            start = 1
            end = len(lines)
            for i in range(1, len(lines)):
                if lines[i].strip() == "```":
                    end = i
                    break
            text = "\n".join(lines[start:end])

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        bracket_start = text.find("[")
        brace_start = text.find("{")
        if bracket_start == -1 and brace_start == -1:
            logger.warning("No JSON array or object found in LLM response")
            return []

        if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
            text = text[bracket_start:]
        elif brace_start != -1:
            text = text[brace_start:]

        import re
        text = re.sub(r",\s*([}\]])", r"\1", text)

        bracket_depth = 0
        brace_depth = 0
        in_string = False
        escape_next = False
        last_valid = 0

        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1

            if bracket_depth == 0 and brace_depth == 0:
                last_valid = i
                break

        if bracket_depth > 0 or brace_depth > 0:
            while bracket_depth > 0:
                if in_string:
                    text += '"'
                    in_string = False
                text += "]"
                bracket_depth -= 1
            while brace_depth > 0:
                if in_string:
                    text += '"'
                    in_string = False
                text += "}"
                brace_depth -= 1
            text = re.sub(r",\s*([}\]])", r"\1", text)
            logger.warning("Repaired truncated JSON from LLM response")
        elif last_valid > 0:
            text = text[: last_valid + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        last_complete = text.rfind("},")
        if last_complete != -1:
            truncated = text[: last_complete + 1] + "]"
            truncated = re.sub(r",\s*([}\]])", r"\1", truncated)
            try:
                result = json.loads(truncated)
                logger.warning(
                    "Recovered %d complete items from truncated JSON", len(result)
                )
                return result
            except json.JSONDecodeError:
                pass

        last_brace = text.rfind("}")
        if last_brace != -1:
            truncated = text[: last_brace + 1]
            if not truncated.startswith("["):
                truncated = "[" + truncated + "]"
            else:
                truncated += "]"
            truncated = re.sub(r",\s*([}\]])", r"\1", truncated)
            try:
                result = json.loads(truncated)
                logger.warning(
                    "Recovered %d items via last-brace fallback", len(result)
                )
                return result
            except json.JSONDecodeError:
                pass

        logger.error("JSON parse failed after all repair attempts")
        return []

    @staticmethod
    def _normalize_status(raw: str) -> str:
        """Map LLM-returned status synonyms to the canonical Literal values.

        Any status not in the canonical set or mapping is treated as "alive"
        to prevent Pydantic validation failures on creative LLM outputs
        like "wounded", "unconscious", "captured", etc.
        """
        canonical = {"alive", "dead", "unknown"}
        if not raw:
            return "alive"
        lower = raw.lower().strip()
        if lower in canonical:
            return lower
        mapping = {
            "active": "alive",
            "living": "alive",
            "healthy": "alive",
            "wounded": "alive",
            "injured": "alive",
            "unconscious": "alive",
            "captured": "alive",
            "missing": "unknown",
            "disappeared": "unknown",
            "absent": "unknown",
            "deceased": "dead",
            "killed": "dead",
            "slain": "dead",
            "fallen": "dead",
            "destroyed": "dead",
        }
        normalized = mapping.get(lower, "alive")
        if lower not in mapping:
            logger.debug("Unmapped character status '%s' defaulted to 'alive'", raw)
        return normalized

    def _commit_single(
        self, proposal: ExtractionProposal, branch_id: str, seq_id: int
    ) -> None:
        """MERGE a single approved proposal to Neo4j."""
        props = proposal.properties

        if proposal.entity_type == "Character":
            char = Character(
                name=proposal.entity_name,
                status=self._normalize_status(props.get("status", "alive")),
                current_location_id=props.get("current_location_id"),
                alignment=props.get("alignment"),
                traits=props.get("traits", []),
            )
            self._gc.merge_character(char, branch_id)

        elif proposal.entity_type == "Location":
            loc = Location(
                name=proposal.entity_name,
                type=props.get("type", "generic"),
                accessible=props.get("accessible", True),
                description_summary=props.get("description_summary", ""),
            )
            self._gc.merge_location(loc)

        elif proposal.entity_type == "Event":
            evt = Event(
                description=props.get("description", proposal.entity_name),
                seq_id=seq_id,
                branch_id=branch_id,
                outcome=props.get("outcome"),
            )
            self._gc.merge_event(evt)

        elif proposal.entity_type == "Object":
            obj = StoryObject(
                name=proposal.entity_name,
                current_owner_id=props.get("current_owner_id"),
                significance=props.get("significance", ""),
                last_seen_location_id=props.get("last_seen_location_id"),
            )
            self._gc.merge_object(obj, branch_id)

        elif proposal.entity_type == "Faction":
            from src.schema import Faction

            faction = Faction(
                name=proposal.entity_name,
                goals=props.get("goals", []),
                member_ids=props.get("member_ids", []),
            )
            self._gc.merge_faction(faction, branch_id)

        elif proposal.entity_type == "Relationship":
            rel = Relationship(
                type=props.get("rel_type", "KNOWS"),
                source=props.get("source", ""),
                target=props.get("target", ""),
                properties={
                    k: v
                    for k, v in props.items()
                    if k not in ("rel_type", "source", "target")
                },
            )
            self._gc.merge_relationship(rel, branch_id)

    def _events_exist(
        self,
        source_ref: str | None,
        target_ref: str | None,
        branch_id: str,
    ) -> bool:
        """Check whether both referenced events exist in the graph."""
        if not source_ref or not target_ref:
            return False
        query = """
        OPTIONAL MATCH (e1:Event {branch_id: $branch_id})
        WHERE e1.description CONTAINS $src OR toString(e1.seq_id) = $src
        OPTIONAL MATCH (e2:Event {branch_id: $branch_id})
        WHERE e2.description CONTAINS $tgt OR toString(e2.seq_id) = $tgt
        RETURN e1 IS NOT NULL AS src_exists, e2 IS NOT NULL AS tgt_exists
        """
        with self._gc._driver.session(database=self._gc._database) as session:
            result = session.run(query, {
                "branch_id": branch_id,
                "src": str(source_ref),
                "tgt": str(target_ref),
            })
            record = result.single()
            if record is None:
                return False
            return bool(record["src_exists"] and record["tgt_exists"])
