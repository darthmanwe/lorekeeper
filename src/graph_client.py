"""Neo4j driver wrapper with typed query methods and idempotent MERGE helpers.

All writes use MERGE (never CREATE) to ensure idempotency.
All queries use parameterized Cypher — no string interpolation.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase, ManagedTransaction

from src.schema import (
    Character,
    Event,
    Faction,
    Location,
    Relationship,
    Segment,
    StoryObject,
)

logger = logging.getLogger(__name__)


class Neo4jWriteError(Exception):
    """Raised when a Neo4j write operation fails."""


class GraphClient:
    """Wraps the Neo4j Python driver with typed query methods for the
    Lorekeeper narrative knowledge graph.

    Args:
        uri: Neo4j Bolt connection URI (e.g. bolt://localhost:7687).
        user: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name (default: neo4j).
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        logger.info("GraphClient connected to %s", uri)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()
        logger.info("GraphClient connection closed")

    def verify_connectivity(self) -> bool:
        """Return True if the driver can reach the Neo4j server.

        Raises:
            Exception: If the server is unreachable.
        """
        self._driver.verify_connectivity()
        return True

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def create_constraints(self) -> None:
        """Create all required uniqueness constraints and indexes.

        Safe to call multiple times — uses IF NOT EXISTS.

        Raises:
            Neo4jWriteError: If any constraint/index creation fails.
        """
        statements = [
            (
                "CREATE CONSTRAINT character_name_unique IF NOT EXISTS "
                "FOR (c:Character) REQUIRE c.name IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT location_name_unique IF NOT EXISTS "
                "FOR (l:Location) REQUIRE l.name IS UNIQUE"
            ),
            (
                "CREATE INDEX event_seq_branch IF NOT EXISTS "
                "FOR (e:Event) ON (e.seq_id, e.branch_id)"
            ),
            (
                "CREATE INDEX segment_seq_branch IF NOT EXISTS "
                "FOR (s:Segment) ON (s.seq_id, s.branch_id)"
            ),
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in statements:
                try:
                    session.run(stmt)
                except Exception as exc:
                    raise Neo4jWriteError(
                        f"Failed to execute schema statement: {stmt}"
                    ) from exc
        logger.info("All constraints and indexes created/verified")

    # ------------------------------------------------------------------
    # MERGE helpers — node types
    # ------------------------------------------------------------------

    def merge_character(self, character: Character, branch_id: str) -> bool:
        """Idempotently write a Character node using MERGE on name.

        Args:
            character: Validated Character model instance.
            branch_id: Active story branch identifier.

        Returns:
            True if a new node was created, False if updated.

        Raises:
            Neo4jWriteError: If the MERGE fails.
        """
        query = """
        MERGE (c:Character {name: $name})
        ON CREATE SET
            c.status = $status,
            c.current_location_id = $current_location_id,
            c.alignment = $alignment,
            c.traits = $traits,
            c.persona_doc_id = $persona_doc_id,
            c.branch_id = $branch_id,
            c.created = true
        ON MATCH SET
            c.status = $status,
            c.current_location_id = $current_location_id,
            c.alignment = $alignment,
            c.traits = $traits,
            c.persona_doc_id = $persona_doc_id,
            c.branch_id = $branch_id,
            c.created = false
        RETURN c.created AS created
        """
        return self._merge_node(query, {
            "name": character.name,
            "status": character.status,
            "current_location_id": character.current_location_id,
            "alignment": character.alignment,
            "traits": character.traits,
            "persona_doc_id": character.persona_doc_id,
            "branch_id": branch_id,
        })

    def merge_location(self, location: Location) -> bool:
        """Idempotently write a Location node using MERGE on name.

        Args:
            location: Validated Location model instance.

        Returns:
            True if a new node was created, False if updated.

        Raises:
            Neo4jWriteError: If the MERGE fails.
        """
        query = """
        MERGE (l:Location {name: $name})
        ON CREATE SET
            l.type = $type,
            l.accessible = $accessible,
            l.description_summary = $description_summary,
            l.created = true
        ON MATCH SET
            l.type = $type,
            l.accessible = $accessible,
            l.description_summary = $description_summary,
            l.created = false
        RETURN l.created AS created
        """
        return self._merge_node(query, {
            "name": location.name,
            "type": location.type,
            "accessible": location.accessible,
            "description_summary": location.description_summary,
        })

    def merge_event(self, event: Event) -> bool:
        """Idempotently write an Event node using MERGE on seq_id + branch_id.

        Args:
            event: Validated Event model instance.

        Returns:
            True if a new node was created, False if updated.

        Raises:
            Neo4jWriteError: If the MERGE fails.
        """
        query = """
        MERGE (e:Event {seq_id: $seq_id, branch_id: $branch_id})
        ON CREATE SET
            e.description = $description,
            e.outcome = $outcome,
            e.timestamp = $timestamp,
            e.created = true
        ON MATCH SET
            e.description = $description,
            e.outcome = $outcome,
            e.timestamp = $timestamp,
            e.created = false
        RETURN e.created AS created
        """
        return self._merge_node(query, {
            "seq_id": event.seq_id,
            "branch_id": event.branch_id,
            "description": event.description,
            "outcome": event.outcome,
            "timestamp": event.timestamp,
        })

    def merge_object(self, obj: StoryObject, branch_id: str) -> bool:
        """Idempotently write an Object node using MERGE on name.

        Args:
            obj: Validated StoryObject model instance.
            branch_id: Active story branch identifier.

        Returns:
            True if a new node was created, False if updated.

        Raises:
            Neo4jWriteError: If the MERGE fails.
        """
        query = """
        MERGE (o:Object {name: $name})
        ON CREATE SET
            o.current_owner_id = $current_owner_id,
            o.significance = $significance,
            o.last_seen_location_id = $last_seen_location_id,
            o.branch_id = $branch_id,
            o.created = true
        ON MATCH SET
            o.current_owner_id = $current_owner_id,
            o.significance = $significance,
            o.last_seen_location_id = $last_seen_location_id,
            o.branch_id = $branch_id,
            o.created = false
        RETURN o.created AS created
        """
        return self._merge_node(query, {
            "name": obj.name,
            "current_owner_id": obj.current_owner_id,
            "significance": obj.significance,
            "last_seen_location_id": obj.last_seen_location_id,
            "branch_id": branch_id,
        })

    def merge_faction(self, faction: Faction, branch_id: str) -> bool:
        """Idempotently write a Faction node using MERGE on name.

        Args:
            faction: Validated Faction model instance.
            branch_id: Active story branch identifier.

        Returns:
            True if a new node was created, False if updated.

        Raises:
            Neo4jWriteError: If the MERGE fails.
        """
        query = """
        MERGE (f:Faction {name: $name})
        ON CREATE SET
            f.goals = $goals,
            f.member_ids = $member_ids,
            f.branch_id = $branch_id,
            f.created = true
        ON MATCH SET
            f.goals = $goals,
            f.member_ids = $member_ids,
            f.branch_id = $branch_id,
            f.created = false
        RETURN f.created AS created
        """
        return self._merge_node(query, {
            "name": faction.name,
            "goals": faction.goals,
            "member_ids": faction.member_ids,
            "branch_id": branch_id,
        })

    def merge_segment(self, segment: Segment) -> bool:
        """Idempotently write a Segment node using MERGE on seq_id + branch_id.

        Args:
            segment: Validated Segment model instance.

        Returns:
            True if a new node was created, False if updated.

        Raises:
            Neo4jWriteError: If the MERGE fails.
        """
        query = """
        MERGE (s:Segment {seq_id: $seq_id, branch_id: $branch_id})
        ON CREATE SET
            s.text = $text,
            s.embedding_id = $embedding_id,
            s.created = true
        ON MATCH SET
            s.text = $text,
            s.embedding_id = $embedding_id,
            s.created = false
        RETURN s.created AS created
        """
        return self._merge_node(query, {
            "seq_id": segment.seq_id,
            "branch_id": segment.branch_id,
            "text": segment.text,
            "embedding_id": segment.embedding_id,
        })

    # ------------------------------------------------------------------
    # MERGE helpers — relationships
    # ------------------------------------------------------------------

    def merge_relationship(self, rel: Relationship, branch_id: str) -> bool:
        """Idempotently write a relationship between two named nodes.

        Source and target are matched by name (or seq_id for Events).
        The relationship type determines which node labels to match against.

        Args:
            rel: Validated Relationship model instance.
            branch_id: Active story branch identifier.

        Returns:
            True if a new relationship was created, False if updated.

        Raises:
            Neo4jWriteError: If the MERGE fails.
        """
        label_map = {
            "KNOWS": ("Character", "Character"),
            "LOCATED_AT": ("Character", "Location"),
            "PARTICIPATED_IN": ("Character", "Event"),
            "CAUSED_BY": ("Event", "Event"),
            "OWNS": ("Character", "Object"),
            "VISITED": ("Character", "Location"),
            "MEMBER_OF": ("Character", "Faction"),
        }

        if rel.type == "REFERENCES_GRAPH_STATE":
            return self._merge_references_graph_state(
                rel.source, rel.target, branch_id
            )

        if rel.type not in label_map:
            raise Neo4jWriteError(f"Unknown relationship type: {rel.type}")

        src_label, tgt_label = label_map[rel.type]

        src_key = "seq_id" if src_label == "Event" else "name"
        tgt_key = "seq_id" if tgt_label == "Event" else "name"

        props = {**rel.properties, "branch_id": branch_id}
        prop_set = ", ".join(f"r.{k} = ${k}" for k in props)

        query = f"""
        MATCH (src:{src_label} {{{src_key}: $src_id}})
        MATCH (tgt:{tgt_label} {{{tgt_key}: $tgt_id}})
        MERGE (src)-[r:{rel.type}]->(tgt)
        SET {prop_set}
        RETURN r
        """
        src_val: Any = int(rel.source) if src_label == "Event" else rel.source
        tgt_val: Any = int(rel.target) if tgt_label == "Event" else rel.target

        params: dict[str, Any] = {
            "src_id": src_val,
            "tgt_id": tgt_val,
            **props,
        }
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, params)
                record = result.single()
                return record is not None
        except Exception as exc:
            raise Neo4jWriteError(
                f"Failed to merge relationship {rel.type}: {rel.source} -> {rel.target}"
            ) from exc

    def create_references_graph_state(
        self,
        segment_seq_id: int,
        branch_id: str,
        referenced_node_names: list[str],
    ) -> None:
        """Create REFERENCES_GRAPH_STATE links from a Segment to referenced nodes.

        Args:
            segment_seq_id: The segment's sequence ID.
            branch_id: Active branch ID.
            referenced_node_names: Names of nodes this segment references.

        Raises:
            Neo4jWriteError: If any link creation fails.
        """
        query = """
        MATCH (s:Segment {seq_id: $seq_id, branch_id: $branch_id})
        MATCH (n)
        WHERE n.name = $node_name
        MERGE (s)-[:REFERENCES_GRAPH_STATE]->(n)
        """
        with self._driver.session(database=self._database) as session:
            for name in referenced_node_names:
                try:
                    session.run(query, {
                        "seq_id": segment_seq_id,
                        "branch_id": branch_id,
                        "node_name": name,
                    })
                except Exception as exc:
                    raise Neo4jWriteError(
                        f"Failed to link segment {segment_seq_id} to node '{name}'"
                    ) from exc

    def link_event_participants(
        self,
        seq_id: int,
        branch_id: str,
        character_names: list[str],
    ) -> int:
        """Create PARTICIPATED_IN edges from characters to an event.

        Uses MERGE to remain idempotent. Characters that don't exist in the
        graph are silently skipped.

        Returns:
            Number of PARTICIPATED_IN edges created or matched.
        """
        if not character_names:
            return 0
        query = """
        MATCH (e:Event {seq_id: $seq_id, branch_id: $branch_id})
        MATCH (c:Character {name: $char_name})
        MERGE (c)-[r:PARTICIPATED_IN]->(e)
        SET r.branch_id = $branch_id
        RETURN c.name AS linked
        """
        linked = 0
        with self._driver.session(database=self._database) as session:
            for name in character_names:
                try:
                    result = session.run(query, {
                        "seq_id": seq_id,
                        "branch_id": branch_id,
                        "char_name": name,
                    })
                    if result.single():
                        linked += 1
                except Exception:
                    logger.warning(
                        "Failed to link character '%s' to event %d", name, seq_id
                    )
        logger.info(
            "Linked %d/%d characters to Event #%d",
            linked, len(character_names), seq_id,
        )
        return linked

    def link_event_causality(
        self,
        seq_id: int,
        branch_id: str,
    ) -> bool:
        """Create a CAUSED_BY edge from this event to the immediately previous
        event in the same branch. Skips if seq_id <= 1 or no predecessor exists.

        Returns:
            True if a CAUSED_BY edge was created/matched, False otherwise.
        """
        if seq_id <= 1:
            return False
        query = """
        MATCH (curr:Event {seq_id: $seq_id, branch_id: $branch_id})
        MATCH (prev:Event {branch_id: $branch_id})
        WHERE prev.seq_id < $seq_id
        WITH curr, prev ORDER BY prev.seq_id DESC LIMIT 1
        MERGE (curr)-[r:CAUSED_BY]->(prev)
        SET r.branch_id = $branch_id
        RETURN prev.seq_id AS prev_seq
        """
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, {
                    "seq_id": seq_id,
                    "branch_id": branch_id,
                })
                record = result.single()
                if record:
                    logger.info(
                        "Linked Event #%d -> CAUSED_BY -> Event #%d",
                        seq_id, record["prev_seq"],
                    )
                    return True
                return False
        except Exception:
            logger.warning("Failed to create CAUSED_BY for event %d", seq_id)
            return False

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def enrich_structural_edges(self, branch_id: str) -> dict[str, int]:
        """Batch-create structural edges from node properties.

        Scans all Characters and Objects and creates edges that should exist
        based on their property values but are missing as actual relationships.
        Useful for retroactive enrichment or after seed ingestion.

        Returns:
            Dict with counts of edges created per type.
        """
        counts: dict[str, int] = {"LOCATED_AT": 0, "OWNS": 0}

        char_q = """
        MATCH (c:Character)
        WHERE c.current_location_id IS NOT NULL
          AND NOT (c)-[:LOCATED_AT]->(:Location {name: c.current_location_id})
        MATCH (l:Location {name: c.current_location_id})
        MERGE (c)-[r:LOCATED_AT]->(l)
        SET r.branch_id = $branch_id
        RETURN c.name AS char_name, l.name AS loc_name
        """
        with self._driver.session(database=self._database) as session:
            for r in session.run(char_q, {"branch_id": branch_id}):
                counts["LOCATED_AT"] += 1
                logger.info(
                    "Enriched: %s -[:LOCATED_AT]-> %s",
                    r["char_name"], r["loc_name"],
                )

        obj_q = """
        MATCH (o:Object)
        WHERE o.current_owner_id IS NOT NULL
          AND NOT (:Character {name: o.current_owner_id})-[:OWNS]->(o)
        MATCH (c:Character {name: o.current_owner_id})
        MERGE (c)-[r:OWNS]->(o)
        SET r.branch_id = $branch_id
        RETURN c.name AS owner, o.name AS obj_name
        """
        with self._driver.session(database=self._database) as session:
            for r in session.run(obj_q, {"branch_id": branch_id}):
                counts["OWNS"] += 1
                logger.info(
                    "Enriched: %s -[:OWNS]-> %s",
                    r["owner"], r["obj_name"],
                )

        logger.info("Structural enrichment: %s", counts)
        return counts

    def get_characters_at_location(
        self, location: str, branch_id: str
    ) -> list[Character]:
        """Return all alive characters at the given location on the given branch.

        Args:
            location: Location name.
            branch_id: Active branch ID.

        Returns:
            List of Character model instances.
        """
        query = """
        MATCH (c:Character)-[:LOCATED_AT]->(l:Location {name: $location})
        WHERE c.status = 'alive' AND c.branch_id = $branch_id
        RETURN c
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, {"location": location, "branch_id": branch_id})
            return [self._record_to_character(r["c"]) for r in result]

    def get_causal_chain(
        self, last_seq_id: int, branch_id: str, depth: int = 3
    ) -> list[Event]:
        """Return the causal ancestry chain from recent events.

        Args:
            last_seq_id: Most recent event sequence ID.
            branch_id: Active branch ID.
            depth: Maximum CAUSED_BY hops to traverse.

        Returns:
            List of Event model instances ordered by seq_id descending.
        """
        query = """
        MATCH (e:Event {branch_id: $branch_id})
        WHERE e.seq_id >= $min_seq
        OPTIONAL MATCH (e)-[:CAUSED_BY*1..$depth]->(cause:Event)
        RETURN DISTINCT e ORDER BY e.seq_id DESC
        """
        params = {
            "branch_id": branch_id,
            "min_seq": max(0, last_seq_id - 3),
            "depth": depth,
        }
        with self._driver.session(database=self._database) as session:
            result = session.run(query, params)
            return [self._record_to_event(r["e"]) for r in result]

    def get_all_character_names(self, branch_id: str) -> list[str]:
        """Return all Character names in the graph for event participant matching."""
        query = """
        MATCH (c:Character)
        WHERE c.branch_id = $branch_id OR c.branch_id IS NULL
        RETURN DISTINCT c.name AS name
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, {"branch_id": branch_id})
            return [r["name"] for r in result]

    def get_all_entity_names(self, branch_id: str) -> list[str]:
        """Return all entity names in the graph for fuzzy matching.

        Args:
            branch_id: Active branch ID (used for branch-scoped entities).

        Returns:
            List of unique entity name strings.
        """
        query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        RETURN DISTINCT n.name AS name
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query)
            return [r["name"] for r in result]

    def snapshot_graph_state(self, branch_id: str, seq_id: int) -> dict[str, Any]:
        """Return a serializable snapshot of graph state as of a given sequence point.

        Captures all nodes and relationships that existed at or before seq_id
        on the given branch via REFERENCES_GRAPH_STATE links.

        Args:
            branch_id: Active branch ID.
            seq_id: Sequence point to snapshot at.

        Returns:
            Dict with 'nodes' and 'relationships' keys.
        """
        nodes_query = """
        MATCH (s:Segment {branch_id: $branch_id})-[:REFERENCES_GRAPH_STATE]->(n)
        WHERE s.seq_id <= $seq_id
        RETURN DISTINCT labels(n) AS labels, properties(n) AS props
        """
        rels_query = """
        MATCH (s:Segment {branch_id: $branch_id})-[:REFERENCES_GRAPH_STATE]->(n)
        WHERE s.seq_id <= $seq_id
        MATCH (n)-[r]->(m)
        WHERE type(r) <> 'REFERENCES_GRAPH_STATE'
        RETURN DISTINCT type(r) AS rel_type, properties(r) AS props,
               properties(n) AS source_props, properties(m) AS target_props
        """
        with self._driver.session(database=self._database) as session:
            nodes_result = session.run(nodes_query, {
                "branch_id": branch_id,
                "seq_id": seq_id,
            })
            nodes = [
                {"labels": list(r["labels"]), "properties": dict(r["props"])}
                for r in nodes_result
            ]

            rels_result = session.run(rels_query, {
                "branch_id": branch_id,
                "seq_id": seq_id,
            })
            relationships = [
                {
                    "type": r["rel_type"],
                    "properties": dict(r["props"]),
                    "source": dict(r["source_props"]),
                    "target": dict(r["target_props"]),
                }
                for r in rels_result
            ]

        return {"nodes": nodes, "relationships": relationships}

    def get_node_counts(self) -> dict[str, int]:
        """Return count of each node label in the database.

        Returns:
            Dict mapping label names to counts.
        """
        query = """
        CALL db.labels() YIELD label
        CALL (label) {
            MATCH (n)
            WHERE label IN labels(n)
            RETURN count(n) AS cnt
        }
        RETURN label, cnt
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query)
            return {r["label"]: r["cnt"] for r in result}

    def get_relationship_counts(self) -> dict[str, int]:
        """Return count of each relationship type in the database.

        Returns:
            Dict mapping relationship type names to counts.
        """
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        CALL (relationshipType) {
            MATCH ()-[r]->()
            WHERE type(r) = relationshipType
            RETURN count(r) AS cnt
        }
        RETURN relationshipType, cnt
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query)
            return {r["relationshipType"]: r["cnt"] for r in result}

    def clear_database(self) -> None:
        """Delete all nodes and relationships. For testing/notebook reset only.

        Raises:
            Neo4jWriteError: If the deletion fails.
        """
        try:
            with self._driver.session(database=self._database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Database cleared — all nodes and relationships deleted")
        except Exception as exc:
            raise Neo4jWriteError("Failed to clear database") from exc

    def get_graph_summary_facts(self, branch_id: str) -> list[str]:
        """Return a list of human-readable fact strings derived from graph state.

        Used by the eval harness to provide the LLM judge with structured
        ground truth (not raw segment text).

        Args:
            branch_id: Active branch ID.

        Returns:
            List of fact strings.
        """
        facts: list[str] = []

        char_query = """
        MATCH (c:Character)
        WHERE c.branch_id = $branch_id OR c.branch_id IS NULL
        RETURN c.name AS name, c.status AS status,
               c.current_location_id AS location, c.alignment AS alignment
        """
        with self._driver.session(database=self._database) as session:
            for r in session.run(char_query, {"branch_id": branch_id}):
                facts.append(
                    f"{r['name']} is {r['status']}"
                    + (f" at {r['location']}" if r["location"] else "")
                )

        rel_query = """
        MATCH (a:Character)-[r:KNOWS]->(b:Character)
        WHERE r.branch_id = $branch_id OR r.branch_id IS NULL
        RETURN a.name AS a_name, b.name AS b_name, r.sentiment AS sentiment
        """
        with self._driver.session(database=self._database) as session:
            for r in session.run(rel_query, {"branch_id": branch_id}):
                facts.append(
                    f"{r['a_name']} and {r['b_name']} have a {r['sentiment']} relationship"
                )

        owns_query = """
        MATCH (c:Character)-[r:OWNS]->(o:Object)
        WHERE r.branch_id = $branch_id OR r.branch_id IS NULL
        RETURN c.name AS owner, o.name AS object_name
        """
        with self._driver.session(database=self._database) as session:
            for r in session.run(owns_query, {"branch_id": branch_id}):
                facts.append(f"{r['owner']} owns {r['object_name']}")

        event_query = """
        MATCH (e:Event {branch_id: $branch_id})
        RETURN e.description AS desc, e.seq_id AS seq_id
        ORDER BY e.seq_id
        """
        with self._driver.session(database=self._database) as session:
            for r in session.run(event_query, {"branch_id": branch_id}):
                facts.append(f"Event #{r['seq_id']}: {r['desc']}")

        return facts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge_node(self, query: str, params: dict[str, Any]) -> bool:
        """Execute a MERGE query and return whether a new node was created."""
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, params)
                record = result.single()
                return bool(record and record.get("created", False))
        except Exception as exc:
            raise Neo4jWriteError(
                f"MERGE failed with params: {params}"
            ) from exc

    @staticmethod
    def _record_to_character(node: Any) -> Character:
        """Convert a Neo4j node record to a Character model."""
        props = dict(node)
        return Character(
            name=props.get("name", ""),
            status=props.get("status", "alive"),
            current_location_id=props.get("current_location_id"),
            alignment=props.get("alignment"),
            traits=props.get("traits", []),
            persona_doc_id=props.get("persona_doc_id"),
        )

    @staticmethod
    def _record_to_event(node: Any) -> Event:
        """Convert a Neo4j node record to an Event model."""
        props = dict(node)
        return Event(
            description=props.get("description", ""),
            seq_id=props.get("seq_id", 0),
            branch_id=props.get("branch_id", "main"),
            outcome=props.get("outcome"),
            timestamp=props.get("timestamp"),
        )

    def _merge_references_graph_state(
        self, source: str, target: str, branch_id: str
    ) -> bool:
        """MERGE a REFERENCES_GRAPH_STATE relationship (Segment -> any node)."""
        query = """
        MATCH (s:Segment {seq_id: $src_id, branch_id: $branch_id})
        MATCH (n {name: $tgt_id})
        MERGE (s)-[r:REFERENCES_GRAPH_STATE]->(n)
        RETURN r
        """
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, {
                    "src_id": int(source),
                    "tgt_id": target,
                    "branch_id": branch_id,
                })
                return result.single() is not None
        except Exception as exc:
            raise Neo4jWriteError(
                f"Failed to merge REFERENCES_GRAPH_STATE: {source} -> {target}"
            ) from exc
