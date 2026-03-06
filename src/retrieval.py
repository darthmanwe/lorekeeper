"""Tiered Cypher retrieval, ChromaDB vector retrieval, and context assembly.

Implements the dual retrieval engine from the design document:
- Graph retrieval: T1 (active scene) -> T2 (causal chain) -> T3 (hostile tensions) -> T4 (orphans)
  with hard token budget enforcement and tier prioritization
- Vector retrieval: ChromaDB semantic search for tonal anchors
- Context assembly: merges graph facts + tonal anchors into structured prompt sections

Token counting uses tiktoken cl100k_base as an approximation for Claude tokenization.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import tiktoken

from src.graph_client import GraphClient
from src.schema import ConstraintViolation, SessionState

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _text_overlap_ratio(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard similarity between two texts.

    Used to detect near-duplicate tonal anchors that could cause the LLM
    to regurgitate previous segments verbatim rather than match their style.

    Returns:
        Similarity ratio in [0.0, 1.0].
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def count_tokens(text: str) -> int:
    """Count approximate tokens using cl100k_base encoding.

    This is an approximation for Claude tokenization (~5% variance).
    Sufficient for budget enforcement where exact counts aren't critical.

    Args:
        text: Input text to count.

    Returns:
        Approximate token count.
    """
    if not text:
        return 0
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Cypher Retrieval — Tiered policy with budget enforcement
# ---------------------------------------------------------------------------


class CypherRetriever:
    """Tiered Cypher retrieval engine with token budget enforcement.

    Tiers are executed in priority order. Each tier's output is included
    only if the remaining budget allows. T1 (active scene) is always
    included; T2-T4 are best-effort.

    Args:
        graph_client: Active GraphClient instance.
        token_budget: Maximum tokens for the entire graph context block.
    """

    def __init__(self, graph_client: GraphClient, token_budget: int = 2000) -> None:
        self._gc = graph_client
        self._budget = token_budget

    def t1_active_scene(self, location: str, branch_id: str) -> str:
        """T1: Characters at current location with KNOWS relationships and recent events.

        Always included (highest priority).

        Args:
            location: Current location name.
            branch_id: Active branch ID.

        Returns:
            Formatted fact string for the active scene.
        """
        query = """
        MATCH (c:Character)-[:LOCATED_AT]->(l:Location {name: $location})
        WHERE c.status = 'alive' AND (c.branch_id = $branch_id OR c.branch_id IS NULL)
        OPTIONAL MATCH (c)-[r:KNOWS]-(other:Character)
        WHERE other.status = 'alive'
        RETURN c.name AS char_name, c.status AS status, c.alignment AS alignment,
               c.traits AS traits, l.name AS loc_name, l.description_summary AS loc_desc,
               other.name AS other_name, r.sentiment AS sentiment
        """
        lines: list[str] = [f"## Active Scene: {location}"]
        seen_chars: set[str] = set()

        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(query, {"location": location, "branch_id": branch_id}):
                char = r["char_name"]
                if char not in seen_chars:
                    traits = r["traits"] or []
                    lines.append(
                        f"- {char} ({r['status']}, {r['alignment'] or 'unaligned'})"
                        f"{', traits: ' + ', '.join(traits) if traits else ''}"
                    )
                    seen_chars.add(char)
                if r["other_name"] and r["sentiment"]:
                    lines.append(
                        f"  → {r['sentiment']} relationship with {r['other_name']}"
                    )

        if not seen_chars:
            lines.append(f"- (no living characters at {location})")

        loc_query = """
        MATCH (l:Location {name: $location})
        RETURN l.description_summary AS desc, l.accessible AS accessible
        """
        with self._gc._driver.session(database=self._gc._database) as session:
            loc_result = session.run(loc_query, {"location": location}).single()
            if loc_result and loc_result["desc"]:
                lines.append(f"Setting: {loc_result['desc']}")
            if loc_result and not loc_result["accessible"]:
                lines.append(f"WARNING: {location} is currently inaccessible")

        return "\n".join(lines)

    def t2_causal_chain(self, last_seq_id: int, branch_id: str) -> str:
        """T2: Causal ancestry of recent events via CAUSED_BY traversal.

        Included if budget remains after T1.

        Args:
            last_seq_id: Most recent event sequence ID.
            branch_id: Active branch ID.

        Returns:
            Formatted causal chain string.
        """
        query = """
        MATCH (e:Event {branch_id: $branch_id})
        WHERE e.seq_id >= $min_seq
        OPTIONAL MATCH path = (e)-[:CAUSED_BY*1..3]->(cause:Event)
        RETURN e.seq_id AS seq_id, e.description AS desc, e.outcome AS outcome,
               cause.seq_id AS cause_seq, cause.description AS cause_desc
        ORDER BY e.seq_id DESC
        """
        lines: list[str] = ["## Causal Chain (recent events)"]
        seen_events: set[int] = set()

        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(query, {
                "branch_id": branch_id,
                "min_seq": max(0, last_seq_id - 3),
            }):
                sid = r["seq_id"]
                if sid not in seen_events:
                    outcome = f" → outcome: {r['outcome']}" if r["outcome"] else ""
                    lines.append(f"- Event #{sid}: {r['desc']}{outcome}")
                    seen_events.add(sid)
                if r["cause_seq"] and r["cause_seq"] not in seen_events:
                    lines.append(f"  ← caused by Event #{r['cause_seq']}: {r['cause_desc']}")
                    seen_events.add(r["cause_seq"])

        if len(lines) == 1:
            lines.append("- (no recent events)")

        return "\n".join(lines)

    def t3_hostile_tensions(self, present_chars: list[str], branch_id: str) -> str:
        """T3: Unresolved hostile relationships between present characters.

        Included if budget remains after T1+T2.

        Args:
            present_chars: Names of characters present in the current scene.
            branch_id: Active branch ID.

        Returns:
            Formatted tension facts string.
        """
        if not present_chars:
            return ""

        query = """
        MATCH (a:Character)-[r:KNOWS {sentiment: 'hostile'}]-(b:Character)
        WHERE a.name IN $present_chars AND b.name IN $present_chars
              AND a.name < b.name
        OPTIONAL MATCH (a)-[:PARTICIPATED_IN]->(e:Event)
        WHERE e.branch_id = $branch_id
        WITH a, b, r, max(e.seq_id) AS last_event
        RETURN a.name AS a_name, b.name AS b_name,
               r.since_event_id AS since_event, last_event
        """
        lines: list[str] = ["## Hostile Tensions"]
        found = False

        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(query, {
                "present_chars": present_chars,
                "branch_id": branch_id,
            }):
                lines.append(
                    f"- {r['a_name']} and {r['b_name']} are hostile"
                    f" (since event #{r['since_event'] or '?'})"
                )
                found = True

        return "\n".join(lines) if found else ""

    def t4_orphan_nodes(self, branch_id: str, current_seq_id: int) -> str:
        """T4: Dormant characters/objects with no recent participation.

        Story hints for characters the narrative has forgotten.
        Included only if significant budget remains after T1-T3.

        Args:
            branch_id: Active branch ID.
            current_seq_id: Current segment sequence ID.

        Returns:
            Formatted orphan hints string.
        """
        recency_threshold = max(1, current_seq_id - 5)
        query = """
        MATCH (c:Character)
        WHERE c.status = 'alive'
              AND (c.branch_id = $branch_id OR c.branch_id IS NULL)
        OPTIONAL MATCH (c)-[:PARTICIPATED_IN]->(e:Event {branch_id: $branch_id})
        WITH c, max(e.seq_id) AS last_event
        WHERE last_event IS NULL OR last_event < $threshold
        RETURN c.name AS name, c.current_location_id AS location, last_event
        LIMIT 3
        """
        lines: list[str] = ["## Story Hints (dormant characters)"]
        found = False

        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(query, {
                "branch_id": branch_id,
                "threshold": recency_threshold,
            }):
                loc = f" (last seen at {r['location']})" if r["location"] else ""
                last = f", last active at event #{r['last_event']}" if r["last_event"] else ", never participated in events"
                lines.append(f"- {r['name']}{loc}{last}")
                found = True

        orphan_objects_query = """
        MATCH (o:Object)
        WHERE (o.branch_id = $branch_id OR o.branch_id IS NULL)
        OPTIONAL MATCH (c:Character)-[:OWNS]->(o)
        WITH o, c
        WHERE c IS NULL OR c.status = 'dead'
        RETURN o.name AS name, o.significance AS significance
        LIMIT 2
        """
        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(orphan_objects_query, {"branch_id": branch_id}):
                sig = f" ({r['significance']})" if r["significance"] else ""
                lines.append(f"- Unclaimed object: {r['name']}{sig}")
                found = True

        return "\n".join(lines) if found else ""

    def retrieve(self, session: SessionState) -> tuple[str, int]:
        """Execute tiered retrieval within the token budget.

        T1 is always included. T2-T4 are added in priority order until
        the budget is exhausted. Each tier is atomic — partially included
        tiers would lose structural coherence.

        Args:
            session: Current SessionState with location, branch, seq_id.

        Returns:
            Tuple of (assembled graph context string, token count used).
        """
        remaining = self._budget
        sections: list[str] = []

        t1 = self.t1_active_scene(session.current_location, session.active_branch_id)
        t1_tokens = count_tokens(t1)
        sections.append(t1)
        remaining -= t1_tokens
        logger.info("T1 active scene: %d tokens (budget remaining: %d)", t1_tokens, remaining)

        if remaining > 50:
            t2 = self.t2_causal_chain(session.last_segment_seq_id, session.active_branch_id)
            t2_tokens = count_tokens(t2)
            if t2_tokens <= remaining:
                sections.append(t2)
                remaining -= t2_tokens
                logger.info("T2 causal chain: %d tokens (remaining: %d)", t2_tokens, remaining)
            else:
                logger.info("T2 skipped: %d tokens exceeds remaining %d", t2_tokens, remaining)

        if remaining > 50:
            t3 = self.t3_hostile_tensions(session.present_characters, session.active_branch_id)
            if t3:
                t3_tokens = count_tokens(t3)
                if t3_tokens <= remaining:
                    sections.append(t3)
                    remaining -= t3_tokens
                    logger.info("T3 hostile tensions: %d tokens (remaining: %d)", t3_tokens, remaining)

        if remaining > 100:
            t4 = self.t4_orphan_nodes(session.active_branch_id, session.last_segment_seq_id)
            if t4:
                t4_tokens = count_tokens(t4)
                if t4_tokens <= remaining:
                    sections.append(t4)
                    remaining -= t4_tokens
                    logger.info("T4 orphan hints: %d tokens (remaining: %d)", t4_tokens, remaining)

        assembled = "\n\n".join(sections)
        used = self._budget - remaining
        return assembled, used


# ---------------------------------------------------------------------------
# Vector Retrieval — ChromaDB semantic search for tonal anchors
# ---------------------------------------------------------------------------


class VectorRetriever:
    """ChromaDB-backed semantic retrieval for tonal anchors.

    Stores segment text with embeddings and retrieves the most
    semantically similar past segments for style continuity.

    Args:
        persist_dir: ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "segments",
    ) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_store/")
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._embed_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorRetriever initialized: collection='%s', model='%s', docs=%d",
            collection_name, model_name, self._collection.count(),
        )

    def add_segment(self, segment_text: str, segment_id: str) -> None:
        """Embed and store a segment in ChromaDB.

        Uses upsert to handle re-runs idempotently.

        Args:
            segment_text: The full segment text.
            segment_id: Unique identifier (e.g. "main_seq_3").
        """
        self._collection.upsert(
            ids=[segment_id],
            documents=[segment_text],
        )

    def search_similar(self, query_text: str, n_results: int = 3) -> list[str]:
        """Find the most semantically similar past segments.

        Args:
            query_text: The text to find similar segments for.
            n_results: Number of results to return.

        Returns:
            List of segment text strings, most similar first.
        """
        if self._collection.count() == 0:
            return []

        n = min(n_results, self._collection.count())
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n,
        )
        docs = results.get("documents", [[]])[0]
        return docs

    def retrieve(
        self,
        last_segments: list[str],
        token_budget: int = 1000,
        exclude_text: str | None = None,
    ) -> tuple[str, int]:
        """Retrieve tonal anchors within token budget.

        Combines the last few segments as a query to find stylistically
        similar past segments. Results are filtered to remove near-duplicates
        of the most recent segment (preventing self-parroting) and truncated
        to fit the budget.

        Args:
            last_segments: Recent segment texts to use as query.
            token_budget: Maximum tokens for tonal context.
            exclude_text: If provided, skip any result with >80% overlap with
                this text (prevents regurgitation of the previous segment).

        Returns:
            Tuple of (formatted tonal context string, token count used).
        """
        if not last_segments or self._collection.count() == 0:
            return "", 0

        query = " ".join(last_segments[-3:])
        similar = self.search_similar(query, n_results=5)

        lines: list[str] = []
        used = 0
        for i, seg in enumerate(similar):
            if exclude_text and _text_overlap_ratio(seg, exclude_text) > 0.8:
                logger.debug("Skipping tonal anchor %d: too similar to exclude_text", i)
                continue
            seg_tokens = count_tokens(seg)
            if used + seg_tokens > token_budget:
                break
            lines.append(f"[Tonal anchor {i + 1}]: {seg}")
            used += seg_tokens

        return "\n\n".join(lines), used

    def reset(self) -> None:
        """Delete all documents from the collection. For testing only."""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)


# ---------------------------------------------------------------------------
# Context Assembler — merges graph + vector context into prompt sections
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Merges graph context, vector context, guard violations, and persona
    documents into the structured prompt sections defined in Section 7.1.

    The key architectural distinction: graph context = facts to honour,
    vector context = tone to match. These are passed in separate prompt
    sections with different instruction framing.
    """

    @staticmethod
    def assemble(
        graph_context: str,
        vector_context: str,
        violations: list[ConstraintViolation] | None = None,
        persona_docs: list[str] | None = None,
        session: SessionState | None = None,
    ) -> dict[str, str]:
        """Build the structured prompt sections for the generation call.

        Args:
            graph_context: Assembled graph facts from CypherRetriever.
            vector_context: Tonal anchors from VectorRetriever.
            violations: Constraint violations from the guard (if any).
            persona_docs: Character persona documents (if available).
            session: Current session state.

        Returns:
            Dict with keys matching Section 7.1 prompt structure:
            'known_facts', 'constraints', 'character_voices', 'tonal_context',
            'previous_segment', 'player_action'.
        """
        sections: dict[str, str] = {}

        sections["known_facts"] = graph_context if graph_context else "(no graph context available)"

        if violations:
            constraint_lines = []
            for v in violations:
                constraint_lines.append(
                    f"[{v.severity.upper()}] {v.check_name}: {v.violation_message}"
                )
            sections["constraints"] = "\n".join(constraint_lines)
        else:
            sections["constraints"] = "(no constraint violations detected)"

        if persona_docs:
            sections["character_voices"] = "\n\n".join(persona_docs)
        else:
            sections["character_voices"] = "(no persona documents loaded yet)"

        sections["tonal_context"] = vector_context if vector_context else "(no tonal anchors available)"

        if session:
            sections["previous_segment"] = session.last_segment_text or "(story beginning)"
        else:
            sections["previous_segment"] = "(story beginning)"

        return sections
