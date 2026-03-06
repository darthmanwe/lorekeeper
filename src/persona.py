"""Persona store for character voice consistency.

Each character gets a persona document containing voice descriptor, emotional
baseline, speech mannerisms, and knowledge boundaries. These documents are
embedded in ChromaDB and retrieved during context assembly to populate the
CHARACTER VOICES section of the generation prompt.

The graph enforces *what happened*; the persona document enforces
*how each character speaks about it*.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PersonaDocument(BaseModel):
    """Structured persona document for a single character.

    Stored as the ChromaDB document text (serialized to a readable format)
    and retrieved by character name.
    """

    character_name: str
    voice_descriptor: str = Field(
        description="Formal/informal, verbose/terse, archaic/modern vocabulary tendency"
    )
    emotional_baseline: str = Field(
        description="Default emotional register: stoic, anxious, sardonic, etc."
    )
    speech_mannerisms: list[str] = Field(
        default_factory=list,
        description="Recurring phrases, punctuation style, metaphor domains",
    )
    knowledge_boundaries: list[str] = Field(
        default_factory=list,
        description="Key facts the character knows or doesn't know as of latest segment",
    )
    alignment_notes: str = Field(
        default="",
        description="How alignment influences speech patterns",
    )

    def to_prompt_text(self) -> str:
        """Serialize to the format injected into the CHARACTER VOICES prompt section.

        The output is human-readable and structured for LLM consumption,
        not JSON — the LLM should treat this as character direction, not data.
        """
        mannerisms = (
            ", ".join(self.speech_mannerisms) if self.speech_mannerisms
            else "no distinctive mannerisms noted"
        )
        knowledge = (
            "; ".join(self.knowledge_boundaries) if self.knowledge_boundaries
            else "no specific knowledge boundaries noted"
        )
        return (
            f"[{self.character_name}]\n"
            f"Voice: {self.voice_descriptor}\n"
            f"Emotional register: {self.emotional_baseline}\n"
            f"Mannerisms: {mannerisms}\n"
            f"Alignment influence: {self.alignment_notes or 'neutral'}\n"
            f"Knowledge boundaries: {knowledge}"
        )

    @classmethod
    def from_llm_dict(cls, data: dict[str, Any]) -> PersonaDocument:
        """Construct from an LLM-returned dict with flexible key handling."""
        return cls(
            character_name=data.get("character_name", data.get("name", "")),
            voice_descriptor=data.get("voice_descriptor", data.get("voice", "")),
            emotional_baseline=data.get(
                "emotional_baseline", data.get("emotional_register", "")
            ),
            speech_mannerisms=data.get(
                "speech_mannerisms", data.get("mannerisms", [])
            ),
            knowledge_boundaries=data.get("knowledge_boundaries", []),
            alignment_notes=data.get("alignment_notes", data.get("alignment", "")),
        )


class PersonaStore:
    """ChromaDB-backed store for character persona documents.

    Persona documents are embedded using the same sentence-transformer model
    as the segment vector store, enabling semantic queries like "find characters
    who speak formally" in addition to exact name lookups.

    Args:
        persist_dir: ChromaDB persistence directory.
        collection_name: Name of the personas collection.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "personas",
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
            "PersonaStore initialized: collection='%s', docs=%d",
            collection_name, self._collection.count(),
        )

    def upsert_persona(self, persona: PersonaDocument) -> str:
        """Store or update a persona document in ChromaDB.

        The document ID is the character name (lowercased, spaces replaced).
        Uses upsert for idempotency.

        Args:
            persona: Validated PersonaDocument instance.

        Returns:
            The document ID used for storage.
        """
        doc_id = self._name_to_id(persona.character_name)
        self._collection.upsert(
            ids=[doc_id],
            documents=[persona.to_prompt_text()],
            metadatas=[{
                "character_name": persona.character_name,
                "voice_descriptor": persona.voice_descriptor,
                "emotional_baseline": persona.emotional_baseline,
            }],
        )
        logger.info("Upserted persona for '%s' (id=%s)", persona.character_name, doc_id)
        return doc_id

    def get_persona(self, character_name: str) -> str | None:
        """Retrieve a persona document by exact character name.

        Args:
            character_name: The character's name.

        Returns:
            The persona prompt text, or None if not found.
        """
        doc_id = self._name_to_id(character_name)
        try:
            result = self._collection.get(ids=[doc_id])
            docs = result.get("documents", [])
            if docs and docs[0]:
                return docs[0]
        except Exception:
            pass
        return None

    def get_personas_for_characters(self, character_names: list[str]) -> list[str]:
        """Retrieve persona documents for a list of characters.

        Characters without personas are silently skipped. This is the
        primary method called by the pipeline during context assembly.

        Args:
            character_names: List of character names to retrieve personas for.

        Returns:
            List of persona prompt text strings (one per character found).
        """
        if not character_names:
            return []

        docs: list[str] = []
        for name in character_names:
            persona_text = self.get_persona(name)
            if persona_text:
                docs.append(persona_text)
            else:
                logger.debug("No persona found for '%s'", name)
        return docs

    def count(self) -> int:
        """Return the number of persona documents in the store."""
        return self._collection.count()

    def list_characters(self) -> list[str]:
        """Return all character names with stored personas."""
        result = self._collection.get()
        metadatas = result.get("metadatas", [])
        return [m.get("character_name", "") for m in metadatas if m]

    def delete_persona(self, character_name: str) -> None:
        """Remove a persona document. For testing only."""
        doc_id = self._name_to_id(character_name)
        try:
            self._collection.delete(ids=[doc_id])
        except Exception:
            pass

    def reset(self) -> None:
        """Delete all persona documents. For testing only."""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)

    @staticmethod
    def _name_to_id(name: str) -> str:
        """Convert a character name to a stable ChromaDB document ID."""
        return name.lower().strip().replace(" ", "_")


# ---------------------------------------------------------------------------
# LLM-based persona generation
# ---------------------------------------------------------------------------


class PersonaGenerator:
    """Generates persona documents from character traits and graph state.

    Uses the LLM to synthesize a structured persona document from the
    character's traits, alignment, relationships, and event participation.

    Args:
        llm: ChatAnthropic instance.
        graph_client: GraphClient for querying character context.
    """

    def __init__(self, llm: Any, graph_client: Any) -> None:
        self._llm = llm
        self._gc = graph_client

    def generate(
        self,
        character_name: str,
        branch_id: str,
        additional_context: str = "",
    ) -> PersonaDocument:
        """Generate a persona document for a character using LLM + graph context.

        Queries the graph for the character's traits, relationships, and
        event participation, then asks the LLM to synthesize a persona.
        Uses the centralized persona_generation_v1 prompt template for
        governance traceability.

        Args:
            character_name: Name of the character.
            branch_id: Active branch ID for graph queries.
            additional_context: Optional extra context (e.g., from recent segments).

        Returns:
            A validated PersonaDocument instance.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.prompts import get_prompt

        char_context = self._get_character_context(character_name, branch_id)

        prompt = get_prompt("persona_generation_v1")
        system_msg = prompt.format_system()

        user_context = char_context
        if additional_context:
            user_context += f"\n\nAdditional context:\n{additional_context}"
        user_msg = prompt.format_user(character_context=user_context)

        response = self._llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ])

        return self._parse_response(response.content, character_name)

    def _get_character_context(self, character_name: str, branch_id: str) -> str:
        """Query the graph for character context used in persona generation."""
        lines: list[str] = []

        char_query = """
        MATCH (c:Character {name: $name})
        WHERE c.branch_id = $branch_id OR c.branch_id IS NULL
        RETURN c.name AS name, c.status AS status, c.alignment AS alignment,
               c.traits AS traits, c.current_location_id AS location
        """
        with self._gc._driver.session(database=self._gc._database) as session:
            result = session.run(char_query, {
                "name": character_name,
                "branch_id": branch_id,
            })
            record = result.single()
            if record:
                traits = record["traits"] or []
                lines.append(f"Name: {record['name']}")
                lines.append(f"Status: {record['status']}")
                lines.append(f"Alignment: {record['alignment'] or 'unaligned'}")
                lines.append(f"Traits: {', '.join(traits) if traits else 'none specified'}")
                lines.append(f"Current location: {record['location'] or 'unknown'}")

        rel_query = """
        MATCH (c:Character {name: $name})-[r:KNOWS]-(other:Character)
        WHERE r.branch_id = $branch_id OR r.branch_id IS NULL
        RETURN other.name AS other_name, r.sentiment AS sentiment
        """
        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(rel_query, {
                "name": character_name,
                "branch_id": branch_id,
            }):
                lines.append(f"Relationship: {r['sentiment']} with {r['other_name']}")

        event_query = """
        MATCH (c:Character {name: $name})-[:PARTICIPATED_IN]->(e:Event {branch_id: $branch_id})
        RETURN e.description AS desc, e.outcome AS outcome
        ORDER BY e.seq_id DESC
        LIMIT 5
        """
        with self._gc._driver.session(database=self._gc._database) as session:
            events = []
            for r in session.run(event_query, {
                "name": character_name,
                "branch_id": branch_id,
            }):
                outcome = f" (outcome: {r['outcome']})" if r["outcome"] else ""
                events.append(f"- {r['desc']}{outcome}")
            if events:
                lines.append(f"Recent events:\n" + "\n".join(events))

        return "\n".join(lines) if lines else f"Name: {character_name}\n(no graph data available)"

    def _parse_response(self, content: str, fallback_name: str) -> PersonaDocument:
        """Parse LLM response into a PersonaDocument with robust error handling."""
        import json
        import re

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

        brace_start = text.find("{")
        if brace_start != -1:
            text = text[brace_start:]

        brace_depth = 0
        last_brace = 0
        for i, ch in enumerate(text):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    last_brace = i
                    break
        if last_brace > 0:
            text = text[: last_brace + 1]

        try:
            data = json.loads(text)
            if "character_name" not in data:
                data["character_name"] = fallback_name
            return PersonaDocument.from_llm_dict(data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "Failed to parse persona response for '%s': %s. Using fallback.",
                fallback_name, exc,
            )
            return PersonaDocument(
                character_name=fallback_name,
                voice_descriptor="neutral, conversational",
                emotional_baseline="measured",
                speech_mannerisms=[],
                knowledge_boundaries=[],
                alignment_notes="",
            )
