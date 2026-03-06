"""Unit tests for persona.py — PersonaDocument, PersonaStore, PersonaGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.persona import PersonaDocument, PersonaGenerator, PersonaStore


# ---------------------------------------------------------------------------
# PersonaDocument
# ---------------------------------------------------------------------------


class TestPersonaDocument:
    def test_to_prompt_text_contains_all_sections(self) -> None:
        doc = PersonaDocument(
            character_name="Kael",
            voice_descriptor="terse, informal, soldier's vocabulary",
            emotional_baseline="stoic with flashes of anger",
            speech_mannerisms=["clipped sentences", "avoids metaphor"],
            knowledge_boundaries=["knows about the ambush", "does not know Elara left"],
            alignment_notes="lawful tendencies influence formal address with authority",
        )
        text = doc.to_prompt_text()
        assert "[Kael]" in text
        assert "terse, informal" in text
        assert "stoic with flashes" in text
        assert "clipped sentences" in text
        assert "knows about the ambush" in text
        assert "lawful tendencies" in text

    def test_to_prompt_text_handles_empty_fields(self) -> None:
        doc = PersonaDocument(
            character_name="Mystery",
            voice_descriptor="unknown",
            emotional_baseline="flat",
        )
        text = doc.to_prompt_text()
        assert "[Mystery]" in text
        assert "no distinctive mannerisms noted" in text
        assert "no specific knowledge boundaries noted" in text
        assert "neutral" in text

    def test_from_llm_dict_standard_keys(self) -> None:
        data = {
            "character_name": "Aria",
            "voice_descriptor": "formal, archaic",
            "emotional_baseline": "anxious",
            "speech_mannerisms": ["speaks in riddles"],
            "knowledge_boundaries": ["knows the prophecy"],
            "alignment_notes": "chaotic good",
        }
        doc = PersonaDocument.from_llm_dict(data)
        assert doc.character_name == "Aria"
        assert doc.voice_descriptor == "formal, archaic"
        assert len(doc.speech_mannerisms) == 1

    def test_from_llm_dict_alternative_keys(self) -> None:
        data = {
            "name": "Maren",
            "voice": "gruff, direct",
            "emotional_register": "sardonic",
            "mannerisms": ["laughs before threats"],
        }
        doc = PersonaDocument.from_llm_dict(data)
        assert doc.character_name == "Maren"
        assert doc.voice_descriptor == "gruff, direct"
        assert doc.emotional_baseline == "sardonic"

    def test_from_llm_dict_missing_keys_defaults(self) -> None:
        data = {"character_name": "Ghost"}
        doc = PersonaDocument.from_llm_dict(data)
        assert doc.character_name == "Ghost"
        assert doc.voice_descriptor == ""
        assert doc.speech_mannerisms == []


# ---------------------------------------------------------------------------
# PersonaStore (with temp directory)
# ---------------------------------------------------------------------------


class TestPersonaStore:
    @pytest.fixture()
    def store(self, tmp_path) -> PersonaStore:
        return PersonaStore(persist_dir=str(tmp_path / "persona_store"))

    def test_upsert_and_get(self, store: PersonaStore) -> None:
        doc = PersonaDocument(
            character_name="Kael",
            voice_descriptor="terse, military",
            emotional_baseline="stoic",
            speech_mannerisms=["sir/ma'am usage"],
        )
        doc_id = store.upsert_persona(doc)
        assert doc_id == "kael"

        retrieved = store.get_persona("Kael")
        assert retrieved is not None
        assert "[Kael]" in retrieved
        assert "terse, military" in retrieved

    def test_get_nonexistent_returns_none(self, store: PersonaStore) -> None:
        assert store.get_persona("Nobody") is None

    def test_get_personas_for_characters(self, store: PersonaStore) -> None:
        for name, voice in [("Aria", "formal"), ("Kael", "gruff"), ("Maren", "cunning")]:
            store.upsert_persona(PersonaDocument(
                character_name=name,
                voice_descriptor=voice,
                emotional_baseline="neutral",
            ))

        docs = store.get_personas_for_characters(["Aria", "Kael", "Unknown"])
        assert len(docs) == 2
        assert any("[Aria]" in d for d in docs)
        assert any("[Kael]" in d for d in docs)

    def test_upsert_is_idempotent(self, store: PersonaStore) -> None:
        doc = PersonaDocument(
            character_name="Aria",
            voice_descriptor="formal",
            emotional_baseline="anxious",
        )
        store.upsert_persona(doc)
        store.upsert_persona(doc)
        assert store.count() == 1

    def test_upsert_updates_content(self, store: PersonaStore) -> None:
        doc_v1 = PersonaDocument(
            character_name="Aria",
            voice_descriptor="formal v1",
            emotional_baseline="anxious",
        )
        store.upsert_persona(doc_v1)

        doc_v2 = PersonaDocument(
            character_name="Aria",
            voice_descriptor="informal v2",
            emotional_baseline="calm",
        )
        store.upsert_persona(doc_v2)

        retrieved = store.get_persona("Aria")
        assert "informal v2" in retrieved
        assert store.count() == 1

    def test_list_characters(self, store: PersonaStore) -> None:
        for name in ["Aria", "Kael"]:
            store.upsert_persona(PersonaDocument(
                character_name=name,
                voice_descriptor="test",
                emotional_baseline="test",
            ))
        names = store.list_characters()
        assert set(names) == {"Aria", "Kael"}

    def test_delete_persona(self, store: PersonaStore) -> None:
        store.upsert_persona(PersonaDocument(
            character_name="Kael",
            voice_descriptor="test",
            emotional_baseline="test",
        ))
        assert store.count() == 1
        store.delete_persona("Kael")
        assert store.count() == 0

    def test_reset_clears_all(self, store: PersonaStore) -> None:
        for name in ["A", "B", "C"]:
            store.upsert_persona(PersonaDocument(
                character_name=name,
                voice_descriptor="test",
                emotional_baseline="test",
            ))
        assert store.count() == 3
        store.reset()
        assert store.count() == 0

    def test_name_to_id_normalization(self) -> None:
        assert PersonaStore._name_to_id("Kael the Brave") == "kael_the_brave"
        assert PersonaStore._name_to_id("  Aria  ") == "aria"
        assert PersonaStore._name_to_id("MAREN") == "maren"


# ---------------------------------------------------------------------------
# PersonaGenerator (mocked LLM)
# ---------------------------------------------------------------------------


class TestPersonaGenerator:
    def _mock_gc(self) -> MagicMock:
        gc = MagicMock()
        session = MagicMock()

        char_result = MagicMock()
        char_result.single.return_value = {
            "name": "Kael",
            "status": "alive",
            "alignment": "lawful neutral",
            "traits": ["brave", "stubborn"],
            "location": "Iron Tavern",
        }

        rel_result = MagicMock()
        rel_result.__iter__ = MagicMock(return_value=iter([
            {"other_name": "Maren", "sentiment": "hostile"},
        ]))

        event_result = MagicMock()
        event_result.__iter__ = MagicMock(return_value=iter([
            {"desc": "Planned journey north", "outcome": "alliance_formed"},
        ]))

        call_count = {"n": 0}
        results = [char_result, rel_result, event_result]

        def run_side(*args, **kwargs):
            idx = min(call_count["n"], len(results) - 1)
            call_count["n"] += 1
            return results[idx]

        session.run.side_effect = run_side
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        gc._driver.session.return_value = session
        gc._database = "neo4j"
        return gc

    def test_parse_response_valid_json(self) -> None:
        llm = MagicMock()
        gc = self._mock_gc()
        gen = PersonaGenerator(llm, gc)

        content = '{"character_name": "Kael", "voice_descriptor": "terse", "emotional_baseline": "stoic", "speech_mannerisms": ["clipped"], "knowledge_boundaries": [], "alignment_notes": "lawful"}'
        doc = gen._parse_response(content, "Kael")
        assert doc.character_name == "Kael"
        assert doc.voice_descriptor == "terse"

    def test_parse_response_markdown_fenced(self) -> None:
        llm = MagicMock()
        gc = self._mock_gc()
        gen = PersonaGenerator(llm, gc)

        content = '```json\n{"character_name": "Aria", "voice_descriptor": "formal", "emotional_baseline": "anxious"}\n```'
        doc = gen._parse_response(content, "Aria")
        assert doc.character_name == "Aria"

    def test_parse_response_malformed_returns_fallback(self) -> None:
        llm = MagicMock()
        gc = self._mock_gc()
        gen = PersonaGenerator(llm, gc)

        content = "This is not JSON at all."
        doc = gen._parse_response(content, "FallbackChar")
        assert doc.character_name == "FallbackChar"
        assert doc.voice_descriptor == "neutral, conversational"

    def test_get_character_context_builds_string(self) -> None:
        llm = MagicMock()
        gc = self._mock_gc()
        gen = PersonaGenerator(llm, gc)

        ctx = gen._get_character_context("Kael", "main")
        assert "Kael" in ctx
        assert "alive" in ctx
