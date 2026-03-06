"""Evaluation harness for paired NKGE vs. baseline comparison.

Implements:
- LLM-judged contradiction detection (Section 5.3 of design doc)
- LLM-judged coherence scoring (Section 5.4)
- Graph coverage rate computation
- Retrieval precision estimation
- Multi-segment story runner with full artifact capture
- Paired comparison reporting

The headline metric is Contradiction Rate Improvement %:
  (Baseline Score − NKGE Score) / Baseline Score × 100
Lower contradiction score is better.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 3.0,
    "major": 2.0,
    "minor": 1.0,
    "soft": 0.5,
}


# ---------------------------------------------------------------------------
# LLM Judge — Contradiction Detection
# ---------------------------------------------------------------------------

JUDGE_CONTRADICTION_SYSTEM = (
    "You are a narrative consistency auditor. You will be given:\n"
    "  (1) A list of established facts extracted from prior story segments.\n"
    "  (2) A newly generated story segment.\n\n"
    "Your task: identify every statement in the new segment that contradicts "
    "an established fact.\n\n"
    "For each contradiction found, return a JSON object with:\n"
    "  - contradiction_text: exact quote from the new segment\n"
    "  - conflicting_fact: the established fact it violates\n"
    "  - severity: critical | major | minor | soft\n"
    "  - reasoning: one sentence explaining the conflict\n\n"
    "Severity guide:\n"
    "  - critical: dead character treated as alive, destroyed location still exists\n"
    "  - major: object ownership wrong, character in impossible location\n"
    "  - minor: character visits place they've never been without explanation\n"
    "  - soft: character knows something they shouldn't\n\n"
    "If no contradictions exist, return an empty array [].\n"
    "Do not invent contradictions. Only flag direct factual conflicts, "
    "not stylistic inconsistencies.\n"
    "Return ONLY the JSON array, no other text."
)


def get_established_facts(graph_client: Any, branch_id: str) -> str:
    """Extract structured facts from the graph for the LLM judge.

    Uses a summary Cypher query rather than raw segment text, ensuring
    the judge evaluates against structured ground truth.
    """
    facts: list[str] = []

    queries = [
        (
            "Characters",
            """
            MATCH (c:Character)
            WHERE c.branch_id = $branch_id OR c.branch_id IS NULL
            RETURN c.name AS name, c.status AS status, c.alignment AS alignment,
                   c.current_location_id AS location
            """,
            lambda r: (
                f"Character '{r['name']}' is {r['status']}"
                + (f", aligned {r['alignment']}" if r['alignment'] else "")
                + (f", located at {r['location']}" if r['location'] else "")
            ),
        ),
        (
            "Relationships",
            """
            MATCH (a:Character)-[r:KNOWS]->(b:Character)
            WHERE r.branch_id = $branch_id OR r.branch_id IS NULL
            RETURN a.name AS from_char, b.name AS to_char, r.sentiment AS sentiment
            """,
            lambda r: f"'{r['from_char']}' has {r['sentiment']} sentiment toward '{r['to_char']}'",
        ),
        (
            "Object Ownership",
            """
            MATCH (c:Character)-[:OWNS]->(o:Object)
            WHERE c.branch_id = $branch_id OR c.branch_id IS NULL
                  OR o.branch_id = $branch_id OR o.branch_id IS NULL
            RETURN c.name AS owner, o.name AS object_name
            """,
            lambda r: f"'{r['owner']}' owns '{r['object_name']}'",
        ),
        (
            "Locations",
            """
            MATCH (l:Location)
            WHERE l.branch_id = $branch_id OR l.branch_id IS NULL
            RETURN l.name AS name, l.terrain AS terrain,
                   l.accessible AS accessible
            """,
            lambda r: (
                f"Location '{r['name']}' ({r['terrain'] or 'unspecified terrain'})"
                + (", inaccessible" if r.get("accessible") is False else "")
            ),
        ),
        (
            "Events",
            """
            MATCH (e:Event {branch_id: $branch_id})
            RETURN e.seq_id AS seq_id, e.description AS desc, e.outcome AS outcome
            ORDER BY e.seq_id
            """,
            lambda r: (
                f"Event #{r['seq_id']}: {r['desc']}"
                + (f" (outcome: {r['outcome']})" if r['outcome'] else "")
            ),
        ),
    ]

    with graph_client._driver.session(database=graph_client._database) as session:
        for category, query, formatter in queries:
            result = session.run(query, {"branch_id": branch_id})
            records = list(result)
            if records:
                facts.append(f"=== {category} ===")
                for rec in records:
                    facts.append(f"- {formatter(rec)}")

    return "\n".join(facts) if facts else "(no established facts yet)"


def judge_contradictions(
    llm: Any,
    established_facts: str,
    generated_text: str,
) -> list[dict[str, Any]]:
    """Run the LLM contradiction judge on a generated segment.

    Args:
        llm: ChatAnthropic instance for judging.
        established_facts: Structured facts from graph (not raw prose).
        generated_text: The segment text to evaluate.

    Returns:
        List of contradiction dicts with keys: contradiction_text,
        conflicting_fact, severity, reasoning.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    user_msg = (
        f"ESTABLISHED FACTS:\n{established_facts}\n\n"
        f"NEW SEGMENT:\n{generated_text}"
    )

    response = llm.invoke([
        SystemMessage(content=JUDGE_CONTRADICTION_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    return _parse_judge_response(response.content)


# ---------------------------------------------------------------------------
# LLM Judge — Coherence Scoring
# ---------------------------------------------------------------------------

JUDGE_COHERENCE_SYSTEM = (
    "You are a narrative quality assessor. Rate the following story segment "
    "on a scale of 1-5 for narrative coherence and logical continuity.\n\n"
    "Scoring guide:\n"
    "  5: Excellent flow, all events logically follow, strong cause-effect chains\n"
    "  4: Good flow with minor awkwardness, all events make sense\n"
    "  3: Adequate but with noticeable jumps or unclear transitions\n"
    "  2: Poor flow with confusing sequences or unmotivated actions\n"
    "  1: Incoherent, events contradict each other within the segment itself\n\n"
    "Consider the previous segment for continuity context.\n"
    "Return ONLY a JSON object: {\"score\": <1-5>, \"reasoning\": \"<one sentence>\"}"
)


def judge_coherence(
    llm: Any,
    generated_text: str,
    previous_text: str = "",
) -> tuple[float, str]:
    """Run the LLM coherence judge on a generated segment.

    Args:
        llm: ChatAnthropic instance for judging.
        generated_text: The segment text to evaluate.
        previous_text: The previous segment for continuity context.

    Returns:
        Tuple of (score 1.0-5.0, reasoning string).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    user_msg = ""
    if previous_text:
        user_msg += f"PREVIOUS SEGMENT:\n{previous_text}\n\n"
    user_msg += f"SEGMENT TO EVALUATE:\n{generated_text}"

    response = llm.invoke([
        SystemMessage(content=JUDGE_COHERENCE_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    try:
        data = _parse_json_safe(response.content)
        score = float(data.get("score", 3.0))
        score = max(1.0, min(5.0, score))
        reasoning = data.get("reasoning", "")
        return score, reasoning
    except Exception:
        logger.warning("Coherence judge parse failed, defaulting to 3.0")
        return 3.0, "parse error"


# ---------------------------------------------------------------------------
# Secondary Metrics
# ---------------------------------------------------------------------------


def compute_graph_coverage(
    graph_client: Any,
    generated_text: str,
    branch_id: str,
) -> float:
    """Compute graph coverage rate: % of named entities in generated text
    that exist in the graph.

    Args:
        graph_client: GraphClient instance.
        generated_text: The generated segment text.
        branch_id: Active branch ID.

    Returns:
        Coverage rate as a float 0.0-1.0.
    """
    query = """
    MATCH (n)
    WHERE (n:Character OR n:Location OR n:Object OR n:Faction)
      AND (n.branch_id = $branch_id OR n.branch_id IS NULL)
    RETURN n.name AS name, labels(n) AS labels
    """
    with graph_client._driver.session(database=graph_client._database) as session:
        result = session.run(query, {"branch_id": branch_id})
        graph_entities = {r["name"] for r in result if r["name"]}

    if not graph_entities:
        return 0.0

    text_lower = generated_text.lower()
    mentioned = sum(1 for name in graph_entities if name.lower() in text_lower)

    entities_in_text = _extract_entity_mentions(generated_text, graph_entities)

    if not entities_in_text:
        return 0.0

    in_graph = sum(1 for e in entities_in_text if e in graph_entities)
    return in_graph / len(entities_in_text) if entities_in_text else 0.0


def compute_retrieval_precision(
    graph_context: str,
    generated_text: str,
) -> float:
    """Compute retrieval precision: % of injected graph facts that are
    actually referenced in the generated output.

    Uses entity/keyword presence checking.

    Args:
        graph_context: The graph context string injected into the prompt.
        generated_text: The generated segment text.

    Returns:
        Precision as a float 0.0-1.0.
    """
    if not graph_context or graph_context == "(no graph context)":
        return 0.0

    fact_lines = [
        line.strip()
        for line in graph_context.split("\n")
        if line.strip() and not line.strip().startswith("===")
    ]

    if not fact_lines:
        return 0.0

    text_lower = generated_text.lower()
    referenced = 0

    for fact in fact_lines:
        keywords = _extract_keywords(fact)
        if any(kw.lower() in text_lower for kw in keywords):
            referenced += 1

    return referenced / len(fact_lines)


# ---------------------------------------------------------------------------
# Story Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Runs multi-segment story generation and captures evaluation artifacts.

    Executes the same sequence of player actions in both NKGE and baseline
    modes, then scores each segment using the LLM judge.

    Args:
        graph_client: GraphClient instance.
        pipeline: StoryPipeline instance.
        judge_llm: LLM instance for judging (can be same or different from generation LLM).
        output_dir: Directory for storing evaluation artifacts.
    """

    def __init__(
        self,
        graph_client: Any,
        pipeline: Any,
        judge_llm: Any,
        output_dir: str = "./eval_runs",
    ) -> None:
        self._gc = graph_client
        self._pipeline = pipeline
        self._judge_llm = judge_llm
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run_story(
        self,
        session: Any,
        player_actions: list[str],
        run_id: str | None = None,
    ) -> "EvalRunResult":
        """Execute a full multi-segment story run and capture all metrics.

        Args:
            session: Initial SessionState.
            player_actions: Ordered list of player actions.
            run_id: Optional run identifier (auto-generated if not provided).

        Returns:
            EvalRunResult containing all segment records and summary.
        """
        from src.schema import (
            ConstraintViolation,
            ContradictionResult,
            EvalRunOutput,
            EvalRunSummary,
            ExtractionProposal,
            SegmentEvalRecord,
        )

        if run_id is None:
            run_id = f"run_{session.mode}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        logger.info("Starting eval run '%s' (mode=%s, %d actions)", run_id, session.mode, len(player_actions))

        segment_records: list[SegmentEvalRecord] = []
        previous_text = session.last_segment_text
        total_contradictions: dict[str, int] = {
            "critical": 0, "major": 0, "minor": 0, "soft": 0,
        }

        initial_nodes, initial_rels = self._get_graph_counts()

        for i, action in enumerate(player_actions):
            seq_id = session.last_segment_seq_id + i + 1
            logger.info("  Segment %d/%d: '%s'", i + 1, len(player_actions), action[:50])

            result = self._pipeline.run(session, player_action=action)

            generated_text = result.get("generated_text", "")
            graph_context = result.get("graph_context", "")
            graph_tokens = result.get("graph_context_tokens", 0)
            vector_tokens = result.get("vector_context_tokens", 0)
            violations = result.get("violations", [])
            extraction_result = result.get("extraction_result")

            guard_violations = [
                ConstraintViolation(
                    check_name=v.check_name,
                    violation_message=v.violation_message,
                    severity=v.severity,
                ) if isinstance(v, ConstraintViolation) else ConstraintViolation(**v)
                for v in violations
            ]

            extraction_proposals = []
            committed_count = 0
            if extraction_result:
                if hasattr(extraction_result, "proposals"):
                    extraction_proposals = [
                        ExtractionProposal(
                            entity_type=p.entity_type,
                            entity_name=p.entity_name,
                            confidence=p.confidence,
                            supporting_quote=p.supporting_quote,
                        ) if hasattr(p, "entity_type") else p
                        for p in extraction_result.proposals
                    ]
                    committed_count = extraction_result.committed_count

            established_facts = get_established_facts(self._gc, session.active_branch_id)

            raw_contradictions = judge_contradictions(
                self._judge_llm, established_facts, generated_text,
            )
            contradictions = []
            seg_contradiction_score = 0.0
            for c in raw_contradictions:
                severity = c.get("severity", "minor")
                if severity not in SEVERITY_WEIGHTS:
                    severity = "minor"
                cr = ContradictionResult(
                    contradiction_text=c.get("contradiction_text", ""),
                    conflicting_fact=c.get("conflicting_fact", ""),
                    severity=severity,
                    reasoning=c.get("reasoning", ""),
                )
                contradictions.append(cr)
                seg_contradiction_score += cr.weighted_score
                total_contradictions[severity] += 1

            coherence_score, _ = judge_coherence(
                self._judge_llm, generated_text, previous_text,
            )

            coverage = compute_graph_coverage(
                self._gc, generated_text, session.active_branch_id,
            )

            precision = compute_retrieval_precision(graph_context, generated_text)

            record = SegmentEvalRecord(
                seq_id=seq_id,
                player_action=action,
                generated_text=generated_text,
                graph_context_tokens=graph_tokens,
                vector_context_tokens=vector_tokens,
                guard_violations=guard_violations,
                extraction_proposals=[
                    ExtractionProposal(
                        entity_type=p.entity_type if hasattr(p, "entity_type") else p.get("entity_type", "Character"),
                        entity_name=p.entity_name if hasattr(p, "entity_name") else p.get("entity_name", ""),
                        confidence=p.confidence if hasattr(p, "confidence") else p.get("confidence", 0.0),
                        supporting_quote=p.supporting_quote if hasattr(p, "supporting_quote") else p.get("supporting_quote", ""),
                    )
                    for p in extraction_proposals
                ],
                contradictions_found=contradictions,
                contradiction_score=seg_contradiction_score,
                coherence_score=coherence_score,
                graph_coverage_rate=coverage,
                retrieval_precision=precision,
            )
            segment_records.append(record)

            session.last_segment_seq_id = seq_id
            session.last_segment_text = generated_text
            previous_text = generated_text

        final_nodes, final_rels = self._get_graph_counts()

        n_segments = len(segment_records)
        summary = EvalRunSummary(
            mean_contradiction_score=(
                sum(s.contradiction_score for s in segment_records) / n_segments
                if n_segments else 0.0
            ),
            mean_coherence_score=(
                sum(s.coherence_score for s in segment_records) / n_segments
                if n_segments else 0.0
            ),
            mean_graph_coverage=(
                sum(s.graph_coverage_rate for s in segment_records) / n_segments
                if n_segments else 0.0
            ),
            mean_retrieval_precision=(
                sum(s.retrieval_precision for s in segment_records) / n_segments
                if n_segments else 0.0
            ),
            total_nodes_created=final_nodes - initial_nodes,
            total_relationships_created=final_rels - initial_rels,
            critical_contradictions_total=total_contradictions["critical"],
            major_contradictions_total=total_contradictions["major"],
            minor_contradictions_total=total_contradictions["minor"],
            soft_contradictions_total=total_contradictions["soft"],
        )

        output = EvalRunOutput(
            run_id=run_id,
            mode=session.mode,
            story_seed=session.story_seed,
            segments=segment_records,
            summary=summary,
        )

        run_dir = self._output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "eval_output.json"
        output_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Eval run saved to %s", output_path)

        return EvalRunResult(output=output, output_path=str(output_path))

    def _get_graph_counts(self) -> tuple[int, int]:
        """Get current node and relationship counts from the graph."""
        try:
            nodes = sum(self._gc.get_node_counts().values())
            rels = sum(self._gc.get_relationship_counts().values())
            return nodes, rels
        except Exception:
            return 0, 0


class EvalRunResult:
    """Container for a completed evaluation run with analysis methods."""

    def __init__(self, output: Any, output_path: str) -> None:
        self.output = output
        self.output_path = output_path

    @property
    def summary(self) -> Any:
        return self.output.summary

    @property
    def segments(self) -> list:
        return self.output.segments

    def headline_score(self) -> float:
        """Return the mean contradiction score (lower is better)."""
        return self.output.summary.mean_contradiction_score

    def worst_segments(self, n: int = 3) -> list:
        """Return the N highest-contradiction segments."""
        return sorted(
            self.output.segments,
            key=lambda s: s.contradiction_score,
            reverse=True,
        )[:n]


# ---------------------------------------------------------------------------
# Paired Comparison
# ---------------------------------------------------------------------------


def compute_improvement(
    nkge_result: EvalRunResult,
    baseline_result: EvalRunResult,
) -> dict[str, Any]:
    """Compute the headline improvement metrics between NKGE and baseline.

    Args:
        nkge_result: Completed NKGE evaluation run.
        baseline_result: Completed baseline evaluation run.

    Returns:
        Dict with improvement percentages and delta metrics.
    """
    ns = nkge_result.summary
    bs = baseline_result.summary

    baseline_score = bs.mean_contradiction_score
    nkge_score = ns.mean_contradiction_score

    if baseline_score > 0:
        improvement_pct = (baseline_score - nkge_score) / baseline_score * 100
    else:
        improvement_pct = 0.0 if nkge_score == 0 else -100.0

    return {
        "nkge_contradiction_score": nkge_score,
        "baseline_contradiction_score": baseline_score,
        "improvement_pct": round(improvement_pct, 1),
        "nkge_coherence": ns.mean_coherence_score,
        "baseline_coherence": bs.mean_coherence_score,
        "coherence_delta": round(ns.mean_coherence_score - bs.mean_coherence_score, 2),
        "nkge_graph_coverage": ns.mean_graph_coverage,
        "baseline_graph_coverage": bs.mean_graph_coverage,
        "nkge_retrieval_precision": ns.mean_retrieval_precision,
        "baseline_retrieval_precision": bs.mean_retrieval_precision,
        "nkge_nodes_created": ns.total_nodes_created,
        "baseline_nodes_created": bs.total_nodes_created,
        "nkge_rels_created": ns.total_relationships_created,
        "baseline_rels_created": bs.total_relationships_created,
        "nkge_total_contradictions": (
            ns.critical_contradictions_total + ns.major_contradictions_total
            + ns.minor_contradictions_total + ns.soft_contradictions_total
        ),
        "baseline_total_contradictions": (
            bs.critical_contradictions_total + bs.major_contradictions_total
            + bs.minor_contradictions_total + bs.soft_contradictions_total
        ),
        "severity_breakdown": {
            "nkge": {
                "critical": ns.critical_contradictions_total,
                "major": ns.major_contradictions_total,
                "minor": ns.minor_contradictions_total,
                "soft": ns.soft_contradictions_total,
            },
            "baseline": {
                "critical": bs.critical_contradictions_total,
                "major": bs.major_contradictions_total,
                "minor": bs.minor_contradictions_total,
                "soft": bs.soft_contradictions_total,
            },
        },
    }


def format_comparison_table(comparison: dict[str, Any]) -> str:
    """Format a paired comparison as a readable markdown table."""
    lines = [
        "| Metric | NKGE | Baseline | Delta |",
        "|--------|------|----------|-------|",
        f"| Contradiction Score | {comparison['nkge_contradiction_score']:.2f} | {comparison['baseline_contradiction_score']:.2f} | {comparison['improvement_pct']:+.1f}% improvement |",
        f"| Coherence (1-5) | {comparison['nkge_coherence']:.2f} | {comparison['baseline_coherence']:.2f} | {comparison['coherence_delta']:+.2f} |",
        f"| Graph Coverage | {comparison['nkge_graph_coverage']:.2%} | {comparison['baseline_graph_coverage']:.2%} | — |",
        f"| Retrieval Precision | {comparison['nkge_retrieval_precision']:.2%} | {comparison['baseline_retrieval_precision']:.2%} | — |",
        f"| Nodes Created | {comparison['nkge_nodes_created']} | {comparison['baseline_nodes_created']} | — |",
        f"| Relationships Created | {comparison['nkge_rels_created']} | {comparison['baseline_rels_created']} | — |",
        f"| Total Contradictions | {comparison['nkge_total_contradictions']} | {comparison['baseline_total_contradictions']} | — |",
    ]
    return "\n".join(lines)


def load_eval_run(path: str) -> EvalRunResult:
    """Load a previously saved evaluation run from disk.

    Args:
        path: Path to the eval_output.json file.

    Returns:
        EvalRunResult with the loaded data.
    """
    from src.schema import EvalRunOutput

    data = Path(path).read_text(encoding="utf-8")
    output = EvalRunOutput.model_validate_json(data)
    return EvalRunResult(output=output, output_path=path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_judge_response(content: str) -> list[dict[str, Any]]:
    """Parse the LLM judge's response into a list of contradiction dicts.

    Handles markdown fences, preamble, and common LLM output quirks.
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

    bracket_start = text.find("[")
    if bracket_start != -1:
        text = text[bracket_start:]
    elif text.startswith("{"):
        text = "[" + text + "]"

    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        return []
    except json.JSONDecodeError:
        last_brace = text.rfind("}")
        if last_brace != -1:
            truncated = text[: last_brace + 1] + "]"
            truncated = re.sub(r",\s*([}\]])", r"\1", truncated)
            try:
                result = json.loads(truncated)
                if isinstance(result, list):
                    logger.warning("Recovered %d contradictions from truncated judge output", len(result))
                    return result
            except json.JSONDecodeError:
                pass

        logger.warning("Judge response parse failed, treating as no contradictions")
        return []


def _parse_json_safe(content: str) -> dict[str, Any]:
    """Parse a JSON object from LLM response content, with fallbacks."""
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
        brace_depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                brace_depth += 1
            elif text[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    text = text[brace_start: i + 1]
                    break

    return json.loads(text)


def _extract_entity_mentions(
    text: str,
    known_entities: set[str],
) -> set[str]:
    """Extract mentions of known entities from generated text.

    Uses case-insensitive matching with word boundary awareness.
    """
    found: set[str] = set()
    text_lower = text.lower()
    for entity in known_entities:
        if entity.lower() in text_lower:
            found.add(entity)
    return found


def _extract_keywords(fact_line: str) -> list[str]:
    """Extract significant keywords from a fact line for precision matching.

    Strips common verbs/prepositions and returns entity-like words.
    """
    stop_words = {
        "is", "are", "was", "were", "has", "have", "had", "the", "a", "an",
        "at", "in", "on", "to", "of", "for", "with", "and", "or", "not",
        "by", "from", "as", "this", "that", "but", "if", "so", "be", "been",
        "-", "—", "character", "location", "event", "object", "faction",
        "aligned", "located", "owns", "toward",
    }
    words = re.findall(r"[A-Za-z]+", fact_line)
    keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    return keywords
