"""Contradiction guard and branch management for the Lorekeeper pipeline.

Implements all five guard checks from Section 4.5 of the design document:
1. Dead character active — Character with status:'dead' in present_chars
2. Location inaccessible — Target location with accessible:false
3. Hostile co-presence — Unresolved hostile KNOWS between present characters
4. Knowledge boundary violation — Character referenced event they couldn't know about
5. Object ownership conflict — Object claimed by character who no longer owns it

Guard modes:
- permissive: violations are logged and injected into the prompt but never block
- strict: Critical/Major violations trigger up to 2 generation retries; if still
  present after retries, a branch is created to isolate the divergent state

Branch management:
- Branches are created when strict-mode violations can't be resolved by retry
- Cap of 5 active branches per session; LRU archival for overflow
- All graph queries filter by branch_id to prevent cross-branch contamination
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.graph_client import GraphClient
from src.schema import ConstraintViolation, SessionState

logger = logging.getLogger(__name__)

_GUARD_MODE = os.getenv("GUARD_MODE", "permissive")


class ContradictionGuard:
    """Pre-generation constraint checker that queries the graph for violations.

    Each check method returns a list of ConstraintViolation objects.
    `run_all_checks` executes every registered check against the current
    session state and returns the combined violation list.

    Args:
        graph_client: Active GraphClient instance.
        mode: "permissive" (inject only) or "strict" (inject + retry/branch).
    """

    def __init__(
        self,
        graph_client: GraphClient,
        mode: str | None = None,
    ) -> None:
        self._gc = graph_client
        self.mode = mode or _GUARD_MODE

    def check_dead_character_active(
        self, present_chars: list[str], branch_id: str
    ) -> list[ConstraintViolation]:
        """Check if any character in the scene is dead in the graph.

        Maps to Appendix A "Dead Character Guard". This is a Critical
        violation — the LLM must not write actions for dead characters.

        Args:
            present_chars: Names of characters in the current scene.
            branch_id: Active branch ID.

        Returns:
            List of violations for dead characters found in present_chars.
        """
        if not present_chars:
            return []

        query = """
        MATCH (c:Character)
        WHERE c.name IN $present_chars
              AND c.status = 'dead'
              AND (c.branch_id = $branch_id OR c.branch_id IS NULL)
        OPTIONAL MATCH (c)-[:PARTICIPATED_IN]->(e:Event {branch_id: $branch_id})
        WITH c, max(e.seq_id) AS last_event
        RETURN c.name AS name, last_event
        """
        violations: list[ConstraintViolation] = []
        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(query, {
                "present_chars": present_chars,
                "branch_id": branch_id,
            }):
                event_ref = f" as of event #{r['last_event']}" if r["last_event"] else ""
                violations.append(ConstraintViolation(
                    check_name="dead_character_active",
                    violation_message=(
                        f"{r['name']} is dead{event_ref}. "
                        f"Do not write any actions, dialogue, or observations for {r['name']}."
                    ),
                    severity="critical",
                ))
        if violations:
            logger.warning("Dead character check: %d violations", len(violations))
        return violations

    def check_location_inaccessible(
        self, target_location: str,
    ) -> list[ConstraintViolation]:
        """Check if the target location is marked as inaccessible.

        Maps to Appendix A "Location Inaccessible Guard". This is a Major
        violation — characters should not be able to enter locked locations.

        Args:
            target_location: Name of the current/target location.

        Returns:
            List with one violation if the location is inaccessible, empty otherwise.
        """
        query = """
        MATCH (l:Location {name: $location})
        WHERE l.accessible = false
        RETURN l.name AS name, l.description_summary AS description
        """
        violations: list[ConstraintViolation] = []
        with self._gc._driver.session(database=self._gc._database) as session:
            result = session.run(query, {"location": target_location})
            for r in result:
                desc = r["description"] or "no reason recorded"
                violations.append(ConstraintViolation(
                    check_name="location_inaccessible",
                    violation_message=(
                        f"{r['name']} is currently inaccessible ({desc}). "
                        f"Characters cannot enter or interact with this location."
                    ),
                    severity="major",
                ))
        if violations:
            logger.warning("Location inaccessible check: %s", target_location)
        return violations

    def check_hostile_copresence(
        self, present_chars: list[str], branch_id: str
    ) -> list[ConstraintViolation]:
        """Check for hostile relationships between characters in the scene.

        Maps to Appendix A "T3 — Hostile Co-Presence". This is a Minor
        violation (tension reminder) — not blocking, but the LLM should
        acknowledge the tension in its narrative.

        Args:
            present_chars: Names of characters in the current scene.
            branch_id: Active branch ID.

        Returns:
            List of violations for each hostile pair found.
        """
        if len(present_chars) < 2:
            return []

        query = """
        MATCH (a:Character)-[r:KNOWS {sentiment: 'hostile'}]-(b:Character)
        WHERE a.name IN $present_chars AND b.name IN $present_chars
              AND a.name < b.name
              AND (r.branch_id = $branch_id OR r.branch_id IS NULL)
        OPTIONAL MATCH (a)-[:PARTICIPATED_IN]->(e:Event {branch_id: $branch_id})
        WITH a, b, r, max(e.seq_id) AS a_last_event
        RETURN a.name AS a_name, b.name AS b_name, a_last_event
        """
        violations: list[ConstraintViolation] = []
        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(query, {
                "present_chars": present_chars,
                "branch_id": branch_id,
            }):
                violations.append(ConstraintViolation(
                    check_name="hostile_copresence",
                    violation_message=(
                        f"{r['a_name']} and {r['b_name']} have a hostile relationship. "
                        f"Their interaction should reflect this tension — "
                        f"do not write them as friendly or cooperative without justification."
                    ),
                    severity="minor",
                ))
        if violations:
            logger.info("Hostile co-presence check: %d pairs", len(violations))
        return violations

    def check_object_ownership(
        self, present_chars: list[str], branch_id: str
    ) -> list[ConstraintViolation]:
        """Check for object ownership conflicts among present characters.

        If a character in the scene owns objects but is dead, or if multiple
        characters claim the same object, flag it. Maps to Appendix A
        "Object Ownership Guard" (adapted for OWNS direction).

        Args:
            present_chars: Names of characters in the current scene.
            branch_id: Active branch ID.

        Returns:
            List of violations for ownership conflicts.
        """
        if not present_chars:
            return []

        query = """
        MATCH (c:Character)-[r:OWNS]->(o:Object)
        WHERE c.name IN $present_chars
              AND (r.branch_id = $branch_id OR r.branch_id IS NULL)
        WITH o, collect(c.name) AS owners
        WHERE size(owners) > 1
        RETURN o.name AS object_name, owners
        """
        violations: list[ConstraintViolation] = []
        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(query, {
                "present_chars": present_chars,
                "branch_id": branch_id,
            }):
                violations.append(ConstraintViolation(
                    check_name="object_ownership_conflict",
                    violation_message=(
                        f"Object '{r['object_name']}' is claimed by multiple characters: "
                        f"{', '.join(r['owners'])}. Only one character can own an object at a time."
                    ),
                    severity="major",
                ))

        dead_owner_query = """
        MATCH (c:Character {status: 'dead'})-[r:OWNS]->(o:Object)
        WHERE (c.branch_id = $branch_id OR c.branch_id IS NULL)
              AND (r.branch_id = $branch_id OR r.branch_id IS NULL)
        RETURN c.name AS dead_owner, o.name AS object_name
        """
        with self._gc._driver.session(database=self._gc._database) as session:
            for r in session.run(dead_owner_query, {"branch_id": branch_id}):
                violations.append(ConstraintViolation(
                    check_name="object_ownership_conflict",
                    violation_message=(
                        f"Dead character {r['dead_owner']} still owns '{r['object_name']}'. "
                        f"This object should be unclaimed or transferred."
                    ),
                    severity="soft",
                ))

        if violations:
            logger.warning("Object ownership check: %d issues", len(violations))
        return violations

    def check_knowledge_boundary(
        self, present_chars: list[str], branch_id: str
    ) -> list[ConstraintViolation]:
        """Check if present characters have knowledge boundary issues.

        A character has a knowledge boundary violation if there are events
        on this branch they neither participated in nor learned about through
        a KNOWS relationship with a participant. This is a Soft violation —
        it warns the LLM not to have characters reference events they
        couldn't know about.

        This is the most heuristic of the five checks — it flags potential
        violations rather than definitive ones, since the LLM might not
        actually reference the unknown events.

        Args:
            present_chars: Names of characters in the current scene.
            branch_id: Active branch ID.

        Returns:
            List of violations for knowledge boundary issues.
        """
        if not present_chars:
            return []

        query = """
        MATCH (e:Event {branch_id: $branch_id})
        WHERE NOT EXISTS {
            MATCH (c:Character {name: $char_name})-[:PARTICIPATED_IN]->(e)
        }
        AND NOT EXISTS {
            MATCH (c:Character {name: $char_name})-[:KNOWS]->(witness:Character)
                  -[:PARTICIPATED_IN]->(e)
        }
        RETURN e.seq_id AS seq_id, e.description AS description
        ORDER BY e.seq_id DESC
        LIMIT 3
        """
        violations: list[ConstraintViolation] = []
        with self._gc._driver.session(database=self._gc._database) as session:
            for char_name in present_chars:
                result = session.run(query, {
                    "char_name": char_name,
                    "branch_id": branch_id,
                })
                unknown_events = [(r["seq_id"], r["description"]) for r in result]
                if unknown_events:
                    event_descs = "; ".join(
                        f"Event #{sid}: {desc[:60]}" for sid, desc in unknown_events
                    )
                    violations.append(ConstraintViolation(
                        check_name="knowledge_boundary",
                        violation_message=(
                            f"{char_name} has no path to know about: {event_descs}. "
                            f"Do not have {char_name} reference or act on information "
                            f"from these events unless they learn about it in this scene."
                        ),
                        severity="soft",
                    ))

        if violations:
            logger.info("Knowledge boundary check: %d chars with gaps", len(violations))
        return violations

    def run_all_checks(self, session: SessionState) -> list[ConstraintViolation]:
        """Execute all guard checks against the current session state.

        Checks run in severity order (critical first) so the most important
        violations appear at the top of the prompt injection.

        Args:
            session: Current session state with location, characters, branch.

        Returns:
            Combined list of all violations from all checks.
        """
        all_violations: list[ConstraintViolation] = []

        all_violations.extend(
            self.check_dead_character_active(
                session.present_characters, session.active_branch_id
            )
        )
        all_violations.extend(
            self.check_location_inaccessible(session.current_location)
        )
        all_violations.extend(
            self.check_object_ownership(
                session.present_characters, session.active_branch_id
            )
        )
        all_violations.extend(
            self.check_hostile_copresence(
                session.present_characters, session.active_branch_id
            )
        )
        all_violations.extend(
            self.check_knowledge_boundary(
                session.present_characters, session.active_branch_id
            )
        )

        logger.info(
            "Guard complete: %d total violations (critical=%d, major=%d, minor=%d, soft=%d)",
            len(all_violations),
            sum(1 for v in all_violations if v.severity == "critical"),
            sum(1 for v in all_violations if v.severity == "major"),
            sum(1 for v in all_violations if v.severity == "minor"),
            sum(1 for v in all_violations if v.severity == "soft"),
        )
        return all_violations

    def has_blocking_violations(self, violations: list[ConstraintViolation]) -> bool:
        """Return True if any violation warrants a retry in strict mode.

        Only Critical and Major violations are considered blocking.

        Args:
            violations: List of violations from run_all_checks.

        Returns:
            True if strict mode should trigger a retry.
        """
        if self.mode != "strict":
            return False
        return any(v.severity in ("critical", "major") for v in violations)


# ---------------------------------------------------------------------------
# Branch Manager
# ---------------------------------------------------------------------------


class BranchManager:
    """Manages story branch creation, switching, and LRU archival.

    Branches are created when strict-mode guard violations cannot be
    resolved by generation retry. Each branch tags all subsequent Neo4j
    writes with a unique branch_id, and all Cypher queries filter by
    branch_id to prevent cross-branch contamination.

    Args:
        graph_client: Active GraphClient instance.
        max_branches: Maximum active branches per session (default 5).
    """

    def __init__(
        self,
        graph_client: GraphClient,
        max_branches: int = 5,
    ) -> None:
        self._gc = graph_client
        self._max_branches = max_branches

    def create_branch(
        self, session: SessionState, reason: str
    ) -> str:
        """Create a new branch from the current session state.

        The new branch ID is derived from the parent branch and the current
        sequence position. The session's branch_ancestry is updated to
        include the new branch.

        Args:
            session: Current session state to branch from.
            reason: Human-readable reason for the branch creation.

        Returns:
            The new branch ID string.
        """
        new_id = f"{session.active_branch_id}_b{session.last_segment_seq_id}"

        self._record_branch_event(
            parent_branch=session.active_branch_id,
            new_branch=new_id,
            fork_seq_id=session.last_segment_seq_id,
            reason=reason,
        )

        session.active_branch_id = new_id
        session.branch_ancestry.append(new_id)

        active = self.get_active_branches(session)
        if len(active) > self._max_branches:
            self._archive_lru_branch(session)

        logger.info(
            "Branch created: %s (reason: %s, ancestry depth: %d)",
            new_id, reason, len(session.branch_ancestry),
        )
        return new_id

    def get_active_branches(self, session: SessionState) -> list[str]:
        """Return all active (non-archived) branch IDs in the session.

        Args:
            session: Current session state.

        Returns:
            List of active branch ID strings.
        """
        query = """
        MATCH (b:BranchEvent)
        WHERE b.session_id = $session_id AND (b.archived IS NULL OR b.archived = false)
        RETURN DISTINCT b.branch_id AS branch_id
        """
        branches = set(session.branch_ancestry)
        with self._gc._driver.session(database=self._gc._database) as db_session:
            for r in db_session.run(query, {"session_id": session.session_id}):
                branches.add(r["branch_id"])
        return list(branches)

    def _record_branch_event(
        self,
        parent_branch: str,
        new_branch: str,
        fork_seq_id: int,
        reason: str,
    ) -> None:
        """Record a branch creation event in the graph for auditability."""
        query = """
        MERGE (b:BranchEvent {branch_id: $new_branch})
        SET b.parent_branch = $parent_branch,
            b.fork_seq_id = $fork_seq_id,
            b.reason = $reason,
            b.archived = false,
            b.session_id = $session_id
        """
        with self._gc._driver.session(database=self._gc._database) as db_session:
            db_session.run(query, {
                "new_branch": new_branch,
                "parent_branch": parent_branch,
                "fork_seq_id": fork_seq_id,
                "reason": reason,
                "session_id": "default",
            })

    def _archive_lru_branch(self, session: SessionState) -> str | None:
        """Archive the least-recently-used branch to stay within the cap.

        LRU is determined by the maximum segment seq_id on each branch.
        The current active branch is never archived.

        Args:
            session: Current session state.

        Returns:
            The archived branch ID, or None if no branch was archived.
        """
        query = """
        MATCH (b:BranchEvent)
        WHERE b.session_id = $session_id
              AND (b.archived IS NULL OR b.archived = false)
              AND b.branch_id <> $active_branch
        OPTIONAL MATCH (s:Segment {branch_id: b.branch_id})
        WITH b, max(s.seq_id) AS last_activity
        ORDER BY last_activity ASC NULLS FIRST
        LIMIT 1
        RETURN b.branch_id AS branch_id
        """
        with self._gc._driver.session(database=self._gc._database) as db_session:
            result = db_session.run(query, {
                "session_id": session.session_id,
                "active_branch": session.active_branch_id,
            })
            record = result.single()
            if not record:
                return None

            lru_branch = record["branch_id"]

            db_session.run(
                "MATCH (b:BranchEvent {branch_id: $branch_id}) SET b.archived = true",
                {"branch_id": lru_branch},
            )

            logger.info("Archived LRU branch: %s", lru_branch)
            return lru_branch

    def is_branch_archived(self, branch_id: str) -> bool:
        """Check if a given branch has been archived.

        Args:
            branch_id: The branch ID to check.

        Returns:
            True if the branch is archived.
        """
        query = """
        MATCH (b:BranchEvent {branch_id: $branch_id})
        RETURN b.archived AS archived
        """
        with self._gc._driver.session(database=self._gc._database) as db_session:
            result = db_session.run(query, {"branch_id": branch_id})
            record = result.single()
            if not record:
                return False
            return bool(record["archived"])
