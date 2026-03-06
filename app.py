"""Streamlit interactive frontend for Lorekeeper.

Run with: streamlit run app.py

Features:
- Interactive story generation with player action input
- Live knowledge graph visualization sidebar (pyvis)
- Guard violation display with severity coloring
- Branch selector for story divergence
- Session state management across turns
- NKGE vs Baseline mode toggle
- Extraction results transparency panel
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lorekeeper — Narrative Knowledge Graph Engine",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Lazy imports — heavy deps loaded after page config
# ---------------------------------------------------------------------------


@st.cache_resource
def init_components():
    """Initialize all pipeline components once (cached across reruns)."""
    from langchain_anthropic import ChatAnthropic

    from src.extraction import ExtractionPipeline
    from src.graph_client import GraphClient
    from src.guard import ContradictionGuard
    from src.persona import PersonaStore
    from src.pipeline import StoryPipeline
    from src.retrieval import CypherRetriever, VectorRetriever

    gc = GraphClient(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", ""),
    )
    gc.verify_connectivity()

    llm = ChatAnthropic(
        model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
        temperature=0.7,
        max_tokens=1024,
    )

    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store/")
    persona_store = PersonaStore(persist_dir=chroma_dir)

    pipeline = StoryPipeline(
        graph_client=gc,
        cypher_retriever=CypherRetriever(gc, token_budget=2000),
        vector_retriever=VectorRetriever(persist_dir=chroma_dir),
        extraction=ExtractionPipeline(llm=llm, graph_client=gc),
        llm=llm,
        guard=ContradictionGuard(gc, mode=os.getenv("GUARD_MODE", "permissive")),
        persona_store=persona_store,
    )

    return gc, pipeline, persona_store


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Set up Streamlit session_state defaults on first load."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.story_segments = []
        st.session_state.mode = "nkge"
        st.session_state.branch_id = "main"
        st.session_state.current_location = "Iron Tavern"
        st.session_state.present_characters = ["Kael", "Maren"]
        st.session_state.last_seq_id = 5
        st.session_state.last_text = "The tavern fell silent as Kael and Maren locked eyes."
        st.session_state.violations_history = []
        st.session_state.extraction_history = []
        st.session_state.generating = False


# ---------------------------------------------------------------------------
# Graph visualization
# ---------------------------------------------------------------------------


def render_graph_viz(gc) -> str | None:
    """Build a pyvis HTML graph from current Neo4j state."""
    from pyvis.network import Network

    COLOR_MAP = {
        "Character": "#4A90D9",
        "Location": "#E67E22",
        "Event": "#9B59B6",
        "Object": "#2ECC71",
        "Segment": "#95A5A6",
        "Faction": "#E74C3C",
    }

    query = """
    MATCH (n)
    WHERE n:Character OR n:Location OR n:Event OR n:Object
    OPTIONAL MATCH (n)-[r]->(m)
    WHERE m:Character OR m:Location OR m:Event OR m:Object
    RETURN n, labels(n) AS n_labels, r, type(r) AS r_type, m, labels(m) AS m_labels
    """

    nodes_seen = set()
    edges = []

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
    )
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
    )

    try:
        with gc._driver.session(database=gc._database) as session:
            result = session.run(query)
            for record in result:
                n = record["n"]
                n_labels = record["n_labels"]
                m = record["m"]
                m_labels = record["m_labels"]
                r_type = record["r_type"]

                n_id = str(dict(n).get("name", dict(n).get("seq_id", id(n))))
                n_label = n_labels[0] if n_labels else "Node"
                n_props = dict(n)
                n_title = _format_node_tooltip(n_label, n_props)
                color = COLOR_MAP.get(n_label, "#888888")

                if n_id not in nodes_seen:
                    net.add_node(
                        n_id,
                        label=n_id,
                        color=color,
                        title=n_title,
                        size=25 if n_label == "Character" else 18,
                    )
                    nodes_seen.add(n_id)

                if m is not None:
                    m_id = str(dict(m).get("name", dict(m).get("seq_id", id(m))))
                    m_label = m_labels[0] if m_labels else "Node"
                    m_props = dict(m)
                    m_title = _format_node_tooltip(m_label, m_props)
                    m_color = COLOR_MAP.get(m_label, "#888888")

                    if m_id not in nodes_seen:
                        net.add_node(
                            m_id,
                            label=m_id,
                            color=m_color,
                            title=m_title,
                            size=25 if m_label == "Character" else 18,
                        )
                        nodes_seen.add(m_id)

                    edge_key = (n_id, m_id, r_type)
                    if edge_key not in edges:
                        net.add_edge(n_id, m_id, label=r_type, color="#555555")
                        edges.append(edge_key)
    except Exception as exc:
        return None

    if not nodes_seen:
        return None

    tmp_dir = Path("./assets")
    tmp_dir.mkdir(exist_ok=True)
    html_path = str(tmp_dir / "graph_viz.html")
    net.save_graph(html_path)
    return html_path


def _format_node_tooltip(label: str, props: dict) -> str:
    """Format node properties into an HTML tooltip for pyvis."""
    lines = [f"<b>{label}</b>"]
    skip_keys = {"branch_id", "persona_doc_id"}
    for k, v in props.items():
        if k in skip_keys or v is None:
            continue
        if isinstance(v, list):
            v = ", ".join(str(i) for i in v)
        lines.append(f"{k}: {v}")
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(gc) -> None:
    """Render the sidebar with graph visualization and session controls."""
    st.sidebar.title("📖 Lorekeeper")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Controls")

    mode = st.sidebar.radio(
        "Generation Mode",
        ["nkge", "baseline"],
        index=0 if st.session_state.mode == "nkge" else 1,
        help="NKGE: full graph RAG + guard + personas. Baseline: rolling text summary only.",
    )
    st.session_state.mode = mode

    st.sidebar.markdown(f"**Session:** `{st.session_state.session_id}`")
    st.sidebar.markdown(f"**Branch:** `{st.session_state.branch_id}`")
    st.sidebar.markdown(f"**Location:** {st.session_state.current_location}")
    st.sidebar.markdown(f"**Characters:** {', '.join(st.session_state.present_characters)}")
    st.sidebar.markdown(f"**Segment:** #{st.session_state.last_seq_id}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Knowledge Graph")
    st.sidebar.caption("Graph visualization is in the expandable panel below the story.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Graph Stats")
    try:
        node_counts = gc.get_node_counts()
        rel_counts = gc.get_relationship_counts()
        total_nodes = sum(node_counts.values())
        total_rels = sum(rel_counts.values())
        st.sidebar.metric("Total Nodes", total_nodes)
        st.sidebar.metric("Total Relationships", total_rels)

        with st.sidebar.expander("Node breakdown"):
            for label, count in sorted(node_counts.items()):
                st.text(f"  {label}: {count}")
        with st.sidebar.expander("Relationship breakdown"):
            for rtype, count in sorted(rel_counts.items()):
                st.text(f"  {rtype}: {count}")
    except Exception:
        st.sidebar.warning("Could not load graph stats.")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------


def render_story_history() -> None:
    """Render the scrollable story history."""
    for i, entry in enumerate(st.session_state.story_segments):
        with st.container():
            st.markdown(
                f"<div style='background-color: #16213e; padding: 8px 12px; "
                f"border-radius: 6px; margin-bottom: 4px;'>"
                f"<span style='color: #E67E22; font-weight: 600;'>▶ Player:</span> "
                f"<span style='color: #ddd;'>{entry['action']}</span></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='background-color: #0f3460; padding: 12px 16px; "
                f"border-radius: 6px; margin-bottom: 8px; line-height: 1.6;'>"
                f"{entry['text']}</div>",
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                n_violations = entry.get("violation_count", 0)
                color = "#2ECC71" if n_violations == 0 else "#E74C3C"
                st.markdown(
                    f"<span style='color: {color};'>Guard: {n_violations} violation(s)</span>",
                    unsafe_allow_html=True,
                )
            with col2:
                n_extracted = entry.get("extracted_count", 0)
                st.markdown(
                    f"<span style='color: #3498DB;'>Extracted: {n_extracted} entities</span>",
                    unsafe_allow_html=True,
                )
            with col3:
                tokens = entry.get("graph_tokens", 0) + entry.get("vector_tokens", 0)
                st.markdown(
                    f"<span style='color: #9B59B6;'>Context: {tokens} tokens</span>",
                    unsafe_allow_html=True,
                )

            if entry.get("violations"):
                with st.expander(f"⚠️ Guard violations (segment #{entry['seq_id']})"):
                    for v in entry["violations"]:
                        severity_colors = {
                            "critical": "#E74C3C",
                            "major": "#E67E22",
                            "minor": "#F1C40F",
                            "soft": "#95A5A6",
                        }
                        sev = v.get("severity", "minor")
                        color = severity_colors.get(sev, "#95A5A6")
                        st.markdown(
                            f"<span style='color: {color}; font-weight: 600;'>"
                            f"[{sev.upper()}]</span> {v.get('check_name', '')}: "
                            f"{v.get('violation_message', '')}",
                            unsafe_allow_html=True,
                        )

            if entry.get("extractions"):
                with st.expander(f"🔍 Extraction results (segment #{entry['seq_id']})"):
                    for ext in entry["extractions"]:
                        status = "✅" if ext.get("committed", False) else "⚠️"
                        st.text(
                            f"  {status} {ext.get('entity_type', '')}: "
                            f"{ext.get('entity_name', '')} "
                            f"(conf: {ext.get('confidence', 0):.2f})"
                        )

            st.markdown("---")


def generate_segment(gc, pipeline) -> None:
    """Run the pipeline for a single segment and update session state."""
    from src.schema import SessionState

    action = st.session_state.get("player_input", "")
    if not action:
        return

    st.session_state.generating = True

    session = SessionState(
        session_id=st.session_state.session_id,
        story_seed="The Iron Tavern",
        active_branch_id=st.session_state.branch_id,
        current_location=st.session_state.current_location,
        present_characters=st.session_state.present_characters,
        last_segment_seq_id=st.session_state.last_seq_id,
        last_segment_text=st.session_state.last_text,
        mode=st.session_state.mode,
    )

    with st.spinner("Generating story segment..."):
        result = pipeline.run(session, player_action=action)

    generated_text = result.get("generated_text", "(generation failed)")
    violations = result.get("violations", [])
    extraction_result = result.get("extraction_result")
    graph_tokens = result.get("graph_context_tokens", 0)
    vector_tokens = result.get("vector_context_tokens", 0)

    new_seq_id = st.session_state.last_seq_id + 1

    violation_dicts = []
    for v in violations:
        if hasattr(v, "check_name"):
            violation_dicts.append({
                "check_name": v.check_name,
                "violation_message": v.violation_message,
                "severity": v.severity,
            })
        elif isinstance(v, dict):
            violation_dicts.append(v)

    extraction_dicts = []
    if extraction_result and hasattr(extraction_result, "proposals"):
        for p in extraction_result.proposals:
            committed = p in (extraction_result.approved if hasattr(extraction_result, "approved") else [])
            extraction_dicts.append({
                "entity_type": p.entity_type,
                "entity_name": p.entity_name,
                "confidence": p.confidence,
                "committed": committed,
            })

    entry = {
        "seq_id": new_seq_id,
        "action": action,
        "text": generated_text,
        "violations": violation_dicts,
        "violation_count": len(violation_dicts),
        "extractions": extraction_dicts,
        "extracted_count": len(extraction_dicts),
        "graph_tokens": graph_tokens,
        "vector_tokens": vector_tokens,
    }
    st.session_state.story_segments.append(entry)

    st.session_state.last_seq_id = new_seq_id
    st.session_state.last_text = generated_text
    st.session_state.generating = False

    _update_present_characters(gc)


def _update_present_characters(gc) -> None:
    """Refresh present_characters from graph based on current location."""
    try:
        chars = gc.get_characters_at_location(
            st.session_state.current_location,
            st.session_state.branch_id,
        )
        if chars:
            st.session_state.present_characters = [c.name for c in chars]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main Streamlit app entry point."""
    init_session_state()

    try:
        gc, pipeline, persona_store = init_components()
    except Exception as exc:
        st.error(
            f"Failed to initialize components. Ensure Neo4j is running and .env is configured.\n\n"
            f"Error: {exc}"
        )
        return

    render_sidebar(gc)

    st.title("Lorekeeper")
    st.caption("An LLM storytelling engine that never forgets what happened.")

    mode_label = "**NKGE**" if st.session_state.mode == "nkge" else "**Baseline**"
    mode_desc = (
        "Graph RAG + Guard + Personas active"
        if st.session_state.mode == "nkge"
        else "Rolling text summary only (no graph memory)"
    )
    st.markdown(f"Mode: {mode_label} — {mode_desc}")

    st.markdown("---")

    if st.session_state.story_segments:
        render_story_history()
    else:
        st.markdown(
            "<div style='background-color: #0f3460; padding: 16px; "
            "border-radius: 8px; text-align: center; color: #aaa;'>"
            "<i>The tavern fell silent as Kael and Maren locked eyes.</i><br><br>"
            "Enter your first action below to begin the story.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

    with st.expander("🗺️ Knowledge Graph Visualization", expanded=False):
        html_path = render_graph_viz(gc)
        if html_path and Path(html_path).exists():
            html_content = Path(html_path).read_text(encoding="utf-8")
            st.components.v1.html(html_content, height=520, scrolling=True)
        else:
            st.info("No graph data to visualize.")

    with st.form("action_form", clear_on_submit=True):
        col_input, col_submit = st.columns([5, 1])
        with col_input:
            player_input = st.text_input(
                "Your action:",
                placeholder="What do you do? (e.g., 'Kael confronts Maren about the ambush')",
                key="player_input",
                label_visibility="collapsed",
            )
        with col_submit:
            submitted = st.form_submit_button("⚔️ Act", use_container_width=True)

    if submitted and player_input:
        generate_segment(gc, pipeline)
        st.rerun()


if __name__ == "__main__":
    main()
