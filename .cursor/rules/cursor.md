# Lorekeeper (LKGE) — Cursor Rules
# Place this file at .cursor/rules/cursor.md

---

## 0. Source of Truth

The file `NKGE_Project_Design_Document.md` in the repository root is the
authoritative specification for this project. Before making any architectural
decision, writing any implementation code, or suggesting any structural change,
read that document. Every layer definition, technology choice, schema design,
retrieval policy, evaluation methodology, and file structure defined there takes
precedence over general best practices or your own defaults.

If you identify a genuine conflict between the design document and a technical
constraint (e.g., a library API has changed), state the conflict explicitly,
propose a resolution, and log it as a Design Decisions Log entry in
`study_packet.md` before proceeding.

---

## 1. Repository Structure

Always maintain this exact structure. Do not create files outside it without
explicit instruction.

```
lorekeeper/
├── notebooks/
│   ├── 01_schema_and_ingest.ipynb
│   ├── 02_extraction_pipeline.ipynb
│   ├── 03_dual_retrieval.ipynb
│   └── 04_eval_harness.ipynb
├── src/
│   ├── schema.py
│   ├── graph_client.py
│   ├── extraction.py
│   ├── retrieval.py
│   ├── guard.py
│   ├── prompts.py
│   └── eval.py
├── tests/
├── eval_runs/
├── assets/
├── app.py
├── api.py
├── prompts_registry.json
├── study_packet.md
├── NKGE_Project_Design_Document.md
├── .env.example
├── requirements.txt
└── README.md
```

---

## 2. Code Quality Rules

### 2.1 Type Annotations and Docstrings
Every function and method in `/src/` must have:
- Complete type annotations on all parameters and return values
- A docstring stating: what the function does, what each parameter is, what it
  returns, and what exceptions it may raise

```python
# CORRECT
def merge_character(self, character: Character, branch_id: str) -> bool:
    """
    Idempotently write a Character node to Neo4j using MERGE on name.

    Args:
        character: Validated Character Pydantic model instance.
        branch_id: Active story branch identifier for this write.

    Returns:
        True if a new node was created, False if an existing node was updated.

    Raises:
        Neo4jWriteError: If the driver fails to execute the MERGE statement.
    """

# WRONG — no annotations, no docstring
def merge_character(self, character, branch_id):
    ...
```

### 2.2 Pydantic v2 Syntax Only
Use Pydantic v2 syntax exclusively. v1 patterns are not permitted.

```python
# CORRECT — v2
from pydantic import BaseModel, field_validator, model_validator

class ExtractionProposal(BaseModel):
    entity_name: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

# WRONG — v1
class ExtractionProposal(BaseModel):
    @validator("confidence")
    def confidence_must_be_valid(cls, v):
        ...
```

### 2.3 Parameterized Cypher — No String Interpolation
All Neo4j queries must use parameterized Cypher. String interpolation in query
construction is never permitted — it is both a security issue and a correctness
issue with special characters in entity names.

```python
# CORRECT
query = """
MATCH (c:Character {name: $name, branch_id: $branch_id})
RETURN c
"""
result = session.run(query, name=character_name, branch_id=branch_id)

# WRONG — never do this
query = f"MATCH (c:Character {{name: '{character_name}'}}) RETURN c"
```

### 2.4 Prompt Registry — No Inline Prompt Strings
All LLM prompt templates must be defined in `src/prompts.py` and registered in
`prompts_registry.json`. Inline prompt strings anywhere else in the codebase are
not permitted.

```python
# CORRECT — in src/prompts.py
GENERATION_PROMPT_V2 = PromptTemplate(
    version="generation_v2",
    system="""You are a narrative engine...""",
    ...
)

# CORRECT — in any other file
from src.prompts import GENERATION_PROMPT_V2
response = llm.invoke(GENERATION_PROMPT_V2.format(**context))

# WRONG — inline prompt string outside prompts.py
response = llm.invoke("You are a narrative engine... " + context)
```

### 2.5 FastAPI — No Raw Dicts in Handlers
Every FastAPI endpoint must define explicit Pydantic request and response models.
Returning raw dicts from handlers is not permitted.

```python
# CORRECT
class GenerateSegmentRequest(BaseModel):
    player_action: str
    session_id: str

class GenerateSegmentResponse(BaseModel):
    segment_text: str
    seq_id: int
    guard_violations: list[ConstraintViolation]

@app.post("/segment", response_model=GenerateSegmentResponse)
async def generate_segment(request: GenerateSegmentRequest) -> GenerateSegmentResponse:
    ...

# WRONG
@app.post("/segment")
async def generate_segment(request: dict):
    return {"segment_text": "..."}
```

---

## 3. Notebook Rules

### 3.1 Header Cell Required
Every notebook must begin with a markdown cell containing:
- The notebook's purpose (one sentence)
- Prerequisites (what must be set up or run before this notebook)
- Expected outputs (what the notebook produces)

### 3.2 150-Line Executable Code Limit
No notebook should contain more than 150 lines of executable Python code across
all cells. Logic beyond that belongs in `/src/` and is imported. Notebooks are
for orchestration, visualization, and interaction — not business logic.

### 3.3 Clean Top-to-Bottom Run Required
Every notebook must run completely and without errors via
`Kernel → Restart and Run All` before the phase it belongs to is marked
complete. If a notebook requires user input mid-run (e.g., entity review
approval), that interaction must be handled via an `ipywidgets` widget that
has a sensible auto-proceed default so automated runs can complete.

---

## 4. Testing Rules

### 4.1 Required Test Coverage
The following modules require unit tests in `/tests/` before their phase is
considered complete:
- `src/extraction.py` — every public function, with tests covering: happy path,
  confidence below threshold, name collision with existing node, status
  consistency violation
- `src/guard.py` — every constraint check, with tests covering: violation
  detected and correctly typed, no violation returns empty list, guard behavior
  in strict vs. permissive mode

### 4.2 Eval Harness Artifact Validation
Before P6 is marked complete:
- A baseline run must complete and produce a valid JSON artifact in `eval_runs/`
  matching the schema in Appendix B of the design document
- An NKGE run must complete and produce the same
- The comparison report cell must render a side-by-side table and a
  contradiction severity chart without errors
- The NKGE contradiction score must be logged — if it is not lower than the
  baseline score, log a note in `study_packet.md` Gotchas section explaining
  why and what would be investigated next

---

## 5. Observability Rules

Every segment generation cycle must emit an OpenTelemetry trace. The trace must
contain spans for each of these steps, in this order:

1. `retrieval.cypher` — T1 through T4 Cypher queries
2. `retrieval.vector` — ChromaDB similarity search
3. `context.assembly` — merging graph and vector context into prompt
4. `guard.check` — all 5 constraint checks
5. `llm.generate` — the generation LLM call
6. `extraction.propose` — structured extraction LLM call
7. `extraction.validate` — deterministic validator
8. `graph.merge` — Neo4j MERGE writes

Every span must include these attributes where applicable:
- `segment.seq_id`
- `branch.id`
- `guard.violations_count`
- `extraction.proposals_count`
- `extraction.committed_count`
- `extraction.flagged_count`
- `llm.input_tokens`
- `llm.output_tokens`
- `llm.model`

---

## 6. Prompt Versioning Rules

### 6.1 Every Prompt Change Creates a New Version
When any prompt template in `src/prompts.py` is modified, a new version entry
must be added to `prompts_registry.json` before the change is used in any run.

### 6.2 Registry Entry Schema
```json
{
  "prompt_id": "generation_v3",
  "prompt_function": "build_generation_prompt",
  "changed_sections": ["System: Known Facts"],
  "change_reason": "Added T4 orphan tier to graph context",
  "breaking_change": false,
  "eval_run_id": "run_2025_03_12_nkge",
  "contradiction_score_before": 2.14,
  "contradiction_score_after": 1.76,
  "coherence_score_before": 3.8,
  "coherence_score_after": 3.9,
  "promoted_to_main": true,
  "promoted_by": "Kutlu Mizrak",
  "promoted_at": "ISO8601 timestamp"
}
```

### 6.3 Breaking Changes Require Re-evaluation
A breaking change is any modification that adds required context fields, removes
existing context sections, or changes the output schema expected from the LLM.
Breaking changes must increment the major version number and must have an
associated eval run before being used in production mode.

---

## 7. study_packet.md Rules

`study_packet.md` is a first-class deliverable. It must be created at project
initialization (before any implementation code) and updated at the end of every
phase.

### 7.1 Required Sections
The file must always contain all six of these top-level sections:
1. Core Concepts
2. Design Decisions Log
3. Technology Reference
4. Evaluation System Deep Dive
5. Interview Q&A
6. Gotchas and Lessons Learned

### 7.2 Phase-End Update Checklist
At the end of every phase (P1–P8), before beginning the next phase, update
`study_packet.md` with:
- [ ] New entries in **Core Concepts** for every concept introduced in this phase
- [ ] New entries in **Design Decisions Log** for every non-trivial choice made
- [ ] New entries in **Technology Reference** for every tool first used in this phase
- [ ] 3–5 new **Interview Q&A** pairs covering the phase's work
- [ ] Any **Gotchas** encountered during the phase (bugs, wrong assumptions, surprises)

### 7.3 Core Concepts Format
Each concept entry must have three parts:
```markdown
### [Concept Name]
**Plain English:** 3–5 sentence accessible explanation.
**Technical:** Precise, interview-ready explanation with correct terminology.
**One-liner:** A single sentence suitable for explaining to a non-technical person.
```

### 7.4 Design Decisions Log Format
```markdown
### [Decision Title] — Phase [N]
**Decision:** What was chosen.
**Alternatives considered:** What else was on the table.
**Reason:** Why this choice was made.
**Scale consideration:** What would change if requirements grew 100×.
```

### 7.5 Interview Q&A Format
```markdown
**Q: [Question a senior ML engineer at a graph-AI company would ask]**
- [Bullet point answer]
- [Bullet point answer]
- [Bullet point answer]
```

---

## 8. Neo4j Rules

### 8.1 All Writes Are Idempotent
Every Neo4j write must use `MERGE`, never `CREATE`, unless the intent is
explicitly to create a duplicate (which should never happen in this project).
MERGE on `name` for Character, Location, Object, and Faction nodes. MERGE on
`seq_id + branch_id` for Event and Segment nodes.

### 8.2 Branch ID on All Mutable State
Every node and relationship that represents story state (Event, LOCATED_AT,
PARTICIPATED_IN, OWNS, CAUSED_BY) must carry a `branch_id` property. Queries
that retrieve story state must always filter by `branch_id`. Queries against
stable reference data (Character names, Location names) do not require branch
filtering.

### 8.3 Constraints and Indexes at Init
The following must be created in `01_schema_and_ingest.ipynb` before any data
is written:
```cypher
CREATE CONSTRAINT character_name_unique IF NOT EXISTS
  FOR (c:Character) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT location_name_unique IF NOT EXISTS
  FOR (l:Location) REQUIRE l.name IS UNIQUE;

CREATE INDEX event_seq_branch IF NOT EXISTS
  FOR (e:Event) ON (e.seq_id, e.branch_id);

CREATE INDEX segment_seq_branch IF NOT EXISTS
  FOR (s:Segment) ON (s.seq_id, s.branch_id);

CREATE VECTOR INDEX character_persona IF NOT EXISTS
  FOR (c:Character) ON (c.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
```

---

## 9. Evaluation Rules

### 9.1 Baseline Definition
The baseline mode must use the same prompt structure, same player actions, and
same story seed as the NKGE mode. The only difference is that the graph context
block is replaced by a rolling text summary of the last 3 segments concatenated.
The baseline is not "no context" — it is the realistic alternative to building
a graph system.

### 9.2 Judge Receives Graph-Derived Facts
The LLM contradiction judge must receive its facts list from a Neo4j summary
Cypher query — not from raw segment text. This is non-negotiable. Passing raw
text to the judge defeats the purpose of having a structured graph.

### 9.3 Never Cherry-Pick Eval Runs
The eval results table in the README must be populated from the first clean
run after Phase 6 completion. Do not re-run until scores improve. If the
first run shows NKGE performing worse than baseline, log it in Gotchas,
investigate, fix the root cause, and document what changed before re-running.

### 9.4 Eval Artifacts Are Append-Only
Files in `eval_runs/` are never modified after creation. Each run produces a
new directory `eval_runs/{run_id}/`. Historical runs are preserved for
comparison. Add `eval_runs/` to `.gitignore` except for a `eval_runs/samples/`
subdirectory containing the two canonical runs (baseline and NKGE) used to
populate the README.

---

## 10. README Rules

The README is assembled in Phase 8 following Section 9 of
`NKGE_Project_Design_Document.md` exactly. It is not written before then.

The README is not complete until all five of these are present:
- [ ] Real eval results table from an actual eval run (no placeholder text)
- [ ] Two graph visualization screenshots in `assets/` (early and late graph)
- [ ] A concrete contradiction example from the highest-scoring baseline segment
- [ ] The Mermaid architecture diagram from Section 9.5 of the design document
- [ ] Links to both `NKGE_Project_Design_Document.md` and `study_packet.md`

---

## 11. Environment and Secrets Rules

- Never hardcode API keys, URIs, usernames, or passwords anywhere in the
  codebase
- All secrets are loaded from `.env` via `python-dotenv`
- `.env` is in `.gitignore` — always
- `.env.example` is committed and kept up to date whenever a new env variable
  is added
- If a new env variable is added during implementation, update `.env.example`
  immediately in the same commit

---

## 12. General Behavior Rules for Cursor

- Do not refactor working code unless a phase explicitly calls for it or a bug
  requires it
- Do not add dependencies not in `requirements.txt` without flagging them and
  explaining why they are needed
- Do not suggest replacing Neo4j with another graph database, ChromaDB with
  another vector store, or LangGraph with another orchestration framework —
  these choices are fixed per the design document
- When in doubt about scope, do less and ask rather than doing more and
  breaking something
- After completing any phase, explicitly state: "Phase [N] complete. Updating
  study_packet.md now." Then update it before touching Phase [N+1] code
- If you encounter a situation where following a rule would produce clearly
  broken behavior, flag it explicitly rather than silently violating the rule
