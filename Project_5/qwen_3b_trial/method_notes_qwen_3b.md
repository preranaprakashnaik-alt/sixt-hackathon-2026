# Project 5 method report

Team name: Two Braincells

Team members and contributions: Prerana Prakash Naik and Harisankar Karamchand.

- Prerana Prakash Naik: owner of the team repository and of the submission. Attempted to solve the problem using a dymanic approach without specific recipe for each of the stage 1 released questions. Proceeded to try with a smaller model Qwen 2.5 coder 3B to check the performance against the Gemma 4 31B model.
- Harisankar Karamchand: method decisions (routing design, the request for a generalization fallback and its evaluation), coordination with the organizer on the model rules, model access, review of results.

Project: Project 5, Enterprise Data Agent Challenge (SIXT + TUM)

Public and final answer files and SHA-256 hashes:

- Stage 1 answer CSV: not generated with the current local model; hash pending.
- Stage 2 answer CSV: not generated with the current local model; hash pending.

Code revision used for the smoke tests: based on `a91518d` with uncommitted local-Qwen, Stage 2 mapping, fallback and validation changes. Record a clean commit before the final runs.

## 1. Approach in one paragraph

The pipeline uses the local Ollama model `qwen2.5-coder:3b` to interpret a business question and select one of 70 calculation recipes with its parameters. Python validates the route and performs the main-path calculations on prepared CSV tables. The shared data layer applies eligibility, identity, currency, allocation, SLA, inventory and other challenge rules once. If the model has no valid recipe, a valid deterministic rules route can be used and is flagged; otherwise a code-generation fallback is attempted and disagreement is reported for review. Stage 1 and Stage 2 use the same calculation code; Stage 2's renamed columns are mapped to the canonical schema before processing.

## 2. Architecture and implementation

```text
stage questions CSV ──► run.py ──► router.py (local Qwen route + validation)
       stage datasets ──► Data layer ──► recipes.py ──► answer CSV + JSON review report
                              ▲               ├─ hard_opt.py (fulfilment optimization)
                Stage 2 map ─┘               └─ flow.py (warehouse event simulation)
```

Core files in `Project_5/pipeline/`:

| File | Purpose |
| --- | --- |
| `run.py` | Loads questions, selects a router, invokes calculations/fallbacks, and writes answer and review files. |
| `data.py` | Loads CSVs, applies an optional schema map, and builds rule-ready customer, order, refund, shipment, support, inventory, procurement, return and contract tables. |
| `recipes.py` | Defines 70 parameterized deterministic metrics and formats answers. |
| `router.py` | Uses Qwen to select a recipe and parameters, validates them against the recipe catalog and data vocabulary, and runs a deterministic keyword cross-check. |
| `codegen.py` | Fallback that asks the model to write a calculation against prepared tables; screens and runs generated code with a time limit and reports vote disagreement. |
| `schema_map.py` | Proposes and validates actual-column-to-canonical-column mappings for a changed stage. |
| `hard_opt.py` | Exact branch-and-bound solver for fulfilment decision cases; includes a separate SciPy/HiGHS MILP checker. |
| `flow.py` | Deterministic replay of stateful warehouse events, including baseline and what-if cases. |
| `usage_summary.py` | Produces a redacted usage summary from supported model-call logs. |

`canonical_schema.json` describes expected table fields and types. `stage2_schema_map.json` maps renamed Stage 2 columns to those canonical fields. Fifteen tables have mappings; the map was checked against the Stage 2 data dictionary, with three tables manually corrected after the mapper flagged them.

Setup and run from `Project_5/pipeline`:

```powershell
pip install -r requirements.txt
ollama pull qwen2.5-coder:3b

# Stage 1: original canonical headers
python run.py --stage ../stage1 --out ../public_answers_qwen.csv --router ollama --votes 3

# Stage 2: renamed headers mapped to the same canonical schema
python run.py --stage ../stage2 --schema-map ../stage2_schema_map.json --out ../final_answers_qwen.csv --router ollama --votes 3
```

The Ollama server must be running locally. These commands do not use the SIXT gateway or `P5_API_KEY`. The 5-slide editable overview is `submission/Project_5_Two_Stage_Explanation.pptx`. `submission/Project_5_Full_Code_Walkthrough.pptx` is a separate 8-slide code walkthrough and is not the 5-slide submission version.

## 3. Validation and results

Code-level safeguards include explicit join-cardinality checks, uniqueness assertions, refund-allocation reconciliation, category and stock-policy interval checks, and errors for missing FX rates. The Stage 2 map is checked for known tables, canonical targets and duplicate target columns.

Six focused local-router and Stage 1/Stage 2 schema tests pass. The tests cover the no-authorization local path, quarter normalization, fallback routing, mapped versus unmapped headers, and Stage 2 map consistency. Python syntax checks also pass.

End-to-end smoke tests used the first 10 questions from each stage:

- Stage 1: 10/10 answer rows were written; P001, P003, P004, P006 and P008 were flagged for review.
- Stage 2: 10/10 answer rows were written; F001, F004 and F009 were flagged for review.

These are completion checks, not accuracy scores. No full 200-question run or official answer-key comparison is reported. The organizer-solved example suite and full Stage 2 data tests have not been run from the current workspace.

## 4. Model use and efficiency

| Model | Role | Parameters |
| --- | --- | --- |
| Qwen2.5-Coder 3B (`qwen2.5-coder:3b`, Ollama) | Routes Stage 1 and Stage 2 questions; supports Stage 2 column mapping and code-generation fallback. | 3.09B |
| Devin coding agent (Cognition), powered by Claude Fable 5.2 Medium | Development assistant used for method design, code, tests and report drafting; it does not run in the pipeline or produce answers directly. | Unknown; publisher does not disclose the count. |

The local Ollama model library entry reports the Qwen tag's Q4_K_M quantization and 3.09B parameters. The local model manifest digest was not archived. The router uses a 16K context and three route votes by default. Local model calls are recorded in `logs/ollama_calls.jsonl`; no SIXT API credential is read or sent in this mode.

Run time, memory, token totals and cost for a full two-stage run were not measured. The Ollama call log is local evidence and may contain challenge question text; do not publish it without authorization.

## 5. Small-model bonus declaration

Bonus claimed: no; eligibility is not claimed in this report.

The pipeline model is below the 40B parameter limit. The development assistant's total parameter count is not published, so this report does not assert that every generative model used is within the limit. The organizers determine eligibility.

Evidence locations: `MODEL_DECLARATION.csv`, `pipeline/router.py`, `pipeline/README.md`, and the local-only `logs/ollama_calls.jsonl`.

## 6. Limitations and reflection

- The 3B model can select an incorrect recipe or disagree across votes; review every report flag.
- The rules router covers known wording patterns and is not a replacement for human review.
- Generated-code fallback is less reliable than a tested recipe; its result can be written even when the agreement check fails, in which case it is flagged.
- Stage 1 and Stage 2 smoke tests covered only 10 questions each. They do not establish benchmark accuracy or validate all 70 recipes against official answers.
- Full runtime, memory and cost were not measured.
- A new optimization or simulation type may require a new dedicated engine rather than a generated-code fallback.

## 7. References and attribution

- Challenge data package, Stage 1: `problem7_student_start.zip`, SHA-256 `dcab020fd06195bcb60c92a61ed450296e2fed81f401ad4aeb5aa98b5cbcf7e0`, accessed 2026-11-13.
- Challenge data package, Stage 2: `problem7_student_final.zip`, SHA-256 `59417f1adf04e08a328c42c6f54450933b3032b0d0ff1398cfc01d2a169738da`, accessed 2026-11-13.
- Qwen Team. Qwen2.5-Coder-3B-Instruct model card, https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct.
- Ollama. `qwen2.5-coder:3b` library entry, https://ollama.com/library/qwen2.5-coder:3b.
- Hui et al. Qwen2.5-Coder Technical Report, arXiv:2409.12186, https://arxiv.org/abs/2409.12186.
- pandas, NumPy, SciPy/HiGHS.

## Business relevance and presentation

The method separates business definitions from question wording: analysts can review versioned rules and metric recipes, while users ask for supported measures in plain language. The same canonical calculations are reused across both stages, and review flags expose uncertain routes and fallbacks.

- Five-slide editable overview for both stages: `submission/Project_5_Two_Stage_Explanation.pptx`.
- Eight-slide full-code walkthrough for both stages: `submission/Project_5_Full_Code_Walkthrough.pptx`. This is a supporting presentation, not the five-slide submission version.
- PDF and HTML exports are intentionally omitted from the submission folder.

## 8. Publication follow-up after the Hackathon

Pending coordination with the organizers. Confirm the title, author list, timing, dataset citation and manuscript content before any upload. Do not include challenge datasets, golden answers or private evaluation material in a manuscript.

- Coordination status and organizer contact: pending
- Agreed manuscript title, author list and contributions: pending
- Agreed upload timing: pending
- Manuscript location/version reviewed with organizers: pending
- Confirmed dataset citation and references: pending
- arXiv upload confirmation and submission date: pending
- arXiv identifier or public link: pending
