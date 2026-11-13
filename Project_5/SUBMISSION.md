# Submission: Project 5, Two Braincells

This report follows the event's `SUBMISSION_TEMPLATE.md`. The additional items that the Project 5 method-report template requires (submission hashes, small-model bonus declaration, validation detail, references, publication follow-up) are in the "Project 5 additions" part at the end.

## Authors

**Team name:** Two Braincells

**Team members and contributions:**

- Prerana Prakash Naik: owner of the team repository and of the submission. Explored a dynamic approach without a specific recipe for each Stage 1 question, and ran the trial with the smaller Qwen2.5-Coder 3B model to compare it with Gemma 4 31B.
- Harisankar Karamchand: method decisions (routing design, the request for a generalization fallback and its evaluation), coordination with the organizer on the model rules, model access, review of results.

**Project:** Project 5: From business question to correct answer (Enterprise Data Agent Challenge, TUM x SIXT)

## Problem and approach

**Problem.** Answer 200 business questions per stage from 37 linked CSV tables. Each question hides many rules: which orders count, which exchange rate applies, how discounts and refunds are split. Answers must be exact to 0.01.

**Main idea: the model reads, the code calculates.** A small open-weight model (Gemma 4 31B) reads each question and returns a short JSON "route" naming one calculation and its parameters, for example `{"recipe": "net_revenue", "params": {"country": "Germany", "period": "Q4"}}`. Ordinary Python code then computes the answer. On this main path the model never sees data rows and never does arithmetic.

```
question ──► Gemma 4 31B router ──► route JSON ──► validator ──► recipe function ──► formatter ──► answer CSV
             (3 independent votes,   (recipe +      (recipe exists,  (deterministic     (2 decimals, exact
              every call logged)      parameters)    values exist     pandas code)       counts, dates,
                    │                                in the data)          ▲             true/false)
                    │ "no matching recipe"                                 │
                    └──► Gemma writes solve(D) ──► restricted run ×3 ──► accepted if ≥2 agree, else flagged
                                                                           │
   37 CSV tables ──► column map (Gemma, checked by code) ──► verified data layer (rules applied once)
                                                              ├─ exact optimizer (questions 151-180)
                                                              └─ flow event simulator (questions 181-200)
```

- **Verified data layer.** The business rules (valid orders, identity resolution, exchange rates, discount and refund allocation, service-level breaches, stock reconciliation) are implemented once. The run stops if an assumption breaks.
- **Recipes.** 78 small parameterised functions, one per question type, on top of the data layer.
- **Two engines.** An exact solver for the fulfilment decision cases and an event simulator for the flow cases.
- **Fallback.** When the model judges that no recipe fits, it writes the calculation itself as a short pandas function against the verified tables. The code is accepted only when independent generations agree.
- **Column mapping.** For the final stage's renamed columns, the model maps them to our names from the data dictionary, and code checks the mapping.
- **People** review anything the checks flag and add a recipe when a new question type appears.

**Why this approach.** A wrong number that looks right is the main risk, so everything that must be exact is tested code. That leaves the model a task small enough for a 31B model.

**Assumptions.** The written business rules are complete and authoritative. Where a rule needed interpretation we followed the rule text, for example "All regions" is the union of regions. The flow simulation follows our reading of the hard-task rules.

## Models and why

| Model | Open-weight or closed | Where it runs | Main metric(s) | Cost and time per item | Chosen? |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 31B as router + recipes (`bedrock_mantle/google.gemma-4-31b`, 30.7B parameters) | open | API: SIXT hackathon LLM gateway | 10/10 organizer-solved examples; three votes agree on 400/400 questions | about 770 tokens and 1.6 s of model time per question for three votes; cost not measured | yes: exact arithmetic in code, cheapest use of the model |
| Gemma 4 31B writing the calculation itself (fallback path, recipes hidden) | open | API: same gateway | 10/10 organizer-solved examples; 58/59 against our recipes in Stage 1; 26/27 on the new Stage 2 question types | about 28,000 tokens and 24 s per question for three generations; cost not measured | fallback and cross-check only: less reliable than a tested recipe and about 37 times the tokens |
| Gemma 4 31B answering directly from raw tables (model-only baseline) | open | API: same gateway | 3/3 public toy examples (tables of three to five rows) | about 500 tokens per attempt | no: only feasible on tiny tables, and the model would do the arithmetic |
| Qwen2.5-Coder 3B as router (`qwen2.5-coder:3b`, 3.09B parameters; side trial, Stage 1 only) | open | local: Ollama on a laptop | 190/200 Stage 1 answers identical to the Gemma answers; 5 of the 10 differing answers have the wrong type for the question | not measured | no: less reliable routing; answers not submitted |
| Keyword parser (no model) | not a model | local | same answers as the Gemma router on 400/400 questions | none | cross-check only: it cannot handle rephrased questions |

Held-out data: the ten organizer-solved practice examples are the only questions with published answers; the 400 scored questions have a private key. Gemma 4 31B was the only open-weight model with a published parameter count on the gateway we were given. We also tried a much smaller local model, Qwen2.5-Coder 3B, as the router in a side trial on Stage 1 (`qwen_3b_trial/`, see its README). It matched the Gemma answers on 190 of 200 questions; the trial was run by one team member, was not re-run for this report, and its answers are not part of the submission.

Settings: temperature 0.0 for the first vote or generation, 0.4 and 0.7 for later ones; 20 questions per routing request; later votes shuffle the questions into different batches. Details are in `evidence/gemma_4_31b.md`.

The pipeline code was written with an AI coding assistant (Devin, Cognition, powered by Claude Fable 5.2 Medium), used with the organizer's approval. It does not run inside the pipeline. All three models are in `MODEL_DECLARATION.csv`.

## Results

Metrics named in the project README: accuracy on the answer CSVs (private key) and the small-model bonus. We have no official feedback, so our accuracy on the 400 scored questions is not known. What we measured ourselves:

| Check | Result |
| --- | --- |
| Organizer-solved practice examples reproduced exactly (Stage 1) | 10 / 10 |
| Baseline: model-only on the three public toy examples | 3 / 3 |
| Routing: three independent votes agree and match the keyword parser, both stages | 400 / 400 |
| Optimization cases: two independent exact solvers agree, both stages | 60 / 60 |
| Stage 1 recipes re-derived by the model with the recipes hidden | 58 / 59 |
| Stage 2 renamed columns mapped by the model and passing code checks | 79 / 79 |
| Stage 2 new question types re-derived by the model, matching our new recipes | 26 / 27 |
| Format check by the organizers' `check_submission.py`, both stages | 200 / 200 each |

What did not work, and the limits:

- **Static library.** 27 of the 200 final questions belonged to eight types we had not seen. We wrote eight new recipes on the day. The fallback agreed with them on 26 of 27, but we did not submit fallback answers and cannot claim that accuracy.
- **Flow simulation.** The 40 flow answers (20 per stage) have no independent check.
- **Shared data layer.** Agreement between two implementations on the same data layer would not reveal an error in that layer.
- **Reference answers.** Only the ten Stage 1 examples compare against organizer answers.
- **Vote independence.** The gateway replays identical requests, so our first "three votes" were effectively two. Later votes now shuffle the batches.

## How to run

From a fresh clone, with Python 3.12:

```
pip install -r Project_5/pipeline/requirements.txt
cd Project_5/pipeline

# Stage 1: extract ../releases/problem7_student_start.zip to Project_5/stage1/
python ../validation/test_examples.py
python run.py --stage ../stage1 --out ../public_answers.csv --router llm --votes 3
python ../stage1/check_submission.py --questions ../stage1/public_leaderboard_questions.csv --submission ../public_answers.csv

# Stage 2: extract ../releases/problem7_student_final.zip to Project_5/stage2/
python schema_map.py --stage ../stage2 --out ../stage2_schema_map.json
python run.py --stage ../stage2 --schema-map ../stage2_schema_map.json --out ../final_answers.csv --router llm --votes 3
python ../stage2/check_submission.py --questions ../stage2/final_questions.csv --submission ../final_answers.csv
```

- **Data location:** the two archives are in `Project_5/releases/`; the extracted folders `stage1/` and `stage2/` are git-ignored.
- **Environment variables (names only):** `P5_API_BASE`, `P5_API_KEY`, `P5_MODEL`.
- **Without model access:** `--router rules` runs the same calculations with the keyword parser and gives the same answers.
- **Run time:** about two minutes of calculation per stage on a laptop CPU, plus about five minutes of model time.

## Demo

`Project_5/submission/slides.pdf` (5 slides; source `submission/slides.html`).

## Next steps

- Let the fallback answer new question types unaided, and measure it on questions that need new base tables.
- Write a second, independent implementation of the hardest metrics and of the flow simulator.
- **For SIXT:** analysts encode a metric definition once as a tested function, and a small, cheap model lets anyone ask for it in plain language. Every answer records which calculation and parameters produced it, so numbers are reproducible and auditable, and the model can be swapped without changing any result.
- Turn this report into the arXiv manuscript with the organizers.

---

# Project 5 additions

These sections are required by the Project 5 method-report template (`METHOD_REPORT_TEMPLATE.md` in the stage package) and the event rules.

## A. Submission files and code revision

- `Project_5/public_answers.csv`: SHA-256 `13bce1c97d87da7b15bb562433fee8841914cdb583ac8eb80b43304b4272628e`
- `Project_5/final_answers.csv`: SHA-256 `f6c4d5c9fe33fd427b8a1dd1c08b9e5dd30d60b276f77544f903b9af914f42ab`
- Code revision used for the calculations: `1d5f562` for the final answers. The public answers were first produced at `f03116e`; every later revision reproduces that file byte for byte.

## B. Implementation files

Custom orchestration, no agent framework. Files in `Project_5/pipeline/`:

| File | Purpose |
| --- | --- |
| `data.py` | Loads the CSVs and builds rule-compliant base tables: eligible customers, resolved identities, valid orders, item-level net revenue and margin, order-time category, shipment lateness, ticket breaches, attribution, reconciled stock, eligible receipts, inspections, worklogs, contracts. |
| `recipes.py` | 78 parameterised functions, one per question type, plus answer formatting. Includes the five documented final-only recipes and eight added for question types that first appeared in the final stage. |
| `hard_opt.py` | Exact branch-and-bound solver for the fulfilment decision cases, with an independent SciPy/HiGHS integer-program check. |
| `flow.py` | Event replay for the flow simulations: known-version selection, event ordering, sticky reservations, dispatch, cancellations, returns, rebates, what-if edits. |
| `router.py` | The small-model router (catalogue prompt, validation against the data vocabulary, three votes, call logging) and the keyword parser used only as a cross-check. |
| `codegen.py` | Fallback: the model writes `solve(D)` against the verified tables, using the stage's own business rules; restricted execution; three generations, accepted when at least two agree. |
| `schema_map.py` | Column mapping for a stage with renamed columns; code checks the mapping (one-to-one, complete, matching value kinds). |
| `run.py` | End-to-end run; writes the answer CSV and a per-question report with flags. |
| `usage_summary.py` | Writes the redacted model-usage record. |

Checks and experiments are in `Project_5/validation/`: `test_examples.py` (the ten organizer-solved examples), `fallback_eval.py` (fallback with the recipes hidden), `stage2_crosscheck.py` (independent check of the recipes added for the final stage), `drift_test.py` (renamed-column rehearsal), `toy_examples.py` (model-only baseline) and `novel_questions.csv` (three invented questions with no recipe).

Table selection and rule application are not left to the model: each recipe is bound to the base tables it needs, and those tables already have the rules applied.

## C. Validation detail

These are our own checks, not official feedback.

**Checks built into the code**

- **Joins and duplicate counting.** Every merge declares its expected cardinality, primary keys are asserted unique, and payments are used only as an existence condition, so several settled payments never duplicate an order.
- **Record eligibility.** Test customers and accounts, invalid orders and events after the reference time are removed once, in the data layer.
- **Refund allocation.** The allocated refunds must sum to the eligible refund total, so each refund is subtracted exactly once.
- **Dates, categories and policies.** Exactly one category interval must match each item and exactly one stock policy each product/warehouse; otherwise the run stops.
- **Exchange rates.** A missing rate raises an error; it is never replaced by 1.
- **Rounding.** Full precision is kept until the final formatting step (two decimals, ties to even); counts must be integral.

**Stage 1**

- **Solved examples.** E001-E010 reproduce exactly. They cover about 10 of the 49 recipes used in questions 1-150.
- **Fallback with the recipes hidden.** For the ten solved examples and the first question of each of the 49 recipes used in questions 1-150 (59 questions), the model wrote the calculation itself, three generations each. Its majority answer matched the reference on 58 of 59, including 10 of 10 organizer answers. The agreement check accepted 57, all correct, and flagged 2 for review (one of them correct). On the first run 3 questions failed because our execution sandbox blocked harmless built-ins; after fixing the sandbox we re-ran only those 3 (`logs/fallback_eval_report.json`, `logs/fallback_eval_rerun.json`). For 49 of the 59 the reference is our own recipe answer, so this shows that two implementations agree, not that both are right. The test is also easier than a truly new question type, because the verified tables already contain the hard parts.
- **New questions end to end.** Three questions we invented that have no recipe were sent to the fallback by the router and answered correctly; a control question with a recipe was routed to its recipe.
- **Renamed-column rehearsal.** On a copy of Stage 1 with 29 columns renamed by us across 9 tables, the model mapped 29 of 29 correctly and the ten solved examples reproduced.

**Stage 2**

- **Rules.** The final-stage business rules and hard-task rules are identical to Stage 1 apart from the title line (file comparison), so no rule constants changed.
- **Renamed columns.** 15 of the 37 tables had renamed columns. The model mapped 79 columns; all passed the code checks and we reviewed the mapping by eye (`stage2_schema_map.json`). The data layer then built with every integrity assertion passing.
- **New question types.** 27 questions in eight new types (for example revenue concentration in the top tenth of customers, a refund exchange-rate counterfactual, a fully observed retention cohort). We added one recipe for each after reading the final questions.
- **Independent check of the new recipes.** With the recipes hidden, the model wrote the calculation itself for all 27, three generations each (`logs/stage2_crosscheck_report.json`). On 26 at least two generations agreed and matched the recipe. On F088 ("All regions" revenue concentration) one generation matched the recipe, one gave a different value and one failed; we kept the recipe answer.
- **Stage 1 unchanged.** After the Stage 2 additions, the pipeline still reproduces `public_answers.csv` byte for byte.

**Representative errors and fixes**

- The keyword parser first read "All families" from the explanatory sentence of every price-variance question, giving the same answer for three product families; it now reads the family named at the start of the question.
- The gateway returned cached replies for identical requests; later votes now shuffle the questions into different batches.
- The fallback sandbox first blocked built-ins that pandas needs internally; three generated programs failed for that reason, not because of the model.

## D. Model use and efficiency

| Model | Role | Parameters |
| --- | --- | --- |
| Gemma 4 31B (`bedrock_mantle/google.gemma-4-31b`) | Routes every question (three votes); maps renamed columns; writes the calculation code in the fallback and in the cross-checks | 30.7B (Google model card) |
| Devin coding agent (Cognition), powered by Claude Fable 5.2 Medium | Coding assistant: method design with the team, pipeline code, the eight Stage 2 recipes, checks and report drafts | Unknown (not published) |
| Qwen2.5-Coder 3B (`qwen2.5-coder:3b`, Ollama) | Side trial only: router for a Stage 1 run whose answers were not submitted | 3.09B (model card) |

Measured from the gateway's usage fields and wall-clock time per call (`logs/model_usage.csv`):

| Run | Model calls | Prompt tokens | Completion tokens | Model time |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 answers (routing, three votes) | 30 | about 132,000 | about 22,000 | about 310 s |
| Stage 2 answers (15 column-mapping calls, then routing with three votes) | 45 | about 188,000 | about 22,500 | about 320 s |
| Everything on the day, including experiments and cross-checks | 525 | about 3.9 million | about 226,000 | about 65 min |

Most of the day's tokens come from the code-writing experiments, where each call carries the table descriptions and business rules (about 9,000 tokens). Cost and memory were not measured. The model runs on the gateway's hardware, which is not disclosed to us.

## E. Small-model bonus declaration

Bonus claimed: yes

All generative models within scope have at most 40B total parameters: the pipeline model, Gemma 4 31B, has 30.7B, and the side-trial model, Qwen2.5-Coder 3B, has 3.09B. The coding assistant has no published parameter count; its use was approved by the organizer and we leave its treatment to the organizers' verification.

Evidence locations for model sizes, settings and usage records: `MODEL_DECLARATION.csv`, `evidence/gemma_4_31b.md`, `logs/model_usage.csv`, `public_answers_report_llm.json`, `final_answers_report_llm.json`, `stage2_schema_map.json`.

The qualifying model does the language-understanding work in both stages: for every question it selects one of 78 calculations and extracts the parameters. A wrong route gives a wrong answer, so its output determines every submitted value. In Stage 2 it also mapped the 79 renamed columns and independently re-derived the answers for the new question types. It is the model that writes the calculation code in the fallback path. No other fallback model is used.

## F. References and attribution

- Ioannis Tzachristas (2026). Enterprise Data Agent Challenge 2026: multi-table calculations, constrained allocation and event replay. Version 7.0.0. Unpublished local dataset. Accessed 2026-11-13. Event content revision `p5-local-v4-bonus40`, policy version `problem7-65-35-plus40-v2`. No DOI has been assigned.
  - Stage 1 archive `problem7_student_start.zip`, SHA-256 `dcab020fd06195bcb60c92a61ed450296e2fed81f401ad4aeb5aa98b5cbcf7e0`.
  - Stage 2 archive `problem7_student_final.zip`, SHA-256 `59417f1adf04e08a328c42c6f54450933b3032b0d0ff1398cfc01d2a169738da`.
- Gemma Team, Google DeepMind. Gemma 4 model card, https://ai.google.dev/gemma/docs/core/model_card_4; Gemma 4 Technical Report, arXiv:2607.02770.
- pandas 3.0.6, NumPy 2.5.3, SciPy 1.18.1 (HiGHS solver), Python 3.12.
- No external code was reused.

## G. Publication follow-up after the Hackathon

The method-report template allows this section to remain open at the final deadline. Status on 2026-11-13: not started.

- Coordination status and organizer contact: to be opened with Ioannis Tzachristas (ioannis.tzachristas@tum.de) after the event.
- Agreed manuscript title, author list and contributions: not yet agreed.
- Agreed upload timing: not yet agreed.
- Manuscript location/version reviewed with the organizers: none yet; the LaTeX starter is in the stage package under `arxiv_report/`.
- Confirmed dataset citation and other references: as in section F, to be confirmed with the organizers.
- arXiv upload confirmation and submission date: none yet.
- arXiv identifier or public link: none yet.
