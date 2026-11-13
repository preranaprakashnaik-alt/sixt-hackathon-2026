# Project 5 method report

Team name: Two Braincells

Team members and contributions:

- Prerana Prakash Naik: owner of the team repository and of the submission.
- Harisankar Karamchand: method decisions (routing design, the request for a generalization fallback and its evaluation), coordination with the organizer on the model rules, model access, review of results.

Project: Project 5, Enterprise Data Agent Challenge (SIXT + TUM)

Public and final submission filenames and SHA-256 hashes:

- `Project_5/public_answers.csv`: `13bce1c97d87da7b15bb562433fee8841914cdb583ac8eb80b43304b4272628e`
- `Project_5/final_answers.csv`: `f6c4d5c9fe33fd427b8a1dd1c08b9e5dd30d60b276f77544f903b9af914f42ab`

Code revision used for your calculations: `1d5f562` for the final answers. The public answers were first produced at `f03116e`; every later revision reproduces that file byte for byte.

## 1. Approach in one paragraph

A small open-weight model (Gemma 4 31B, 30.7B parameters) reads each business question and decides which calculation it asks for and with which parameters; it returns a short JSON "route" such as `{"recipe": "net_revenue", "params": {"country": "Germany", "period": "Q4"}}`. Ordinary Python code then computes the answer from the CSV tables. On this main path the model never sees data rows and never does arithmetic. The business rules (valid orders, identity resolution, exchange rates, discount and refund allocation, service-level breaches, stock reconciliation) are implemented once in a verified data layer, and each question type is a small function ("recipe") on top of it. The optimization questions are solved by an exact solver and the flow questions by an event simulator. The same model also maps renamed columns for a new stage, and, when it judges that no recipe fits, writes the calculation itself as a short pandas function against the verified tables; that code is accepted only when independent generations agree. People intervene in two places: they review anything the checks flag, and they add a recipe when a new question type appears. In neither stage was a question flagged, and no submitted answer came from the fallback.

## 2. Architecture and implementation

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

Custom orchestration, no agent framework. Files in `Project_5/pipeline/`:

| File | Purpose |
| --- | --- |
| `data.py` | Loads the CSVs and builds rule-compliant base tables: eligible customers, resolved identities, valid orders, item-level net revenue and margin, order-time category, shipment lateness, ticket breaches, attribution, reconciled stock, eligible receipts, inspections, worklogs, contracts. |
| `recipes.py` | 78 parameterised functions, one per question type, plus answer formatting. Includes the five documented final-only recipes and eight added for question types that first appeared in the final stage. |
| `hard_opt.py` | Exact branch-and-bound solver for the fulfilment decision cases, with an independent SciPy/HiGHS integer-program check. |
| `flow.py` | Event replay for the flow simulations: known-version selection, event ordering, sticky reservations, dispatch, cancellations, returns, rebates, what-if edits. |
| `router.py` | The small-model router (catalogue prompt, validation against the data vocabulary, three votes, call logging) and a model-free keyword parser used only as a cross-check. |
| `codegen.py` | Fallback for questions with no matching recipe: the model writes `solve(D)` against the verified tables, using the stage's own business rules; restricted execution; three generations, accepted when at least two agree. |
| `schema_map.py` | For a stage with renamed columns: the model maps actual columns to our canonical names from the stage's data dictionary; code checks the mapping (one-to-one, complete, matching value kinds). |
| `run.py` | End-to-end run; writes the answer CSV and a per-question report with flags. |
| `usage_summary.py` | Writes the redacted model-usage record. |

Checks and experiments are in `Project_5/validation/`: `test_examples.py` (the ten organizer-solved examples), `fallback_eval.py` (fallback with the recipes hidden), `stage2_crosscheck.py` (independent check of the recipes added for the final stage), `drift_test.py` (renamed-column rehearsal), `toy_examples.py` (model-only baseline on the public toy examples) and `novel_questions.csv` (three invented questions with no recipe).

Table selection and rule application are not left to the model: each recipe is bound to the base tables it needs, and those tables already have the rules applied.

Setup and run (Python 3.12; dependencies pinned in `pipeline/requirements.txt`):

```
pip install -r Project_5/pipeline/requirements.txt
# environment variables: P5_API_BASE, P5_API_KEY, P5_MODEL
cd Project_5/pipeline

# Stage 1: extract releases/problem7_student_start.zip to Project_5/stage1/
python ../validation/test_examples.py
python run.py --stage ../stage1 --out ../public_answers.csv --router llm --votes 3
python ../stage1/check_submission.py --questions ../stage1/public_leaderboard_questions.csv --submission ../public_answers.csv

# Stage 2: extract releases/problem7_student_final.zip to Project_5/stage2/
python schema_map.py --stage ../stage2 --out ../stage2_schema_map.json
python run.py --stage ../stage2 --schema-map ../stage2_schema_map.json --out ../final_answers.csv --router llm --votes 3
python ../stage2/check_submission.py --questions ../stage2/final_questions.csv --submission ../final_answers.csv
```

## 3. Validation and results

These are our own checks. We received no official feedback on scored answers, so our accuracy on the 400 scored questions is not known.

### Checks built into the code

- **Joins and duplicate counting.** Every merge declares its expected cardinality, primary keys are asserted unique, and payments are used only as an existence condition, so several settled payments never duplicate an order.
- **Record eligibility.** Test customers and accounts, invalid orders and events after the reference time are removed once, in the data layer.
- **Refund allocation.** The data layer asserts that the allocated refunds sum to the eligible refund total, so each refund is subtracted exactly once.
- **Dates, categories and policies.** Exactly one category interval must match each item and exactly one stock policy each product/warehouse; otherwise the run stops.
- **Exchange rates.** A missing rate raises an error; it is never replaced by 1.
- **Rounding.** Full precision is kept until the final formatting step (two decimals, ties to even); counts must be integral.
- **CSV structure.** The organizers' `check_submission.py` reports 200 rows and 200 non-blank answers for each stage.

### Stage 1

- **Solved examples.** All ten practice examples with published answers (E001-E010) reproduce exactly. They cover about 10 of the 49 recipes used in questions 1-150.
- **Routing.** The model routed every question three times, with the questions shuffled into different batches for each vote (the gateway replays identical requests, so unshuffled repeats would not be independent). All three votes agreed on all 200 questions, and the resulting answers equal those from the model-free keyword parser on all 200.
- **Optimization cases.** Two independent exact methods (branch-and-bound and an integer program) agree on all 30 answers (15 cases, baseline and counterfactual).
- **Fallback measured with the recipes hidden.** For the ten solved examples and the first question of each of the 49 recipes used in questions 1-150 (59 questions), the model wrote the calculation itself, three generations each. Its majority answer matched the reference on 58 of 59, including 10 of 10 organizer answers. The agreement check accepted 57, all correct, and flagged 2 for review (one of them correct). On the first run 3 questions failed because our execution sandbox blocked harmless built-ins; after fixing the sandbox we re-ran only those 3 (`logs/fallback_eval_report.json`, `logs/fallback_eval_rerun.json`). For 49 of the 59 the reference is our own recipe answer, so this shows that two implementations agree, not that both are right. The test is also easier than a truly new question type, because the verified tables already contain the hard parts (validity, allocation, lateness, at-risk flags).
- **New questions end to end.** Three questions we invented that have no recipe were sent to the fallback by the router and answered correctly; a control question with a recipe was routed to its recipe.
- **Renamed-column rehearsal.** On a copy of Stage 1 with 29 columns renamed by us across 9 tables, the model mapped 29 of 29 correctly and the ten solved examples reproduced.

### Stage 2

- **Rules.** The final-stage business rules and hard-task rules are identical to Stage 1 apart from the title line (checked with a file comparison), so no rule constants changed.
- **Renamed columns.** 15 of the 37 tables had renamed columns. The model mapped 79 columns from the final-stage data dictionary; all passed the code checks and we reviewed the mapping by eye (`stage2_schema_map.json`). The data layer then built with every integrity assertion passing.
- **New question types.** 27 of the 200 final questions belong to eight types that did not appear in Stage 1 (for example revenue concentration in the top tenth of customers, a refund exchange-rate counterfactual, a fully observed retention cohort). We added one recipe for each after reading the final questions.
- **Independent check of the new recipes.** With the recipes hidden, the model wrote the calculation itself for all 27 questions, three generations each (`logs/stage2_crosscheck_report.json`). On 26 questions at least two generations agreed and matched the recipe answer. On the remaining one (F088, "All regions" revenue concentration) one generation matched the recipe, one gave a different value and one failed; we kept the recipe answer, which follows the rule that "All regions" is the union of regions.
- **Routing.** Three independent votes agreed on all 200 final questions, and the answers equal those from the keyword parser on all 200. No question was sent to the fallback.
- **Optimization cases.** The two exact methods agree on all 30 final-stage answers.
- **Stage 1 unchanged.** After the Stage 2 additions, the pipeline still reproduces `public_answers.csv` byte for byte and the ten solved examples.
- **No reference answers.** The final stage has no solved examples, so nothing in it is checked against organizer answers.

### Baseline

Model-only baseline: given the raw tables of the three public toy examples and asked to reason and calculate by itself, Gemma 4 31B answered all three correctly at three temperature settings (`validation/toy_examples.py`). The question wording included the rule hints from the published explanations. These are tables of three to five rows; the result does not show that the model could do the same on the full dataset, which is why the pipeline keeps the arithmetic in code.

### Representative errors and fixes

- The keyword parser first read "All families" from the explanatory sentence of every price-variance question, giving the same answer for three product families; it now reads the family named at the start of the question.
- The gateway returned cached replies for identical requests, so our first "three votes" were effectively two; later votes now shuffle the questions into different batches.
- The fallback sandbox first blocked built-ins that pandas needs internally; three generated programs failed for that reason, not because of the model.

### Limits of this evidence

- The 40 flow-simulation answers (20 per stage) have no independent check; they rest on our reading of the rules.
- Agreement between two implementations that share the same data layer would not reveal an error in that layer.
- Only the ten Stage 1 examples compare against organizer answers.

## 4. Model use and efficiency

| Model | Role | Parameters |
| --- | --- | --- |
| Gemma 4 31B (`bedrock_mantle/google.gemma-4-31b`) | Routes every question to a recipe and parameters (three votes); maps renamed columns; writes the calculation code in the fallback and in the cross-checks | 30.7B (Google model card) |
| Devin coding agent (Cognition), powered by Claude Fable 5.2 Medium | Coding assistant: method design with the team, pipeline code, the eight Stage 2 recipes, checks and report drafts | Unknown (not published) |

Both are listed in `MODEL_DECLARATION.csv`; see also `evidence/gemma_4_31b.md` and `logs/model_usage.csv`. No other model was tried: Gemma 4 31B was the only open-weight model with a published parameter count on the gateway we were given. The coding assistant was used with the organizer's approval given on the day. It does not run inside the pipeline and did not produce answers directly.

Measured, from the gateway's usage fields and wall-clock time per call (`logs/model_usage.csv`):

| Run | Model calls | Prompt tokens | Completion tokens | Model time |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 answers (routing, three votes) | 30 | about 132,000 | about 22,000 | about 310 s |
| Stage 2 answers (15 column-mapping calls, then routing with three votes) | 45 | about 188,000 | about 22,500 | about 320 s |
| Everything on the day, including experiments and cross-checks | 525 | about 3.9 million | about 226,000 | about 65 min |

Most of the day's tokens come from the code-writing experiments, where each call carries the table descriptions and business rules (about 9,000 tokens). The deterministic calculation of 200 answers takes about two minutes on a laptop CPU. Cost and memory were not measured. The model runs on the gateway's hardware, which is not disclosed to us.

## 5. Small-model bonus declaration

Bonus claimed: yes

All generative models within scope have at most 40B total parameters: the pipeline model, Gemma 4 31B, has 30.7B. The coding assistant named in section 4 has no published parameter count; its use was approved by the organizer and we leave its treatment to the organizers' verification.

Evidence locations for model sizes, settings and representative usage/run records: `MODEL_DECLARATION.csv`, `evidence/gemma_4_31b.md`, `logs/model_usage.csv`, `public_answers_report_llm.json`, `final_answers_report_llm.json`, `stage2_schema_map.json`.

The qualifying model does the language-understanding work of the workflow in both stages: for every question it selects one of 78 calculations and extracts the parameters (country, region, period, segment, case and so on). A wrong route gives a wrong answer, so its output determines every submitted value. In Stage 2 it also mapped the 79 renamed columns and independently re-derived the answers for the new question types. It is the model that writes the calculation code in the fallback path. No other fallback model is used.

## 6. Limitations and reflection

- The final stage showed the limit of a static library: 27 questions needed eight new recipes, written on the day. The model's own code agreed with them on 26 of 27, which suggests the fallback could have answered most of them unaided, but we did not submit fallback answers and cannot claim that accuracy.
- The fallback is less reliable than a tested recipe and has only been measured on questions whose hard parts the data layer already solves. A new kind of optimization or simulation task would still need hand-written code.
- Some rules needed interpretation, for example whether "All regions" revenue concentration ranks customers across the union of regions (our reading) or within each region.
- The flow simulator is unverified against any reference.
- Agreement between the model and the keyword parser shows stable routing, not correct calculations.
- Design trade-off: we chose reliability and low cost over generality. The model's task is small enough for a 31B model, and routing costs under 1,000 tokens per question, but new question types need engineering work.
- With more time: a second, independently written implementation of the hardest metrics and of the flow simulator, and a fallback evaluation on question types that need new base tables.

## 7. References and attribution

- Ioannis Tzachristas (2026). Enterprise Data Agent Challenge 2026: multi-table calculations, constrained allocation and event replay. Version 7.0.0. Unpublished local dataset. Accessed 2026-11-13. Event content revision `p5-local-v4-bonus40`, policy version `problem7-65-35-plus40-v2`. No DOI has been assigned.
  - Stage 1 archive `problem7_student_start.zip`, SHA-256 `dcab020fd06195bcb60c92a61ed450296e2fed81f401ad4aeb5aa98b5cbcf7e0`.
  - Stage 2 archive `problem7_student_final.zip`, SHA-256 `59417f1adf04e08a328c42c6f54450933b3032b0d0ff1398cfc01d2a169738da`.
- Gemma Team, Google DeepMind. Gemma 4 model card, https://ai.google.dev/gemma/docs/core/model_card_4; Gemma 4 Technical Report, arXiv:2607.02770.
- pandas 3.0.6, NumPy 2.5.3, SciPy 1.18.1 (HiGHS solver), Python 3.12.
- No external code was reused.

## Business relevance and reuse

SIXT could reuse the pattern directly: analysts encode a metric definition once as a tested function, and a small, cheap model lets anyone ask for it in plain language. Answers are reproducible and auditable because every number comes from versioned code and each answer records which calculation and parameters produced it. The model can be swapped without changing any result.

## Presentation

`Project_5/submission/slides.pdf` (5 slides; source `submission/slides.html`). The slides show one question flowing through the system, the validation evidence, and the design choices and limits.

## 8. Publication follow-up after the Hackathon

The template allows this section to remain open at the final deadline. Status on 2026-11-13: not started.

- Coordination status and organizer contact: to be opened with Ioannis Tzachristas (ioannis.tzachristas@tum.de) after the event.
- Agreed manuscript title, author list and contributions: not yet agreed.
- Agreed upload timing: not yet agreed.
- Manuscript location/version reviewed with the organizers: none yet; the LaTeX starter is in the stage package under `arxiv_report/`.
- Confirmed dataset citation and other references: as in section 7, to be confirmed with the organizers.
- arXiv upload confirmation and submission date: none yet.
- arXiv identifier or public link: none yet.
