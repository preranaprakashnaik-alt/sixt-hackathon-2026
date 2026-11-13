# Project 5 method report

Team name: TODO

Team members and contributions: TODO

Project: Project 5, Enterprise Data Agent Challenge (SIXT + TUM)

Public and final submission filenames and SHA-256 hashes:

- `Project_5/public_answers.csv`: `13bce1c97d87da7b15bb562433fee8841914cdb583ac8eb80b43304b4272628e`
- `Project_5/final_answers.csv`: pending (Stage 2 not yet released)

Code revision used for your calculations: `f03116e`

## 1. Approach in one paragraph

A small open-weight model (Gemma 4 31B, 30.7B parameters) reads each business question and decides which calculation it asks for and with which parameters; it returns a short JSON "route" such as `{"recipe": "net_revenue", "params": {"country": "Germany", "period": "Q4"}}`. Ordinary Python code then computes the answer from the CSV tables. The model never sees data rows and never does arithmetic. The business rules (valid orders, identity resolution, exchange rates, discount and refund allocation, service-level breaches, stock reconciliation) are implemented once in a verified data layer, and each question template is a small function on top of it. The 30 optimization questions are solved by an exact solver and the 20 flow questions by an event simulator. People intervene only where the checks raise a flag; in Stage 1 no question was flagged.

## 2. Architecture and implementation

```
question text ──► Gemma 4 31B router ──► route JSON ──► validator ──► recipe function ──► formatter ──► public_answers.csv
                  (3 votes, logged)      (recipe +      (recipe exists,   (deterministic      (2 decimals,
                                          parameters)    values exist      pandas code)        integers, dates,
                                                         in the data)           ▲              true/false)
                                                                               │
                              37 CSV tables ──► verified data layer (rules applied once)
                                                 ├─ exact optimizer (questions 151-180)
                                                 └─ flow event simulator (questions 181-200)
```

Custom orchestration, no agent framework. Files in `Project_5/pipeline/`:

| File | Purpose |
| --- | --- |
| `data.py` | Loads the CSVs and builds rule-compliant base tables: eligible customers, resolved identities, valid orders, item-level net revenue and margin, order-time category, shipment lateness, ticket breaches, attribution, reconciled stock, eligible receipts, inspections, worklogs, contracts. |
| `recipes.py` | One parameterised function per question template (70, including the five final-stage-only recipes), plus answer formatting. |
| `hard_opt.py` | Exact branch-and-bound solver for the fulfilment decision cases, with an independent SciPy/HiGHS integer-program check. |
| `flow.py` | Event replay for the flow simulations: known-version selection, event ordering, sticky reservations, dispatch, cancellations, returns, rebates, what-if edits. |
| `router.py` | The small-model router (catalogue prompt, validation against the data vocabulary, three votes, call logging) and a keyword parser used only as a cross-check. |
| `run.py` | End-to-end run; writes the answer CSV and a per-question report with flags. |
| `test_examples.py` | Regression test on the ten organizer-solved practice examples. |
| `usage_summary.py` | Writes the redacted model-usage record. |

Table selection and rule application are not left to the model: each recipe is bound to the base tables it needs, and those tables already have the rules applied.

Setup and run (Python 3.12):

```
pip install -r Project_5/pipeline/requirements.txt
# extract releases/problem7_student_start.zip to Project_5/stage1/
cd Project_5/pipeline
python test_examples.py
# environment variables: P5_API_BASE, P5_API_KEY, P5_MODEL
python run.py --stage ../stage1 --out ../public_answers.csv --router llm --votes 3
python ../stage1/check_submission.py --questions ../stage1/public_leaderboard_questions.csv --submission ../public_answers.csv
```

## 3. Validation and results

Our own checks (there has been no official feedback on scored answers):

- **Solved examples.** All ten practice examples with published answers (E001-E010) reproduce exactly.
- **Joins and duplicate counting.** Every merge declares its expected cardinality, primary keys are asserted unique, and payments are used only as an existence condition, so several settled payments never duplicate an order.
- **Refund allocation.** The data layer asserts that the allocated refunds sum to the eligible refund total, so each refund is subtracted exactly once.
- **Categories and policies.** Exactly one category interval must match each item and exactly one stock policy each product/warehouse; otherwise the run stops.
- **Exchange rates.** A missing rate raises an error; it is never replaced by 1.
- **Rounding.** Full precision is kept until the final formatting step (two decimals, ties to even); counts must be integral.
- **Optimization cases.** Two independent exact methods (branch-and-bound and an integer program) agree on all 15 cases, baseline and counterfactual.
- **Routing.** The model routed every question three times; all three votes agreed on all 200 questions, and the resulting answers equal those from the independent keyword parser on all 200.
- **CSV structure.** The organizers' `check_submission.py` reports 200 rows and 200 non-blank answers.

Representative errors found and fixed: the keyword parser first read "All families" from the explanatory sentence of every price-variance question, giving the same answer for three different product families; the fix was to read the family named at the start of the question.

Limits of this evidence: accuracy on the 200 scored questions is not measured, because the key is private. Only about 10 of roughly 90 question templates have a published answer to compare against. The flow simulation answers have no independent check; they rest on our reading of the rules.

## 4. Model use and efficiency

| Model | Role | Parameters |
| --- | --- | --- |
| Gemma 4 31B (`bedrock_mantle/google.gemma-4-31b`) | Routes every question to a recipe and parameters; three votes per question | 30.7B (Google model card) |
| Devin coding agent (Cognition), powered by Claude Fable 5.2 Medium | Coding assistant: method design with the team, pipeline code, tests and report drafts | Unknown (not published) |

See `MODEL_DECLARATION.csv`, `evidence/gemma_4_31b.md` and `logs/model_usage.csv`.

The coding assistant was used with the organizer's approval given on the day. It does not run inside the pipeline and did not produce answers directly. Both models are listed in `MODEL_DECLARATION.csv`.

Measured for the final three-vote Stage 1 run: 30 model calls of 20 questions each, about 132,000 prompt tokens and 22,000 completion tokens, about 70 seconds of model time (from the gateway's usage fields and wall-clock time per call, recorded in `logs/model_usage.csv`). Cost: not measured. The deterministic calculation of all 200 answers takes about two minutes on a laptop CPU; memory: not measured.

## 5. Small-model bonus declaration

Bonus claimed: yes

All generative models within scope have at most 40B total parameters: the pipeline model, Gemma 4 31B, has 30.7B. The coding assistant named in section 4 has no published parameter count; its use was approved by the organizer and we leave its treatment to the organizers' verification.

Evidence locations for model sizes, settings and representative usage/run records: `MODEL_DECLARATION.csv`, `evidence/gemma_4_31b.md`, `logs/model_usage.csv`, `public_answers_report_llm.json`.

The qualifying model does the language-understanding work of the workflow: for every question it selects one of 70 calculations and extracts the parameters (country, region, period, segment, case and so on). A wrong route gives a wrong answer, so its output determines every submitted value. Stage 2 will use the same model for routing and for mapping renamed columns to our canonical schema. No fallback model is used.

## 6. Limitations and reflection

- The recipe library covers the question templates we have seen plus the documented final-stage recipes. A genuinely new template in Stage 2 would need a new recipe.
- Some rules needed interpretation, for example which customer record defines geography for "customers in Italy who opened a ticket". Both readings gave the same answer here, but that is not guaranteed in Stage 2.
- The flow simulator is unverified against any reference.
- Agreement between the model and the keyword parser shows stable routing, not correct calculations.
- Design trade-off: we chose reliability and low cost over generality. The model's task is small enough for a 31B model, and each question costs under 1,000 tokens, but new question types need engineering work.
- With more time: a second, independently written implementation of the hardest metrics, and model-written SQL as a fallback for unseen templates.

## 7. References and attribution

- Ioannis Tzachristas (2026). Enterprise Data Agent Challenge 2026: multi-table calculations, constrained allocation and event replay. Version 7.0.0. Unpublished local dataset. Archive `problem7_student_start.zip`, SHA-256 `dcab020fd06195bcb60c92a61ed450296e2fed81f401ad4aeb5aa98b5cbcf7e0`, accessed 2026-11-13. Event content revision `p5-local-v4-bonus40`, policy version `problem7-65-35-plus40-v2`.
- Gemma Team, Google DeepMind. Gemma 4 model card, https://ai.google.dev/gemma/docs/core/model_card_4; Gemma 4 Technical Report, arXiv:2607.02770.
- pandas, NumPy, SciPy (HiGHS solver).

## Business relevance and reuse

SIXT could reuse the pattern directly: analysts encode a metric definition once as a tested function, and a small, cheap model lets anyone ask for it in plain language. Answers are reproducible and auditable because every number comes from versioned code, and the model can be swapped without changing any result.

## Demo

`Project_5/submission/slides.pdf`: TODO

## 8. Publication follow-up after the Hackathon

Pending.

- Coordination status and organizer contact: pending; Ioannis Tzachristas, ioannis.tzachristas@tum.de
- Agreed manuscript title, author list and contributions: pending
- Agreed upload timing: pending
- Manuscript location/version reviewed with the organizers: pending
- Confirmed dataset citation and other references: pending
- arXiv upload confirmation and submission date: pending
- arXiv identifier or public link, when available: pending
