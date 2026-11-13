# Trial: Qwen2.5-Coder 3B as the question router

This folder is a side experiment by Prerana Prakash Naik. It is **not** part of the scored submission. The submitted answers are `Project_5/public_answers.csv` and `Project_5/final_answers.csv`, produced with Gemma 4 31B.

The trial replaces Gemma 4 31B with a much smaller local model, `qwen2.5-coder:3b` (3.09B parameters, run with Ollama), in the routing step of the same pipeline. The calculations are unchanged.

## Contents

| File | What it is |
| --- | --- |
| `stage1_answers_qwen2.5-coder-3b.csv` | Stage 1 answers (questions P001-P200) from the 3B-model run. Stage 1 only; there is no Stage 2 file. |
| `pipeline_qwen_3b/` | The pipeline variant used for the trial: a copy of `Project_5/pipeline/` from an earlier revision (70 recipes) with a local Ollama router added. |
| `method_notes_qwen_3b.md` | Method notes for the trial, in the method-report layout. Written before the full Stage 1 run, so its "not generated" and "10-question smoke test" statements describe an earlier state. |
| `slides_qwen_3b_two_stage.pptx` | Five-slide overview of the trial. |

## Result, compared with the Gemma 4 31B answers

Comparison of `stage1_answers_qwen2.5-coder-3b.csv` with `Project_5/public_answers.csv` (neither is checked against the private answer key):

- 200 of 200 questions answered.
- 190 of 200 answers are identical to the Gemma answers.
- 10 differ: P001, P010, P020, P035, P042, P045, P058, P086, P088, P091.
- In 5 of those 10 the answer has the wrong type for the question, so it cannot be correct: P010, P045 and P058 expect a name but contain a number; P042 and P088 expect true/false but contain a number.

We did not re-run the trial ourselves, and the model-call log for it is not included. Stage 2 was not run in full with this model.
