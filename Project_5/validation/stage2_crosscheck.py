"""Independent check of the question types that first appeared in the final stage.

For every final-stage question answered by one of the recipes added for Stage 2, the small
model writes the calculation itself (recipes hidden) and the result is compared with the
recipe answer. There are no organizer answers for these, so agreement is evidence, not proof.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import csv
import json
from concurrent.futures import ThreadPoolExecutor

import recipes as R
from codegen import CodeGen, describe_tables
from data import Data
from fallback_eval import same
from router import LLMRouter, rules_route, vocabulary

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "stage2"
NEW = {"avg_days_touch_to_order", "latest_refund_date", "valid_order_count", "revenue_repeat_late_breached",
       "top_decile_revenue_share", "channel_marketing_roi_pct", "cohort_retention_fully_observed_pct", "refund_fx_counterfactual"}


def main(samples=3, workers=6):
    D = Data(STAGE / "datasets", ROOT / "stage2_schema_map.json")
    describe_tables(D)
    vocab = vocabulary(D)
    cg = CodeGen(LLMRouter(vocab, log_name="codegen_calls.jsonl"), D, STAGE, samples=samples)
    tests = []
    for q in csv.DictReader((STAGE / "final_questions.csv").open(encoding="utf-8")):
        rt = rules_route(q["question"], vocab)
        if rt and rt["recipe"] in NEW:
            tests.append((q["question_id"], q["question"], q["answer_type"], R.run(D, rt["recipe"], rt["params"], q["answer_type"]), rt["recipe"]))

    def work(t):
        qid, text, atype, ref, name = t
        out = cg.answer(text, atype)
        out.pop("code", None)
        return dict(question_id=qid, recipe=name, recipe_answer=ref, agrees=same(out["answer"], ref, atype), **out)

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(work, tests))
    for r in results:
        print(f"{r['question_id']} {'AGREE   ' if r['agrees'] else 'DISAGREE'} recipe={r['recipe_answer']} model_code={r['samples']} ({r['recipe']})", flush=True)
    summary = {"questions": len(results), "model_code_agrees_with_recipe": sum(r["agrees"] for r in results),
               "by_recipe": {n: f"{sum(r['agrees'] for r in results if r['recipe'] == n)}/{sum(r['recipe'] == n for r in results)}" for n in sorted(NEW)}}
    print(json.dumps(summary, indent=1))
    (ROOT / "logs" / "stage2_crosscheck_report.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
