"""Measure the code-writing fallback: hide the recipes and let the small model write the calculation.

Test set: the ten organizer-solved examples (true answers) plus the first Stage 1 question of
every recipe used in P001-P150 (reference = our recipe answer, itself unverified).
Scenario questions (optimizer, flow simulation) are excluded: they need the dedicated engines.
"""
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import recipes as R
from codegen import CodeGen, describe_tables
from data import Data
from router import LLMRouter, rules_route, vocabulary

STAGE = Path(__file__).resolve().parent.parent / "stage1"


def same(a, b, atype):
    if a is None:
        return False
    if atype in ("number", "integer"):
        try:
            return abs(float(a) - float(b)) <= 0.01
        except ValueError:
            return False
    return " ".join(str(a).lower().split()) == " ".join(str(b).lower().split())


def main(samples=3, workers=6, only=None):
    D = Data(STAGE / "datasets")
    describe_tables(D)  # warm every base table before threads start
    vocab = vocabulary(D)
    llm = LLMRouter(vocab, log_name="codegen_calls.jsonl")
    cg = CodeGen(llm, D, STAGE, samples=samples)
    tests = [(r["question_id"], r["question"], r["answer_type"], r["answer"], "organizer answer")
             for r in csv.DictReader((STAGE / "solved_examples.csv").open(encoding="utf-8"))]
    seen = set()
    for q in csv.DictReader((STAGE / "public_leaderboard_questions.csv").open(encoding="utf-8")):
        rt = rules_route(q["question"], vocab)
        if int(q["question_id"][1:]) > 150 or rt["recipe"] in seen:
            continue
        seen.add(rt["recipe"])
        tests.append((q["question_id"], q["question"], q["answer_type"], R.run(D, rt["recipe"], rt["params"], q["answer_type"]), f"recipe {rt['recipe']}"))

    if only:
        tests = [t for t in tests if t[0] in only.split(",")]

    def work(t):
        qid, text, atype, ref, src = t
        out = cg.answer(text, atype)
        out.pop("code", None)
        return dict(question_id=qid, reference_source=src, reference=ref, correct=same(out["answer"], ref, atype), **out)

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(work, tests))
    for r in results:
        print(f"{r['question_id']} {'OK ' if r['correct'] else 'BAD'} accepted={r['accepted']} ref={r['reference']} got={r['samples']} ({r['reference_source']})", flush=True)
    n = len(results)
    acc = [r for r in results if r["accepted"]]
    gold = [r for r in results if r["reference_source"] == "organizer answer"]
    summary = {
        "questions": n, "samples_per_question": samples,
        "correct_majority_answer": sum(r["correct"] for r in results),
        "accepted_by_agreement_check": len(acc), "accepted_and_correct": sum(r["correct"] for r in acc),
        "accepted_but_wrong": sum(not r["correct"] for r in acc),
        "flagged_for_review": n - len(acc), "flagged_but_correct": sum(r["correct"] for r in results if not r["accepted"]),
        "organizer_solved_examples_correct": f"{sum(r['correct'] for r in gold)}/{len(gold)}",
    }
    print(json.dumps(summary, indent=1))
    (STAGE.parent / "logs" / ("fallback_eval_rerun.json" if only else "fallback_eval_report.json")).write_text(json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]), *sys.argv[3:4])
