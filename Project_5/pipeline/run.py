"""End-to-end run: questions CSV -> routes -> deterministic answers -> submission CSV.

Usage:
  python run.py --stage ../stage1 --out ../public_answers.csv --router llm     (small model routes; rules cross-check)
  python run.py --stage ../stage1 --out ../public_answers.csv --router rules   (no model; dry run / debugging only)
"""
import argparse
import csv
import json
from pathlib import Path

import recipes as R
from data import Data
from router import LLMRouter, rules_route, vocabulary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="../stage1")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", default="../public_answers.csv")
    ap.add_argument("--router", choices=["llm", "rules"], default="llm")
    ap.add_argument("--schema-map", default=None)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    stage = Path(a.stage)
    qfile = Path(a.questions) if a.questions else next(stage.glob("*questions.csv"))
    D = Data(stage / "datasets", a.schema_map)
    vocab = vocabulary(D)
    questions = list(csv.DictReader(qfile.open(encoding="utf-8-sig")))[: a.limit]
    llm = LLMRouter(vocab, votes=a.votes) if a.router == "llm" else None

    report, rows = [], []
    for q in questions:
        qid, text, atype = q["question_id"], q["question"], q.get("answer_type", "number")
        rule = rules_route(text, vocab)
        if llm:
            route, agreement = llm.route(text)
        else:
            route, agreement = rule, None
        entry = {"question_id": qid, "route": route, "rules_route": rule, "vote_agreement": agreement, "flags": []}

        def compute(rt):
            if not rt or not rt.get("recipe") or rt.get("error"):
                return None
            try:
                return R.run(D, rt["recipe"], rt["params"], atype)
            except Exception as e:
                entry["flags"].append(f"compute error {type(e).__name__}: {e}")
                return None

        answer, rule_answer = compute(route), compute(rule) if llm else None
        if llm:
            entry["rules_answer"] = rule_answer
            if agreement < 1:
                entry["flags"].append("model votes disagree")
            if answer != rule_answer:
                entry["flags"].append("model route and rules cross-check give different answers")
        if answer is None:
            entry["flags"].append("NO ANSWER")
        entry["answer"] = answer
        report.append(entry)
        rows.append((qid, answer if answer is not None else ""))
        print(qid, answer, "| FLAGS: " + "; ".join(entry["flags"]) if entry["flags"] else "")

    out = Path(a.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "answer"])
        w.writerows(rows)
    rep = out.with_name(out.stem + f"_report_{a.router}.json")
    rep.write_text(json.dumps(report, indent=1), encoding="utf-8")
    flagged = [e["question_id"] for e in report if e["flags"]]
    print(f"\nwrote {out} ({sum(bool(r[1]) for r in rows)}/{len(rows)} answered); report {rep.name}; flagged: {flagged or 'none'}")


if __name__ == "__main__":
    main()
