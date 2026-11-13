"""Baseline experiment: let the small model answer the three public toy examples directly.

Unlike the pipeline, the model here sees the raw tables and must apply the rules and do
the arithmetic itself. This measures what the model can do without the deterministic layer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import csv
import re

from router import LLMRouter

EX = Path(__file__).resolve().parent.parent / "examples"
QUESTIONS = {
    "DEMO001": ("How many valid orders are there? For this example, a valid order must be completed or shipped "
                "and have at least one settled payment.", ["orders.csv", "payments.csv"]),
    "DEMO002": ("What is net revenue in EUR for this valid single-line order? The discounts are totals in USD. "
                "Use the order-month rate for the order amounts and the refund-month rate for the eligible approved "
                "refund. Round only the final answer to two decimals.", ["revenue_inputs.csv"]),
    "DEMO003": ("How many replenishment units are suggested? Reorder only when end stock is strictly below the reorder "
                "point; if triggered, replenish to target stock. End stock is opening stock plus posted movements "
                "inside the stocktake-to-reference window.", ["stock_policy.csv", "stock_movements.csv"]),
}
SYSTEM = ("You answer business questions from the CSV tables given. Think step by step, then finish with a last line "
          "of the form 'ANSWER: <value>' containing only the number.")


def main(runs=3):
    gold = {r["question_id"]: r["answer"] for r in csv.DictReader((EX / "demo_answers.csv").open())}
    llm = LLMRouter(vocab={}, log_name="toy_calls.jsonl")
    correct = 0
    for qid, (question, files) in QUESTIONS.items():
        tables = "\n\n".join(f"{f}:\n{(EX / f).read_text().strip()}" for f in files)
        for i in range(runs):
            text = llm._call([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{tables}\n\nQuestion: {question}"}],
                             [0.0, 0.4, 0.7][i % 3], max_tokens=1500)  # distinct settings: the gateway caches identical requests
            m = re.findall(r"ANSWER:\s*\**\s*(-?[\d.,]+)", text)
            got = m[-1].replace(",", "").rstrip(".") if m else None
            ok = got is not None and abs(float(got) - float(gold[qid])) <= 0.01
            correct += ok
            print(f"{qid} run {i + 1}: model={got} expected={gold[qid]} {'CORRECT' if ok else 'WRONG'}", flush=True)
    print(f"{correct}/{len(QUESTIONS) * runs} correct")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
