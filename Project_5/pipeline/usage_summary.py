"""Redacted model-usage evidence: one row per model call, without question text or secrets."""
import csv
import json
import re
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"


def main():
    rows = []
    for name, purpose in (("router_calls.jsonl", "routing"), ("toy_calls.jsonl", "toy-example baseline"),
                          ("codegen_calls.jsonl", "code-writing fallback"), ("schema_map_calls.jsonl", "column mapping rehearsal")):
        if (LOGS / name).exists():
            rows += [dict(json.loads(line), purpose=purpose) for line in (LOGS / name).open(encoding="utf-8")]
    with (LOGS / "model_usage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "purpose", "model_requested", "model_returned", "temperature", "question_ids", "n_routes_returned",
                    "prompt_tokens", "completion_tokens", "seconds"])
        for r in rows:
            ids = re.findall(r"^\[(\w+)\]", r["user"], re.M) if r["purpose"] == "routing" else []
            u = r.get("usage") or {}
            w.writerow([r["ts"], r["purpose"], r["model"], r.get("returned_model"), r["temperature"],
                        f"{ids[0]}..{ids[-1]}" if ids else "", r["response"].count('"recipe"'),
                        u.get("prompt_tokens"), u.get("completion_tokens"), r["seconds"]])
    tot = lambda k: sum((r.get("usage") or {}).get(k, 0) or 0 for r in rows)
    print(f"{len(rows)} calls; prompt tokens {tot('prompt_tokens')}; completion tokens {tot('completion_tokens')}; "
          f"model seconds {sum(r['seconds'] for r in rows):.0f}")


if __name__ == "__main__":
    main()
