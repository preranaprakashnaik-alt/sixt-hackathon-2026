"""Column mapping for a new stage: the small model maps renamed columns to our canonical names.

The final stage uses the same business entities with different column names. For every table
whose header differs from the canonical schema, the model reads the stage's data dictionary,
the actual header and sample values, and proposes {actual_column: canonical_column}. Code
validates the proposal (columns exist, one-to-one, every missing canonical column covered,
value kinds match) and asks again on failure. The result is a JSON file for `run.py --schema-map`.

Usage:
  python schema_map.py --build-canonical ../stage1          (once, from the stage with known names)
  python schema_map.py --stage ../stage2 --out ../stage2_schema_map.json
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

CANONICAL = Path(__file__).resolve().parent / "canonical_schema.json"


def kind(series):
    s = series.dropna()
    if s.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(s) or set(s.astype(str).str.lower().unique()) <= {"true", "false"}:
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if s.astype(str).str.match(r"^\d{4}-\d{2}(-\d{2})?([ T]\d{2}:\d{2}(:\d{2})?)?$").all():
        return "date"
    return "text"


def build_canonical(stage_dir):
    schema = {}
    for f in sorted((Path(stage_dir) / "datasets").glob("*.csv")):
        df = pd.read_csv(f)
        schema[f.stem] = {c: kind(df[c]) for c in df.columns}
    CANONICAL.write_text(json.dumps(schema, indent=1), encoding="utf-8")
    return schema


def check(mapping, actual_df, canon, missing):
    if not isinstance(mapping, dict):
        return "reply must be a JSON object {actual_column: canonical_column}"
    for a, c in mapping.items():
        if a not in actual_df.columns:
            return f"'{a}' is not a column of the actual table"
        if c not in missing:
            return f"'{c}' is not one of the canonical columns that need a source: {sorted(missing)}"
        k = kind(actual_df[a])
        if k != "empty" and canon[c] != "empty" and k != canon[c]:
            return f"'{a}' holds {k} values but canonical '{c}' holds {canon[c]} values"
    if len(set(mapping.values())) != len(mapping):
        return "two actual columns map to the same canonical column"
    uncovered = missing - set(mapping.values())
    if uncovered:
        return f"no actual column given for canonical columns {sorted(uncovered)}"
    return None


def build_map(stage_dir, llm):
    """Returns (schema_map, report). Tables that already match are left out of the map."""
    stage_dir = Path(stage_dir)
    schema = json.loads(CANONICAL.read_text(encoding="utf-8"))
    dictionary = (stage_dir / "data_dictionary.md").read_text(encoding="utf-8")
    out, report = {}, {}
    for table, canon in schema.items():
        path = stage_dir / "datasets" / f"{table}.csv"
        if not path.exists():
            report[table] = "MISSING FILE: needs human review"
            continue
        df = pd.read_csv(path)
        missing = set(canon) - set(df.columns)
        if not missing:
            continue
        unknown = [c for c in df.columns if c not in canon]
        samples = "\n".join(f"- {c}: {', '.join(map(str, df[c].dropna().unique()[:3]))}" for c in unknown)
        user = (f"Table `{table}`.\nCanonical columns that need a source column: {sorted(missing)}\n"
                f"Actual columns not yet matched, with sample values:\n{samples}\n\n"
                "Using the data dictionary, map each actual column to the canonical column with the same meaning. "
                'Reply with JSON only: {"<actual_column>": "<canonical_column>", ...}. Leave out actual columns with no canonical counterpart.')
        messages = [{"role": "system", "content": "You align database columns. Data dictionary of the new stage:\n\n" + dictionary},
                    {"role": "user", "content": user}]
        err = "no reply"
        for _ in range(3):
            text = llm._call(messages, 0.0, max_tokens=600)
            try:
                mapping = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                err = check(mapping, df, canon, missing)
            except Exception as e:
                mapping, err = None, f"invalid JSON ({e})"
            if not err:
                out[table] = mapping
                report[table] = f"mapped {len(mapping)} columns"
                break
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": f"Rejected: {err}. Reply with corrected JSON only."}]
        else:
            report[table] = f"FAILED: {err}: needs human review"
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-canonical")
    ap.add_argument("--stage")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.build_canonical:
        print(f"canonical schema for {len(build_canonical(a.build_canonical))} tables written to {CANONICAL.name}")
        return
    from router import LLMRouter
    mapping, report = build_map(a.stage, LLMRouter({}, log_name="schema_map_calls.jsonl"))
    Path(a.out).write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    for t, r in report.items():
        print(t, "->", r)
    print(f"wrote {a.out}; {len(mapping)} tables mapped; {sum('review' in r for r in report.values())} need review")


if __name__ == "__main__":
    main()
