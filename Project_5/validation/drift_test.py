"""Rehearsal for the final stage's column-name drift.

Makes a temporary copy of Stage 1 with renamed columns (and a data dictionary rewritten to
the new names), lets the small model build the column map, compares it with the known
renames and re-runs the ten organizer-solved examples on the drifted copy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import shutil

import pandas as pd

import test_examples
from router import LLMRouter
from schema_map import build_map

ROOT = Path(__file__).resolve().parent.parent
RENAMES = {  # canonical -> drifted name
    "orders": {"order_date": "placed_on", "order_status": "fulfilment_state", "payment_status": "pay_state",
               "order_level_discount_native": "basket_discount_local", "fraud_flag": "suspected_fraud", "is_duplicate": "dup_marker"},
    "order_items": {"quantity": "qty", "unit_price_native": "price_each_local", "line_discount_native": "row_discount_local"},
    "customers": {"email": "contact_email", "country": "cust_country", "region": "cust_region",
                  "acquisition_channel": "acq_source", "signup_date": "registered_on", "is_test_customer": "qa_record"},
    "payments": {"payment_date": "paid_on", "payment_status_detail": "settlement_state"},
    "refunds": {"refund_date": "refunded_on", "refund_amount_native": "refund_value_local", "refund_status": "refund_state"},
    "shipments": {"shipped_at": "dispatch_date", "promised_delivery_date": "eta_date", "delivered_at": "arrival_date"},
    "support_tickets": {"opened_at": "created_ts", "resolved_at": "closed_ts", "priority": "severity"},
    "accounts": {"segment": "tier", "is_test_account": "qa_account"},
    "fx_rates": {"eur_rate": "to_eur"},
}


def main():
    src, dst = ROOT / "stage1", ROOT / "stage1_drift_test"
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "datasets").mkdir(parents=True)
    dictionary = (src / "data_dictionary.md").read_text(encoding="utf-8")
    for f in (src / "datasets").glob("*.csv"):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        ren = RENAMES.get(f.stem, {})
        df.rename(columns=ren).to_csv(dst / "datasets" / f.name, index=False)
    for ren in RENAMES.values():
        for old, new in ren.items():
            dictionary = dictionary.replace(f"`{old}`", f"`{new}`")
    (dst / "data_dictionary.md").write_text(dictionary, encoding="utf-8")
    try:
        mapping, report = build_map(dst, LLMRouter({}, log_name="schema_map_calls.jsonl"))
        truth = {t: {new: old for old, new in ren.items()} for t, ren in RENAMES.items()}
        total = sum(len(v) for v in truth.values())
        right = sum(mapping.get(t, {}).get(a) == c for t, m in truth.items() for a, c in m.items())
        for t, r in report.items():
            print(t, "->", r)
        print(f"columns mapped correctly: {right}/{total}")
        fails = test_examples.main(dst / "datasets", mapping)
        print(f"solved examples on drifted copy: {10 - fails}/10")
    finally:
        shutil.rmtree(dst, ignore_errors=True)


if __name__ == "__main__":
    main()
