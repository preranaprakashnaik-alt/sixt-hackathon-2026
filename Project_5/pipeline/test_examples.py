"""Regression test: the 10 organizer-solved practice examples must reproduce exactly."""
import sys

import recipes as R
from data import Data

CASES = [
    ("E001", "333368.46", "number", "net_revenue", dict(country="Germany", period="Q4")),
    ("E002", "11", "integer", "at_risk_customers", dict(country="France")),
    ("E003", "0.82", "number", "refund_rate_pct", dict(category="Laptop", period="Q3")),
    ("E004", "32.59", "number", "cohort_retention_pct", dict(month="2024-03", days=60)),
    ("E005", "Laptop;Phone;Tablet", "string", "top3_categories", dict(region="Western Europe", period="Q4")),
    ("E006", "1037509.54", "number", "procurement_received_cost", dict(warehouse="Central")),
    ("E007", "8453", "integer", "inventory_reorder_units", dict(warehouse="Central")),
    ("E008", "1178.44", "number", "returns_recovered_value", dict(category="Accessory")),
    ("E009", "16826.49", "number", "support_labor_cost", dict(issue_type="account")),
    ("E010", "54.08", "number", "returns_restockable_pct", dict(refund_reason="goodwill")),
]


def main(data_dir="../stage1/datasets"):
    D = Data(data_dir)
    fails = 0
    for qid, expected, atype, name, params in CASES:
        try:
            got = R.run(D, name, params, atype)
        except Exception as e:
            got = f"ERROR {type(e).__name__}: {e}"
        ok = got == expected
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {qid} expected={expected} got={got}")
    return fails


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
