"""Fallback for questions with no matching recipe: the small model writes the calculation.

The model receives the stage's own business rules, a description of the verified base
tables from data.py (rules for eligibility, identity, FX, allocation already applied) and
the question. It writes a short pandas function `solve(D)`. The code is executed in a
restricted namespace. Several independent generations are run; an answer is accepted only
when at least two executed results agree, otherwise the question is flagged.
"""
import re
import threading
from collections import Counter

import numpy as np
import pandas as pd

from data import REF
from recipes import format_answer

TABLE_DOCS = {
    "D.items": "One row per item line of a VALID order (all validity rules already applied). EUR amounts already use the "
               "correct monthly FX. net_eur = gross_eur - line_disc_eur - od_eur (allocated order discount) - refund_eur "
               "(allocated eligible approved refunds). margin_eur = net_eur - cost_eur. category is the category at order "
               "time. country/region/account_id/segment/industry come from the raw customer on the order. identity is "
               "the resolved customer identity. month is the order month 'YYYY-MM'.",
    "D.vorders": "One row per VALID order with net_eur (sum of its item lines), order_ts, month, identity, raw-customer "
                 "country/region/account_id/segment/industry/acquisition_channel.",
    "D.cust_stats": "One row per real (resolved, eligible) customer, indexed by identity, taken from its canonical raw row: "
                    "country, region, account_id, account_manager, segment, acquisition_channel, signup_ts, trial_ts; plus "
                    "ltv (lifetime valid-order net revenue EUR across aliases), n_orders, first_order, last_order and the "
                    "boolean flags high_value, active, at_risk as defined in the rules.",
    "D.account_ltv": "One row per eligible (non-test) account, indexed by account_id, including zero-revenue accounts: segment, "
                     "industry, account_country, account_region, account_manager, ltv (lifetime net revenue EUR), high_value.",
    "D.tickets": "Eligible support tickets (eligible raw customer, opened at or before reference): ticket_id, customer_id, "
                 "identity, opened_ts, resolved_ts, issue_type, priority, support_channel, country, region, account_id, "
                 "segment, industry (from the ticket's raw customer), unresolved (bool), elapsed_h, breach (SLA breached, bool).",
    "D.tickets_cov": "D.tickets plus contract coverage: covered (bool), credit (service credit EUR of the covering contract), "
                     "credit_due (credit if covered and breached, else 0).",
    "D.shipments": "Eligible shipments (valid order, shipped at or before reference): shipment_id, order_id, carrier, "
                   "warehouse_region, ship_ts, promise_ts, deliv_ts, delivered (bool, delivered by reference), delay_days, "
                   "late (bool), ship_cost_eur, net_eur (net revenue of the linked order), and the order's identity, country, "
                   "region, account_id, segment, industry.",
    "D.touches": "Eligible marketing touches: touch_id, customer_id, identity, touch_ts, marketing_channel, campaign, cost_eur.",
    "D.attributed": "One row per valid order with its 30-day last-touch attribution: order_id, identity, order_ts, net_eur, "
                    "touch_id, campaign, marketing_channel (NaN when unattributed).",
    "D.sessions": "Eligible web sessions: session_id, customer_id, identity, session_ts, traffic_source, page_views, minutes_on_site.",
    "D.po_lines": "Eligible purchase order lines (eligible supplier, open/closed, ordered by reference): po_line_id, supplier_id, "
                  "supplier_name, country (supplier country), product_id, family, warehouse_region, ordered_ts, promised_ts, "
                  "ordered_units, unit_cost_native, currency, unit_eur (purchase unit cost EUR at purchase month), std_eur "
                  "(product standard cost EUR at purchase month), commit_eur (ordered_units * unit_eur).",
    "D.receipts": "Eligible posted goods receipts joined to their purchase line: receipt_id, po_line_id, received_ts, "
                  "accepted_units, rejected_units, cost_eur (accepted units * unit price * receipt-month FX), supplier_id, "
                  "supplier_name, country (supplier country), product_id, family, warehouse_region, ordered_ts, promised_ts, "
                  "ordered_units.",
    "D.stock": "One row per product/warehouse: product_id, warehouse_region, on_hand_units, end_stock (reconciled at reference), "
               "reorder_point, target_units (policy effective at reference), below (end_stock < reorder_point), reorder_units "
               "(suggested replenishment), value_eur (end stock at standard cost, 2025-01 FX).",
    "D.movements": "Posted inventory movements after the stocktake and through reference: movement_id, product_id, "
                   "warehouse_region, event_ts, movement_type (receipt, return_restock, sale_issue, adjustment), units_delta.",
    "D.refunds_cat": "Eligible approved refunds on valid order/product pairs: refund_id, order_id, product_id, refund_ts, "
                     "refund_reason, refund_eur, category (order-time category of the refunded product).",
    "D.inspections": "Eligible completed return inspections joined to their eligible refund: inspection_id, refund_id, "
                     "warehouse_region, insp_ts, units_received, restockable_units, recovery_eur, disposition, refund_reason, "
                     "category (order-time category), refund_eur.",
    "D.worklogs": "Eligible approved support worklogs joined to their eligible ticket: worklog_id, ticket_id, worked_ts, "
                  "minutes_worked, labor_eur, is_billable (bool), issue_type, support_channel, country, region, account_id, segment.",
    "D.contracts": "Selected (latest known at reference) ACTIVE contract versions of non-test accounts: contract_id, account_id, "
                   "segment, vf (valid_from), vt (valid_to, exclusive), service_credit_eur, quarterly_fee_eur, included_support_minutes.",
    "D.raw('sales_targets')": "Raw table: month ('YYYY-MM'), country, region, net_revenue_target_eur.",
    "D.raw('products')": "Raw table: product_id, product_name, initial_category, family, list_price_eur, standard_cost, cost_currency.",
}

SYSTEM = """You are a careful data analyst. You write one short Python function that answers a business question.

Rules of the task:
- Define `def solve(D):` and return the answer value (a number, a string, a bool, or the string "N/A").
- `pd` (pandas), `np` (numpy) and `REF` (reference time, pandas Timestamp 2025-01-01) are already available. Do not import anything.
- Use ONLY the prepared tables listed below. Eligibility, validity, identity resolution, FX conversion and discount/refund
  allocation are ALREADY applied in them. Do not re-apply those rules and do not read raw files.
- Do not round; return full precision. Percentages are 0-100. Dates as 'YYYY-MM-DD' strings. Rankings: break ties alphabetically.
- Quarters are calendar quarters of 2024 (Q1 = 2024-01..2024-03). Filter by the `month` column where available.
- Follow the business rules below for definitions (thresholds, statistics, tie-breaks).
- Reply with one ```python code block and nothing else.
"""

BANNED = re.compile(r"\bimport\b|__|\bopen\s*\(|\bexec\s*\(|\beval\s*\(|subprocess|\bos\.|\bsys\.|globals\s*\(|getattr\s*\(")
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in
                 ["len", "sum", "min", "max", "abs", "round", "sorted", "set", "list", "dict", "tuple", "range", "enumerate",
                  "zip", "float", "int", "str", "bool", "any", "all", "map", "filter", "isinstance", "print", "reversed", "hasattr",
                  # pandas imports lazily inside some methods (e.g. strftime); generated code itself may not import (see BANNED)
                  "__import__"]}


def describe_tables(D):
    out = []
    for expr, doc in TABLE_DOCS.items():
        df = eval(expr, {"D": D})
        cols = []
        for c in df.columns:
            s = df[c]
            hint = ""
            if s.dtype == object or str(s.dtype).startswith("str"):
                vals = s.dropna().unique()
                if len(vals) <= 12:
                    hint = " in {" + ", ".join(map(str, sorted(vals))) + "}"
            cols.append(f"{c}{hint}")
        idx = f" Index: {df.index.name}." if df.index.name else ""
        out.append(f"### {expr}\n{doc}{idx}\nColumns: {'; '.join(cols)}")
    return "\n\n".join(out)


def rules_excerpt(stage_dir):
    """The stage's own metric definitions (everything from 'Retained metric definitions' on)."""
    text = (stage_dir / "business_rules.md").read_text(encoding="utf-8")
    i = text.find("## Retained metric definitions")
    return text[i if i >= 0 else 0:]


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m[-1] if m else text).strip()


def execute(code, D):
    if BANNED.search(code):
        raise ValueError("code uses a forbidden construct (imports, file access or dunder names)")
    ns = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, "REF": REF}
    exec(code, ns)
    box = {}

    def runner():
        try:
            box["v"] = ns["solve"](D)
        except Exception as e:
            box["e"] = e
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(60)
    if t.is_alive():
        raise TimeoutError("solve() ran longer than 60 seconds")
    if "e" in box:
        raise box["e"]
    v = box["v"]
    if isinstance(v, (np.generic,)):
        v = v.item()
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return v


class CodeGen:
    def __init__(self, llm, D, stage_dir, samples=3):
        self.llm, self.D, self.samples = llm, D, samples
        self.system = SYSTEM + "\n## PREPARED TABLES\n\n" + describe_tables(D) + "\n\n## BUSINESS RULES\n\n" + rules_excerpt(stage_dir)

    def _one(self, question, answer_type, temperature):
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"Question: {question}\nExpected answer type: {answer_type}"}]
        err = None
        for _ in range(2):
            text = self.llm._call(messages, temperature, max_tokens=1200)
            code = extract_code(text)
            try:
                return format_answer(execute(code, self.D), answer_type), code, None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"Running your code failed with: {err}. Fix it and reply with the full corrected code block only."}]
        return None, code, err

    def answer(self, question, answer_type="number"):
        """Returns dict(answer, agreement, accepted, samples=[...])."""
        temps = [0.0, 0.4, 0.7, 0.9, 1.0]
        runs = [self._one(question, answer_type, temps[i % len(temps)]) for i in range(self.samples)]
        answers = [a for a, _, _ in runs if a is not None]
        if not answers:
            return {"answer": None, "agreement": 0.0, "accepted": False, "samples": [r[0] for r in runs], "errors": [r[2] for r in runs]}
        best, n = Counter(answers).most_common(1)[0]
        return {"answer": best, "agreement": n / self.samples, "accepted": n >= 2, "samples": [r[0] for r in runs],
                "errors": [r[2] for r in runs], "code": next(c for a, c, _ in runs if a == best)}
