"""Question routing: business question -> {recipe, params}.

Primary router: a small language model (<= 40B parameters) called through an
OpenAI-compatible API. It reads the recipe catalogue and the vocabulary found in the
data and returns JSON. Output is validated against the catalogue and vocabulary, and
routed several times (self-consistency vote). Every call is logged to logs/*.jsonl.

Cross-check: `rules_route` is a deterministic keyword parser with no model. It never
replaces the model silently; disagreements are reported for human review.

Environment: P5_API_BASE (e.g. https://api.together.xyz/v1), P5_API_KEY, P5_MODEL.
"""
import json
import os
import random
import re
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

from recipes import RECIPES

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_LOCK = threading.Lock()


# ---------------- vocabulary from the data ----------------
def vocabulary(D):
    u = lambda t, c: sorted(D.raw(t)[c].dropna().astype(str).unique())
    return {
        "country": u("customers", "country"), "region": u("customers", "region"),
        "segment": u("accounts", "segment"), "industry": u("accounts", "industry"),
        "category": u("product_category_history", "category"), "family": u("products", "family") + ["All families"],
        "carrier": u("shipments", "carrier"), "channel_acquisition": u("customers", "acquisition_channel"),
        "channel_marketing": u("marketing_touches", "marketing_channel"), "campaign": u("marketing_touches", "campaign"),
        "issue_type": u("support_tickets", "issue_type"), "priority": u("support_tickets", "priority"),
        "support_channel": u("support_tickets", "support_channel"), "source": u("web_sessions", "traffic_source"),
        "refund_reason": u("refunds", "refund_reason"), "supplier_country": u("suppliers", "country"),
        "warehouse": u("warehouse_inventory", "warehouse_region"),
        "case_id": u("hard_opt_cases", "case_id") + u("flow_cases", "case_id"),
    }


NO_MATCH = "no_matching_recipe"
PARAM_VOCAB = {"country": "country", "region": "region", "segment": "segment", "industry": "industry",
               "category": "category", "family": "family", "carrier": "carrier", "campaign": "campaign",
               "issue_type": "issue_type", "priority": "priority", "support_channel": "support_channel",
               "source": "source", "refund_reason": "refund_reason", "supplier_country": "supplier_country",
               "warehouse": "warehouse", "case_id": "case_id"}


def validate(route, vocab):
    """Return (clean_route, error). Canonicalises values against the data vocabulary."""
    if isinstance(route, dict) and route.get("recipe") == NO_MATCH:
        return {"recipe": NO_MATCH, "params": {}}, None
    if not isinstance(route, dict) or route.get("recipe") not in RECIPES:
        return None, f"unknown recipe {route.get('recipe') if isinstance(route, dict) else route!r}"
    name, spec = route["recipe"], RECIPES[route["recipe"]]
    params = {}
    for k, v in (route.get("params") or {}).items():
        if k not in spec["params"] or v in (None, "", "null", "None"):
            continue
        v = str(v).strip() if not isinstance(v, (int, float)) else v
        if k == "channel":
            vk = "channel_marketing" if name == "top_attributed_campaign" else "channel_acquisition"
        else:
            vk = PARAM_VOCAB.get(k)
        if vk:
            match = [x for x in vocab[vk] if x.lower() == str(v).lower()]
            if not match:
                if k == "region" and str(v).lower().startswith("all"):
                    continue
                return None, f"{k}={v!r} not in data vocabulary {vocab[vk]}"
            v = match[0]
        elif k in ("period", "period_from", "period_to"):
            v = str(v).upper().replace(" 2024", "").strip()
            if not re.fullmatch(r"Q[1-4]|\d{4}-\d{2}", v):
                return None, f"bad period {v!r}"
        elif k == "month":
            if not re.fullmatch(r"\d{4}-\d{2}", str(v)):
                return None, f"bad month {v!r}"
        elif k in ("days", "threshold_pct"):
            v = float(v) if k == "threshold_pct" else int(float(v))
        params[k] = v
    required = [k for k, d in spec["params"].items() if "null" not in d]
    missing = [k for k in required if k not in params]
    if missing:
        return None, f"missing params {missing}"
    return {"recipe": name, "params": params}, None


# ---------------- small-model router ----------------
def build_system_prompt(vocab):
    lines = ["You route business questions to calculation recipes. You never calculate anything yourself.",
             "Choose exactly one recipe and fill its parameters using values copied from the question.",
             'Reply with JSON only: {"recipe": "<name>", "params": {...}}. Omit parameters that do not apply.',
             "Periods: quarters as Q1..Q4, months as YYYY-MM. 'customer region' values look like 'Western Europe';",
             "warehouses are single words like 'Central'. Use `country` for a country and `region` for a region, never both.",
             f'If no recipe computes exactly what the question asks, reply with recipe "{NO_MATCH}" and empty params.',
             "Never force a poor match: a wrong recipe gives a wrong answer.",
             "", "RECIPES:"]
    for name, spec in RECIPES.items():
        ps = ", ".join(f"{k} ({d})" for k, d in spec["params"].items()) or "no params"
        lines.append(f"- {name}: {spec['desc']} Params: {ps}")
    lines += ["", "ALLOWED VALUES:"] + [f"- {k}: {', '.join(v)}" for k, v in vocab.items()]
    return "\n".join(lines)


class LLMRouter:
    def __init__(self, vocab, model=None, base=None, key=None, votes=3, log_name="router_calls.jsonl"):
        self.vocab = vocab
        self.model = model or os.environ["P5_MODEL"]
        self.base = (base or os.environ["P5_API_BASE"]).rstrip("/")
        self.key = key or os.environ.get("P5_API_KEY", "")
        self.votes = votes
        self.system = build_system_prompt(vocab)
        LOG_DIR.mkdir(exist_ok=True)
        self.log_path = LOG_DIR / log_name

    def _call(self, messages, temperature, max_tokens=300):
        body = json.dumps({"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(f"{self.base}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        t0 = time.time()
        for attempt in range(60):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    out = json.loads(r.read())
                break
            except Exception as e:
                if attempt == 59:
                    raise
                wait = 2 * (attempt + 1)
                if getattr(e, "code", None) == 429:
                    wait = 5  # shared team quota: poll for a free slot
                    print("  rate limited, retrying in 5s", flush=True)
                time.sleep(wait)
        text = out["choices"][0]["message"]["content"]
        with _LOG_LOCK, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": self.model, "returned_model": out.get("model"),
                                "temperature": temperature, "user": messages[-1]["content"], "response": text,
                                "usage": out.get("usage"), "seconds": round(time.time() - t0, 2)}) + "\n")
        return text

    def _one(self, question, temperature):
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": question}]
        err = None
        for _ in range(3):
            text = self._call(messages, temperature)
            m = re.search(r"\{.*\}", text, re.S)
            try:
                route, err = validate(json.loads(m.group(0)), self.vocab)
            except Exception as e:
                route, err = None, f"invalid JSON ({e})"
            if route:
                return route
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That was rejected by the validator: {err}. Reply again with corrected JSON only."}]
        return {"recipe": None, "params": {}, "error": err}

    def route(self, question):
        """Self-consistency: route `votes` times; return (majority route, agreement share)."""
        outs = [self._one(question, 0.0 if i == 0 else 0.4) for i in range(self.votes)]
        keys = [json.dumps(o, sort_keys=True) for o in outs]
        best, n = Counter(keys).most_common(1)[0]
        return json.loads(best), n / len(outs)


    # ---- batched routing (few requests; needed under a shared request-rate limit) ----
    def _batch_once(self, items, temperature):
        """items: [(qid, question)] -> {qid: route-or-error-dict}"""
        result, pending, note = {}, list(items), ""
        for _ in range(3):
            if not pending:
                break
            user = (note + "Route each question below. Reply with a JSON array only, one object per question: "
                    '[{"id": "<id>", "recipe": "<name>", "params": {...}}, ...]\n\n'
                    + "\n".join(f"[{qid}] {text}" for qid, text in pending))
            text = self._call([{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                              temperature, max_tokens=120 * len(pending) + 200)
            try:
                arr = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
            except Exception as e:
                note = f"Your previous reply was not a valid JSON array ({e}). "
                continue
            errors, got = [], {}
            for obj in arr if isinstance(arr, list) else []:
                if isinstance(obj, dict):
                    got[str(obj.get("id"))] = obj
            for qid, _ in pending:
                route, err = validate(got.get(qid, {}), self.vocab) if qid in got else (None, "missing from reply")
                if route:
                    result[qid] = route
                else:
                    result[qid] = {"recipe": None, "params": {}, "error": err}
                    errors.append(f"{qid}: {err}")
            pending = [(qid, t) for qid, t in pending if result[qid].get("error")]
            note = "The validator rejected earlier answers: " + "; ".join(errors) + ". Correct them. "
        return result

    def route_many(self, items, batch_size=20):
        """Self-consistency over batches: {qid: (majority route, agreement share)}."""
        votes = {qid: [] for qid, _ in items}
        for v in range(self.votes):
            # votes after the first shuffle the questions into different batches, so each vote is a
            # genuinely different request (the gateway caches identical requests)
            order = list(items)
            if v:
                random.Random(v).shuffle(order)
            for i in range(0, len(order), batch_size):
                chunk = order[i:i + batch_size]
                out = self._batch_once(chunk, 0.0 if v == 0 else 0.4)
                for qid, _ in chunk:
                    votes[qid].append(json.dumps(out.get(qid, {"recipe": None, "params": {}, "error": "no reply"}), sort_keys=True))
                print(f"  vote {v + 1}/{self.votes}: routed {min(i + batch_size, len(items))}/{len(items)}", flush=True)
        final = {}
        for qid, ks in votes.items():
            best, n = Counter(ks).most_common(1)[0]
            final[qid] = (json.loads(best), n / len(ks))
        return final


# ---------------- deterministic cross-check (no model) ----------------
KEYWORDS = [
    ("fulfilment decision case", None), ("fulfilment case FLOW", None),
    ("purchase commitment", "procurement_commitment"), ("remain outstanding", "procurement_open_units"),
    ("accepted on time", "procurement_on_time_pct"), ("rejected-unit percentage", "procurement_rejection_pct"),
    ("purchase price variance", "procurement_price_variance"), ("EUR cost of accepted units", "procurement_received_cost"),
    ("highest rejected/", "procurement_worst_supplier"), ("total reconciled stock", "inventory_on_hand"),
    ("standard-cost EUR value", "inventory_value"), ("how many product/warehouse pairs", "inventory_below_reorder"),
    ("replenishment units", "inventory_reorder_units"), ("are restockable", "returns_restockable_pct"),
    ("total recovery value", "returns_recovered_value"),
    ("opened within 14 days after a late shipment", "tickets_after_late_delivery_breach_pct"),
    ("late shipments handled by", "carrier_weighted_delay"), ("which carrier has the highest", "top_delay_carrier"),
    ("highest margin rate", "top_margin_category"), ("margin rate percentage", "margin_rate_pct"),
    ("highest approved-refund rate", "top_refund_rate_category"), ("approved-refund rate as a percentage", "refund_rate_pct"),
    ("percentage growth in net revenue", "revenue_growth_pct"), ("Did net revenue in", "revenue_growth_at_least"),
    ("account manager", "top_manager_at_risk"), ("email domain", "top_email_domain"),
    ("summed monthly revenue targets", "revenue_vs_target_pct"), ("meet or exceed its net revenue target", "revenue_meets_target"),
    ("at least one late shipment and at least one SLA-breaching", "accounts_late_and_breach"),
    ("90th percentile", "account_ltv_p90"), ("Which region had the highest net-revenue z-score", "region_highest_zscore"),
    ("z-score", "revenue_zscore"), ("issue type occurred most often", "top_issue_high_value_accounts"),
    ("first valid order occurred in", "cohort_retention_pct"),
    ("time-correct product category generated the highest net revenue", "top_category_by_segment"),
    ("highest-lifetime-revenue account", "top_account_category_share"), ("Pearson correlation", "discount_refund_correlation"),
    ("web session", "web_conversion_pct"), ("last-touch attribution", "top_attributed_campaign"),
    ("high-value at-risk", "at_risk_customers"), ("before their first support ticket", "top_category_before_first_ticket"),
    ("median net revenue per valid order", "median_order_revenue_segment"), ("how many unique real customers", "unique_customers"),
    ("at least one eligible touch for campaign", "revenue_touch_and_ticket"), ("top 3 product categories", "top3_categories"),
    ("earliest valid order date", "earliest_order_by_channel"), ("which product name generated", "top_product_name"),
    ("after trial start", "trial_conversion_pct"), ("active real customers", "active_customers"),
    ("breached SLA", "sla_breach_pct"), ("total net revenue in EUR", "net_revenue"),
]


def rules_route(question, vocab):
    q = question
    name = None
    for kw, rec in KEYWORDS:
        if kw.lower() in q.lower():
            name = rec
            if kw == "fulfilment decision case":
                name = "opt_cost_change" if "change in minimum" in q else "opt_baseline_cost"
            elif kw == "fulfilment case FLOW":
                name = "flow_change" if "what-if" in q else "flow_baseline"
            break
    if not name:
        return None
    find = lambda key: next((v for v in sorted(vocab[key], key=len, reverse=True) if re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", q)), None)
    p = {}
    quarters = [f"Q{x}" for x in re.findall(r"Q([1-4]) 2024", q)]
    months = [m for m in re.findall(r"(?<!\d)(2024-\d{2})(?!-\d)", q)]
    want = RECIPES[name]["params"]
    if "region" in want:
        p["region"] = find("region")
    if "country" in want and not p.get("region"):
        p["country"] = find("country")
    if "period" in want:
        p["period"] = (quarters or months or [None])[0]
    if "period_from" in want:
        p["period_from"], p["period_to"] = quarters[0], quarters[1]
    if "month" in want:
        p["month"] = months[0]
    if "threshold_pct" in want:
        p["threshold_pct"] = float(re.search(r"at least (\d+(?:\.\d+)?)%", q).group(1))
    if "days" in want:
        p["days"] = int(re.search(r"within (\d+) days", q).group(1))
    for k in ("segment", "industry", "category", "carrier", "campaign", "refund_reason", "case_id"):
        if k in want:
            p[k] = find(k)
    if "family" in want:
        m = re.search(r"product family (.+?),", q)
        p["family"] = m.group(1).strip() if m else find("family")
    if "issue_type" in want:
        p["issue_type"] = re.search(r"issue (?:type )?'?([a-z_]+)'?", q).group(1)
    if "priority" in want:
        p["priority"] = re.search(r"priority '(\w+)'", q).group(1)
    if "source" in want:
        p["source"] = re.search(r"traffic source '(\w+)'", q).group(1)
    if "supplier_country" in want:
        p["supplier_country"] = re.search(r"suppliers in (\w+)", q).group(1)
    if "warehouse" in want:
        p["warehouse"] = re.search(r"warehouse (\w+)", q).group(1)
    if "channel" in want:
        key = "channel_marketing" if name == "top_attributed_campaign" else "channel_acquisition"
        m = re.search(r"(?:channel was|acquired through) ([A-Z][\w ]*?)(?:[?,.]| what)", q)
        p["channel"] = m.group(1).strip() if m else find(key)
    route, err = validate({"recipe": name, "params": p}, vocab)
    return route or {"recipe": name, "params": p, "error": err}
