"""Verified core data layer: loads the stage CSVs and builds rule-compliant base tables.

Every business rule from business_rules.md that defines eligibility, identity, FX,
discount/refund allocation, category-at-order-time, SLA and shipment lateness is
applied exactly once here. Recipes only filter and aggregate these tables.
"""
import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

REF = pd.Timestamp("2025-01-01 00:00:00")
SLA_HOURS = {"urgent": 24, "high": 48, "medium": 72, "low": 120}


def _ts(s):
    return pd.to_datetime(s, format="ISO8601")


def _bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def resolve_email(email):
    e = str(email).strip().lower()
    local, _, domain = e.rpartition("@")
    local = local.split("+", 1)[0]
    if domain in ("gmail.com", "googlemail.com"):
        local, domain = local.replace(".", ""), "gmail.com"
    return f"{local}@{domain}"


class Data:
    def __init__(self, data_dir, schema_map=None):
        self.dir = Path(data_dir)
        # schema_map: {table: {actual_column: canonical_column}} for stage-2 column drift
        self.schema_map = schema_map or {}
        if isinstance(self.schema_map, (str, Path)):
            self.schema_map = json.loads(Path(self.schema_map).read_text())

    def raw(self, name):
        df = pd.read_csv(self.dir / f"{name}.csv", keep_default_na=True)
        return df.rename(columns=self.schema_map.get(name, {}))

    # ---------- FX ----------
    @cached_property
    def fx(self):
        f = self.raw("fx_rates")
        return {(m, c): r for m, c, r in zip(f.month, f.currency, f.eur_rate)}

    def fx_rate(self, ts, cur):
        """Vectorised monthly rate lookup; a missing non-EUR rate is an integrity error."""
        months = pd.Series(ts).dt.strftime("%Y-%m").to_numpy()
        out = np.array([self.fx.get(k, np.nan) for k in zip(months, np.asarray(cur))], dtype=float)
        if np.isnan(out).any():
            bad = {k for k in zip(months, np.asarray(cur)) if k not in self.fx}
            raise ValueError(f"missing FX rates: {sorted(bad)[:5]}")
        return out

    # ---------- customers / identity ----------
    @cached_property
    def accounts(self):
        a = self.raw("accounts")
        a["is_test_account"] = _bool(a.is_test_account)
        return a

    @cached_property
    def customers(self):
        c = self.raw("customers")
        c["is_test_customer"] = _bool(c.is_test_customer)
        c["signup_ts"] = _ts(c.signup_date)
        c["trial_ts"] = _ts(c.trial_start_date)
        c = c.merge(self.accounts[["account_id", "segment", "industry", "account_manager", "is_test_account"]],
                    on="account_id", how="left", validate="m:1")
        c["eligible"] = ~c.is_test_customer & ~c.is_test_account.fillna(False).astype(bool)
        c["identity"] = c.email.map(resolve_email)
        assert c.customer_id.is_unique
        return c

    @cached_property
    def canonical(self):
        """One row per resolved identity: earliest signup, then smallest customer_id."""
        c = self.customers
        c = c[c.eligible & (c.signup_ts <= REF)].sort_values(["signup_ts", "customer_id"])
        can = c.drop_duplicates("identity", keep="first").set_index("identity")
        assert can.index.is_unique
        return can

    @cached_property
    def cust_idx(self):
        return self.customers.set_index("customer_id")

    # ---------- orders ----------
    @cached_property
    def orders(self):
        o = self.raw("orders")
        o["order_ts"] = _ts(o.order_date)
        p = self.raw("payments")
        p["pay_ts"] = _ts(p.payment_date)
        settled = set(p[(p.payment_status_detail == "settled") & (p.pay_ts <= REF)].order_id)
        c = self.cust_idx
        for col in ["eligible", "identity", "country", "region", "account_id", "segment", "industry",
                    "acquisition_channel"]:
            o[col] = o.customer_id.map(c[col])
        o["valid"] = ((o.order_ts <= REF) & o.order_status.isin(["completed", "shipped"])
                      & (o.payment_status == "paid") & o.order_id.isin(settled)
                      & ~_bool(o.fraud_flag) & ~_bool(o.is_duplicate) & o.eligible.fillna(False).astype(bool))
        o["month"] = o.order_ts.dt.strftime("%Y-%m")
        assert o.order_id.is_unique
        return o

    @cached_property
    def refunds(self):
        """Approved refunds dated <= reference on a valid order whose product is on the order."""
        r = self.raw("refunds")
        r["refund_ts"] = _ts(r.refund_date)
        r = r[(r.refund_status == "approved") & (r.refund_ts <= REF)].copy()
        it = self.raw("order_items")
        valid = set(self.orders[self.orders.valid].order_id)
        pairs = set(zip(it.order_id, it.product_id))
        r = r[r.order_id.isin(valid) & np.array([k in pairs for k in zip(r.order_id, r.product_id)], dtype=bool)].copy()
        r["refund_eur"] = r.refund_amount_native * self.fx_rate(r.refund_ts, r.currency)
        o = self.orders.set_index("order_id")
        r["order_ts"], r["country"], r["region"] = r.order_id.map(o.order_ts), r.order_id.map(o.country), r.order_id.map(o.region)
        return r

    @cached_property
    def items(self):
        """Valid-order item lines with EUR gross, discounts, refund, net, cost, margin, category."""
        it = self.raw("order_items")
        n0 = len(it)
        o = self.orders
        cols = ["order_id", "order_ts", "month", "currency", "order_level_discount_native", "customer_id",
                "identity", "country", "region", "account_id", "segment", "industry", "valid"]
        it = it.merge(o[cols], on="order_id", how="inner", validate="m:1")
        it = it[it.valid].copy()
        rate = self.fx_rate(it.order_ts, it.currency)
        it["gross_native"] = it.quantity * it.unit_price_native
        it["pre_native"] = it.gross_native - it.line_discount_native
        tot = it.groupby("order_id").pre_native.transform("sum")
        it["od_native"] = np.where(tot != 0, it.order_level_discount_native * it.pre_native / tot.where(tot != 0, 1), 0.0)
        it["gross_eur"] = it.gross_native * rate
        it["line_disc_eur"] = it.line_discount_native * rate
        it["od_eur"] = it.od_native * rate
        # refunds: aggregate by (order, product), allocate over that product's lines
        ragg = self.refunds.groupby(["order_id", "product_id"]).refund_eur.sum().rename("ref_pair").reset_index()
        it = it.merge(ragg, on=["order_id", "product_id"], how="left", validate="m:1")
        it["ref_pair"] = it.ref_pair.fillna(0.0)
        w = it.pre_native.clip(lower=0)
        g = it.groupby(["order_id", "product_id"])
        wsum, cnt = w.groupby([it.order_id, it.product_id]).transform("sum"), g.order_item_id.transform("count")
        it["refund_eur"] = np.where(wsum > 0, it.ref_pair * w / wsum.where(wsum > 0, 1), it.ref_pair / cnt)
        it["net_eur"] = it.gross_eur - it.line_disc_eur - it.od_eur - it.refund_eur
        # category at order time (inclusive interval), exactly one match
        h = self.raw("product_category_history")
        h["vf"], h["vt"] = _ts(h.valid_from), _ts(h.valid_to)
        m = it[["order_item_id", "product_id", "order_ts"]].merge(h, on="product_id")
        m = m[(m.vf <= m.order_ts) & (m.order_ts <= m.vt)]
        assert m.order_item_id.is_unique, "overlapping category intervals"
        it["category"] = it.order_item_id.map(m.set_index("order_item_id").category)
        assert it.category.notna().all(), "item without category interval"
        p = self.raw("products").set_index("product_id")
        it["product_name"] = it.product_id.map(p.product_name)
        it["family"] = it.product_id.map(p.family)
        it["cost_eur"] = it.quantity * it.product_id.map(p.standard_cost) * self.fx_rate(it.order_ts, it.product_id.map(p.cost_currency))
        it["margin_eur"] = it.net_eur - it.cost_eur
        assert it.order_item_id.is_unique and len(it) <= n0
        assert abs(it.refund_eur.sum() - self.refunds.refund_eur.sum()) < 1e-6, "refund not allocated exactly once"
        return it

    @cached_property
    def vorders(self):
        """Valid orders with net revenue (one row per order)."""
        o = self.orders[self.orders.valid].copy()
        o["net_eur"] = o.order_id.map(self.items.groupby("order_id").net_eur.sum()).fillna(0.0)
        return o

    # ---------- customer status ----------
    @cached_property
    def cust_stats(self):
        """Per resolved identity with a canonical row: lifetime revenue, order stats, flags."""
        can = self.canonical.copy()
        o = self.vorders
        g = o.groupby("identity")
        can["ltv"] = g.net_eur.sum().reindex(can.index).fillna(0.0)
        can["n_orders"] = g.order_id.nunique().reindex(can.index).fillna(0)
        can["first_order"] = g.order_ts.min().reindex(can.index)
        can["last_order"] = g.order_ts.max().reindex(can.index)
        t = self.tickets
        unresolved = set(t[t.unresolved].identity)
        can["high_value"] = can.ltv >= 2500
        can["active"] = (can.n_orders >= 2) & (can.last_order >= REF - pd.Timedelta(days=90))
        can["at_risk"] = can.high_value & (can.last_order < REF - pd.Timedelta(days=75)) & can.index.isin(unresolved)
        return can

    @cached_property
    def account_ltv(self):
        """Eligible (non-test) accounts incl. zero revenue; revenue by the order's account."""
        a = self.accounts[~self.accounts.is_test_account].set_index("account_id").copy()
        a["ltv"] = self.vorders.groupby("account_id").net_eur.sum().reindex(a.index).fillna(0.0)
        a["high_value"] = a.ltv >= 7000
        return a

    # ---------- support ----------
    @cached_property
    def tickets(self):
        t = self.raw("support_tickets")
        t["opened_ts"], t["resolved_ts"] = _ts(t.opened_at), _ts(t.resolved_at)
        c = self.cust_idx
        for col in ["eligible", "identity", "country", "region", "account_id", "segment", "industry"]:
            t[col] = t.customer_id.map(c[col])
        t = t[t.eligible.fillna(False).astype(bool) & (t.opened_ts <= REF)].copy()
        t["unresolved"] = t.resolved_ts.isna() | (t.resolved_ts > REF)
        end = t.resolved_ts.where(~t.unresolved, REF)
        t["elapsed_h"] = (end - t.opened_ts) / pd.Timedelta(hours=1)
        t["breach"] = t.elapsed_h > t.priority.map(SLA_HOURS)
        assert t.ticket_id.is_unique
        return t

    # ---------- shipments ----------
    @cached_property
    def shipments(self):
        s = self.raw("shipments")
        s["ship_ts"], s["promise_ts"], s["deliv_ts"] = _ts(s.shipped_at), _ts(s.promised_delivery_date), _ts(s.delivered_at)
        o = self.vorders.set_index("order_id")
        s = s[s.order_id.isin(o.index) & (s.ship_ts <= REF)].copy()
        for col in ["identity", "country", "region", "account_id", "segment", "industry", "net_eur"]:
            s[col] = s.order_id.map(o[col])
        s["delivered"] = s.deliv_ts.notna() & (s.deliv_ts <= REF)
        eff = s.deliv_ts.where(s.delivered, REF)
        s["delay_days"] = ((eff.dt.normalize() - s.promise_ts.dt.normalize()) / pd.Timedelta(days=1)).clip(lower=0)
        s["late"] = (s.delay_days > 0) & (s.promise_ts < REF)
        s["ship_cost_eur"] = s.shipping_cost_native * self.fx_rate(s.ship_ts, s.currency)
        assert s.shipment_id.is_unique
        return s

    # ---------- marketing / web ----------
    @cached_property
    def touches(self):
        m = self.raw("marketing_touches")
        m["touch_ts"] = _ts(m.touch_date)
        c = self.cust_idx
        m["eligible"], m["identity"] = m.customer_id.map(c.eligible), m.customer_id.map(c.identity)
        m = m[m.eligible.fillna(False).astype(bool) & (m.touch_ts <= REF)].copy()
        m["cost_eur"] = m.cost_native * self.fx_rate(m.touch_ts, m.currency)
        return m

    @cached_property
    def attributed(self):
        """Valid orders with their 30-day last-touch attribution (campaign/channel or NaN)."""
        o = self.vorders[["order_id", "identity", "order_ts", "net_eur"]]
        m = o.merge(self.touches[["touch_id", "identity", "touch_ts", "campaign", "marketing_channel"]], on="identity")
        d = m.order_ts - m.touch_ts
        m = m[(d >= pd.Timedelta(0)) & (d <= pd.Timedelta(days=30))]
        m = m.sort_values(["touch_ts", "campaign", "touch_id"], ascending=[False, True, True]).drop_duplicates("order_id")
        return o.merge(m[["order_id", "touch_id", "touch_ts", "campaign", "marketing_channel"]], on="order_id", how="left", validate="1:1")

    @cached_property
    def sessions(self):
        w = self.raw("web_sessions")
        w["session_ts"] = _ts(w.session_date)
        c = self.cust_idx
        w["eligible"], w["identity"] = w.customer_id.map(c.eligible), w.customer_id.map(c.identity)
        return w[w.eligible.fillna(False).astype(bool) & (w.session_ts <= REF)].copy()

    # ---------- procurement ----------
    @cached_property
    def po_lines(self):
        l = self.raw("purchase_order_lines")
        l["ordered_ts"], l["promised_ts"] = _ts(l.ordered_at), _ts(l.promised_at)
        s = self.raw("suppliers")
        s["cs"], s["ce"] = _ts(s.contract_start), _ts(s.contract_end)
        s["is_test_supplier"] = _bool(s.is_test_supplier)
        l = l.merge(s[["supplier_id", "supplier_name", "country", "cs", "ce", "is_test_supplier"]], on="supplier_id", how="left", validate="m:1")
        ok = (l.status.isin(["open", "closed"]) & (l.ordered_ts <= REF) & ~l.is_test_supplier.fillna(True).astype(bool)
              & (l.cs <= l.ordered_ts) & (l.ordered_ts < l.ce))
        l = l[ok].copy()
        p = self.raw("products").set_index("product_id")
        l["family"] = l.product_id.map(p.family)
        l["unit_eur"] = l.unit_cost_native * self.fx_rate(l.ordered_ts, l.currency)
        l["std_eur"] = l.product_id.map(p.standard_cost) * self.fx_rate(l.ordered_ts, l.product_id.map(p.cost_currency))
        l["commit_eur"] = l.ordered_units * l.unit_eur
        assert l.po_line_id.is_unique
        return l

    @cached_property
    def receipts(self):
        r = self.raw("goods_receipts")
        r["received_ts"] = _ts(r.received_at)
        l = self.po_lines
        r = r.merge(l[["po_line_id", "supplier_id", "supplier_name", "country", "product_id", "family", "warehouse_region",
                       "ordered_ts", "promised_ts", "ordered_units", "unit_cost_native", "currency"]],
                    on="po_line_id", how="inner", validate="m:1")
        r = r[(r.status == "posted") & (r.ordered_ts <= r.received_ts) & (r.received_ts <= REF)].copy()
        r["cost_eur"] = r.accepted_units * r.unit_cost_native * self.fx_rate(r.received_ts, r.currency)
        assert r.receipt_id.is_unique
        return r

    # ---------- inventory ----------
    @cached_property
    def stock(self):
        """Reconciled end stock per product/warehouse with the policy effective at reference."""
        w = self.raw("warehouse_inventory")
        w["stocktake_ts"] = _ts(w.last_stocktake_date)
        assert not w.duplicated(["product_id", "warehouse_region"]).any()
        mv = self.raw("inventory_movements")
        mv["event_ts"] = _ts(mv.event_at)
        mv = mv[mv.status == "posted"].merge(w[["product_id", "warehouse_region", "stocktake_ts"]], on=["product_id", "warehouse_region"], how="inner")
        mv = mv[(mv.event_ts > mv.stocktake_ts) & (mv.event_ts <= REF)]
        self._movements = mv
        key = ["product_id", "warehouse_region"]
        w = w.merge(mv.groupby(key).units_delta.sum().rename("delta").reset_index(), on=key, how="left")
        w["end_stock"] = w.on_hand_units + w.delta.fillna(0)
        sp = self.raw("stock_policies")
        sp["vf"], sp["vt"] = _ts(sp.valid_from), _ts(sp.valid_to)
        sp = sp[(sp.vf <= REF) & (REF < sp.vt)]
        assert not sp.duplicated(key).any()
        w = w.merge(sp[key + ["reorder_point", "target_units"]], on=key, how="left", validate="1:1")
        assert w.reorder_point.notna().all()
        w["below"] = w.end_stock < w.reorder_point
        w["reorder_units"] = np.where(w.below, (w.target_units - w.end_stock).clip(lower=0), 0)
        p = self.raw("products").set_index("product_id")
        ref_month = pd.Series([REF] * len(w))
        w["value_eur"] = w.end_stock * w.product_id.map(p.standard_cost).to_numpy() * self.fx_rate(ref_month, w.product_id.map(p.cost_currency))
        return w

    @cached_property
    def movements(self):
        self.stock
        return self._movements

    # ---------- returns ----------
    @cached_property
    def refunds_cat(self):
        """Eligible refunds with the refunded product's category at original order time."""
        r = self.refunds.copy()
        cat = self.items.drop_duplicates(["order_id", "product_id"]).set_index(["order_id", "product_id"]).category
        r["category"] = [cat.get(k) for k in zip(r.order_id, r.product_id)]
        return r

    @cached_property
    def inspections(self):
        i = self.raw("return_inspections")
        i["insp_ts"] = _ts(i.inspected_at)
        r = self.refunds_cat[["refund_id", "refund_ts", "refund_reason", "category", "refund_eur"]]
        i = i.merge(r, on="refund_id", how="inner", validate="m:1")
        i = i[(i.status == "completed") & (i.refund_ts <= i.insp_ts) & (i.insp_ts <= REF)].copy()
        i["recovery_eur"] = i.restockable_units * i.recovery_unit_value_eur
        assert i.inspection_id.is_unique
        return i

    # ---------- worklogs / contracts ----------
    @cached_property
    def worklogs(self):
        w = self.raw("support_worklogs")
        w["worked_ts"] = _ts(w.worked_at)
        t = self.tickets[["ticket_id", "opened_ts", "issue_type", "support_channel", "country", "region", "account_id", "segment"]]
        w = w.merge(t, on="ticket_id", how="inner", validate="m:1")
        w = w[(w.work_status == "approved") & (w.opened_ts <= w.worked_ts) & (w.worked_ts <= REF)].copy()
        w["labor_eur"] = w.minutes_worked / 60 * w.hourly_rate_native * self.fx_rate(w.worked_ts, w.currency)
        w["is_billable"] = _bool(w.is_billable)
        assert w.worklog_id.is_unique
        return w

    @cached_property
    def contracts(self):
        """Selected (latest known) contract versions that are active, for non-test accounts."""
        c = self.raw("account_contract_versions")
        c["rec_ts"], c["vf"], c["vt"] = _ts(c.recorded_at), _ts(c.valid_from), _ts(c.valid_to)
        c = c[c.rec_ts <= REF].sort_values(["rec_ts", "revision"]).drop_duplicates("contract_id", keep="last")
        c = c[(c.status == "active") & c.account_id.isin(self.account_ltv.index)].copy()
        c["segment"] = c.account_id.map(self.account_ltv.segment)
        return c

    @cached_property
    def tickets_cov(self):
        """Eligible tickets with contract coverage and service credit."""
        t = self.tickets.copy()
        m = t[["ticket_id", "account_id", "opened_ts"]].merge(self.contracts[["account_id", "vf", "vt", "service_credit_eur"]], on="account_id")
        m = m[(m.vf <= m.opened_ts) & (m.opened_ts < m.vt)]
        assert m.ticket_id.is_unique, "overlapping contract intervals"
        t["credit"] = t.ticket_id.map(m.set_index("ticket_id").service_credit_eur)
        t["covered"] = t.credit.notna()
        t["credit_due"] = np.where(t.covered & t.breach, t.credit.fillna(0), 0.0)
        return t
