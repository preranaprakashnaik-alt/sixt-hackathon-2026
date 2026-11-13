"""Recipe library: one parameterised, deterministic function per question template.

Each recipe is registered with a description and its parameter names. The small model
only selects a recipe and fills parameters; all numbers come from this code.
"""
import numpy as np
import pandas as pd

from data import REF

RECIPES = {}
NA = "N/A"


def recipe(desc, **params):
    def deco(fn):
        RECIPES[fn.__name__] = {"fn": fn, "desc": desc, "params": params}
        return fn
    return deco


def qmonths(q):
    q = int(str(q).upper().replace("Q", ""))
    return [f"2024-{m:02d}" for m in range(3 * q - 2, 3 * q + 1)]


def period_months(period):
    """'Q3' -> its three 2024 months; '2024-05' -> ['2024-05']; None -> None (lifetime)."""
    if period in (None, "", "lifetime"):
        return None
    p = str(period).strip()
    return qmonths(p) if p.upper().startswith("Q") else [p]


def _geo(df, country=None, region=None):
    if country:
        df = df[df.country == country]
    if region and not str(region).lower().startswith("all"):
        df = df[df.region == region]
    return df


def _items(D, country=None, region=None, period=None, category=None, segment=None):
    it = _geo(D.items, country, region)
    months = period_months(period)
    if months:
        it = it[it.month.isin(months)]
    if category:
        it = it[it.category == category]
    if segment:
        it = it[it.segment == segment]
    return it


def _top(series, ascending=False):
    """Highest value, alphabetical tie-break."""
    s = series.dropna()
    if s.empty:
        return NA
    df = s.rename("v").reset_index()
    df.columns = ["k", "v"]
    return df.sort_values(["v", "k"], ascending=[ascending, True]).k.iloc[0]


def _pct(num, den):
    return NA if den == 0 else 100 * num / den


# ================= revenue =================
@recipe("Total net revenue EUR for a customer country OR customer region during a period (quarter like Q3, or month like 2024-05).",
        country="country or null", region="region or null", period="Q1-Q4 or YYYY-MM")
def net_revenue(D, country=None, region=None, period=None):
    return _items(D, country, region, period).net_eur.sum()


@recipe("Percentage growth in net revenue from one quarter to another for a customer country or region.",
        country="country or null", region="region or null", period_from="Q1-Q4", period_to="Q1-Q4")
def revenue_growth_pct(D, country=None, region=None, period_from="Q3", period_to="Q4"):
    a, b = net_revenue(D, country, region, period_from), net_revenue(D, country, region, period_to)
    return NA if a == 0 else 100 * (b - a) / a


@recipe("Boolean: did net revenue for a country/region grow between two quarters by at least threshold_pct percent?",
        country="country or null", region="region or null", period_from="Q1-Q4", period_to="Q1-Q4", threshold_pct="number")
def revenue_growth_at_least(D, country=None, region=None, period_from="Q3", period_to="Q4", threshold_pct=0):
    g = revenue_growth_pct(D, country, region, period_from, period_to)
    return NA if g == NA else bool(g >= float(threshold_pct))


def _target(D, country, region, period):
    t = D.raw("sales_targets")
    t = _geo(t, country, region)
    return t[t.month.isin(period_months(period))].net_revenue_target_eur.sum()


@recipe("Net revenue as a percentage of the summed monthly revenue targets for a country/region and period.",
        country="country or null", region="region or null", period="Q1-Q4 or YYYY-MM")
def revenue_vs_target_pct(D, country=None, region=None, period=None):
    return _pct(net_revenue(D, country, region, period), _target(D, country, region, period))


@recipe("Boolean: did a country/region meet or exceed its net revenue target in a period?",
        country="country or null", region="region or null", period="Q1-Q4 or YYYY-MM")
def revenue_meets_target(D, country=None, region=None, period=None):
    return bool(net_revenue(D, country, region, period) >= _target(D, country, region, period))


def _z(D, region, month):
    it = _geo(D.items, None, region)
    m = it.groupby("month").net_eur.sum()
    base = np.array([m.get(f"2024-{i:02d}", 0.0) for i in range(1, 10)])
    sd = base.std(ddof=0)
    return NA if sd == 0 else (m.get(month, 0.0) - base.mean()) / sd


@recipe("Z-score of one month's net revenue for a region relative to that region's Jan-Sep 2024 monthly distribution.",
        region="region", month="YYYY-MM")
def revenue_zscore(D, region, month):
    return _z(D, region, month)


@recipe("Which region had the highest net-revenue z-score in a given month (vs its own Jan-Sep 2024 distribution).",
        month="YYYY-MM")
def region_highest_zscore(D, month):
    zs = {r: _z(D, r, month) for r in sorted(D.customers.region.dropna().unique())}
    return _top(pd.Series({k: v for k, v in zs.items() if v != NA}, dtype=float))


@recipe("Top 3 product categories by net revenue in a region/country and period, semicolon separated, descending.",
        country="country or null", region="region or null", period="Q1-Q4 or YYYY-MM")
def top3_categories(D, country=None, region=None, period=None):
    s = _items(D, country, region, period).groupby("category").net_eur.sum().reset_index()
    s = s.sort_values(["net_eur", "category"], ascending=[False, True])
    return ";".join(s.category.head(3))


@recipe("Which product NAME generated the highest net revenue among valid orders in a region/country during a period.",
        country="country or null", region="region or null", period="Q1-Q4 or YYYY-MM")
def top_product_name(D, country=None, region=None, period=None):
    return _top(_items(D, country, region, period).groupby("product_name").net_eur.sum())


@recipe("Which time-correct product category generated the highest net revenue among valid orders from accounts in a segment during a period.",
        segment="account segment", period="Q1-Q4 or YYYY-MM")
def top_category_by_segment(D, segment, period=None):
    return _top(_items(D, period=period, segment=segment).groupby("category").net_eur.sum())


@recipe("Median net revenue per valid order for accounts in a segment.", segment="account segment")
def median_order_revenue_segment(D, segment):
    o = D.vorders[D.vorders.segment == segment]
    return NA if o.empty else float(np.median(o.net_eur))


@recipe("90th percentile of account lifetime net revenue among accounts in a segment (includes zero-revenue accounts).",
        segment="account segment")
def account_ltv_p90(D, segment):
    v = np.sort(D.account_ltv[D.account_ltv.segment == segment].ltv.to_numpy())
    if len(v) == 0:
        return NA
    i = (len(v) - 1) * 0.9
    lo = int(np.floor(i))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


@recipe("For the highest-lifetime-revenue account in a segment, percentage of that account's net revenue from a category.",
        segment="account segment", category="product category")
def top_account_category_share(D, segment, category):
    a = D.account_ltv[D.account_ltv.segment == segment]
    acc = _top(a.ltv)
    it = D.items[D.items.account_id == acc]
    return _pct(it[it.category == category].net_eur.sum(), it.net_eur.sum())


@recipe("Which normalized (resolved) email domain contributed the most valid-order net revenue.")
def top_email_domain(D):
    o = D.vorders
    return _top(o.groupby(o.identity.str.split("@").str[1]).net_eur.sum())


@recipe("Earliest valid order date among orders whose ordering RAW customer's acquisition channel is given.",
        channel="acquisition channel")
def earliest_order_by_channel(D, channel):
    o = D.vorders[D.vorders.acquisition_channel == channel]
    return NA if o.empty else o.order_ts.min().strftime("%Y-%m-%d")


@recipe("Count of valid orders placed by customers in a country or region during a period.",
        country="country or null", region="region or null", period="Q1-Q4 or YYYY-MM or null")
def valid_order_count(D, country=None, region=None, period=None):
    o = _geo(D.vorders, country, region)
    months = period_months(period)
    return int(len(o[o.month.isin(months)] if months else o))


@recipe("Net revenue in a period for a country/region from valid orders whose resolved customer has at least two lifetime valid orders, at least one late shipment on a valid order, and at least one support ticket unresolved and SLA-breached at the reference time.",
        country="country or null", region="region or null", period="Q1-Q4 or YYYY-MM")
def revenue_repeat_late_breached(D, country=None, region=None, period=None):
    o = D.vorders
    n = o.groupby("identity").order_id.nunique()
    ids = set(n[n >= 2].index) & set(D.shipments[D.shipments.late].identity) & set(D.tickets[D.tickets.unresolved & D.tickets.breach].identity)
    sel = _geo(o[o.identity.isin(ids)], country, region)
    return sel[sel.month.isin(period_months(period))].net_eur.sum()


@recipe("Revenue concentration: percentage of lifetime net revenue contributed by the top ceil(10% of customer count) resolved customers with strictly positive revenue in a region (or 'All regions').",
        region="region or 'All regions'")
def top_decile_revenue_share(D, region=None):
    rev = _geo(D.vorders, None, region).groupby("identity").net_eur.sum()
    rev = rev[rev > 0].rename("v").reset_index().sort_values(["v", "identity"], ascending=[False, True])
    if rev.empty:
        return NA
    k = int(np.ceil(0.1 * len(rev)))
    return _pct(rev.v.head(k).sum(), rev.v.sum())


@recipe("Change in total net revenue EUR (counterfactual minus actual) if each eligible approved refund used the ORDER-month FX rate instead of the refund-month FX rate, for a region (or 'All regions').",
        region="region or 'All regions'")
def refund_fx_counterfactual(D, region=None):
    r = _geo(D.refunds, None, region)
    cf = r.refund_amount_native * D.fx_rate(r.order_ts, r.currency)
    return -(cf - r.refund_eur).sum()


@recipe("Fully observed cohort retention: among resolved customers whose first valid order was in a month and at least N complete days before the reference time, percentage with another distinct valid order strictly after the first and no later than N days after it.",
        month="YYYY-MM", days="integer")
def cohort_retention_fully_observed_pct(D, month, days):
    o = D.vorders[D.vorders.identity.isin(D.canonical.index)]
    first = o.groupby("identity").order_ts.min()
    win = pd.Timedelta(days=int(days))
    cohort = first[(first.dt.strftime("%Y-%m") == month) & (REF - first >= win)]
    f = o.identity.map(cohort)
    later = set(o[f.notna() & (o.order_ts > f) & (o.order_ts - f <= win)].identity)
    return _pct(len(later), len(cohort))


# ================= margin / refunds =================
@recipe("Margin rate percentage (total margin / total net revenue) for valid order items in a category, optional period.",
        category="product category", period="Q1-Q4, YYYY-MM or null")
def margin_rate_pct(D, category, period=None):
    it = _items(D, period=period, category=category)
    return _pct(it.margin_eur.sum(), it.net_eur.sum())


@recipe("Which product category had the highest margin rate during a period.", period="Q1-Q4, YYYY-MM or null")
def top_margin_category(D, period=None):
    g = _items(D, period=period).groupby("category")[["margin_eur", "net_eur"]].sum()
    g = g[g.net_eur != 0]
    return _top(g.margin_eur / g.net_eur)


@recipe("Approved-refund rate as percentage of gross item revenue for a category, optional period.",
        category="product category", period="Q1-Q4, YYYY-MM or null")
def refund_rate_pct(D, category, period=None):
    it = _items(D, period=period, category=category)
    return _pct(it.refund_eur.sum(), it.gross_eur.sum())


@recipe("Which product category had the highest approved-refund rate among valid order items from customers in a country/region.",
        country="country or null", region="region or null")
def top_refund_rate_category(D, country=None, region=None):
    g = _items(D, country, region).groupby("category")[["refund_eur", "gross_eur"]].sum()
    g = g[g.gross_eur != 0]
    return _top(g.refund_eur / g.gross_eur)


@recipe("Latest approved (eligible) refund date for valid order items in an order-time product category.", category="product category")
def latest_refund_date(D, category):
    r = D.refunds_cat[D.refunds_cat.category == category]
    return NA if r.empty else r.refund_ts.max().strftime("%Y-%m-%d")


@recipe("Pearson correlation across product categories between item (line) discount rate and approved-refund rate.")
def discount_refund_correlation(D):
    g = D.items.groupby("category")[["line_disc_eur", "refund_eur", "gross_eur"]].sum()
    x, y = g.line_disc_eur / g.gross_eur, g.refund_eur / g.gross_eur
    if len(g) < 2 or x.std() == 0 or y.std() == 0:
        return NA
    return float(np.corrcoef(x, y)[0, 1])


# ================= customers =================
@recipe("Count of unique real (identity-resolved, non-test) customers in a country or region.",
        country="country or null", region="region or null")
def unique_customers(D, country=None, region=None):
    return len(_geo(D.cust_stats, country, region))


@recipe("Count of high-value at-risk real customers in a country or region as of the reference date.",
        country="country or null", region="region or null")
def at_risk_customers(D, country=None, region=None):
    return int(_geo(D.cust_stats, country, region).at_risk.sum())


@recipe("Count of active real customers in a country or region as of the reference date.",
        country="country or null", region="region or null")
def active_customers(D, country=None, region=None):
    return int(_geo(D.cust_stats, country, region).active.sum())


@recipe("Which account manager owns the largest total lifetime net revenue among at-risk real customers.")
def top_manager_at_risk(D):
    c = D.cust_stats[D.cust_stats.at_risk]
    return _top(c.groupby("account_manager").ltv.sum())


@recipe("Cohort retention: among real customers whose first valid order was in a month, percentage with another valid order within N days.",
        month="YYYY-MM", days="integer")
def cohort_retention_pct(D, month, days):
    o = D.vorders[D.vorders.identity.isin(D.canonical.index)].sort_values(["order_ts", "order_id"])
    first = o.groupby("identity").order_ts.transform("min")
    rank = o.groupby("identity").cumcount()
    cohort = o[(rank == 0) & (first.dt.strftime("%Y-%m") == month)].identity
    later = o[(rank > 0) & (o.order_ts - first <= pd.Timedelta(days=int(days)))].identity
    return _pct(cohort.isin(set(later)).sum(), len(cohort))


@recipe("Trial conversion: percentage of real customers acquired through a channel whose first valid order came within 30 days after trial start.",
        channel="acquisition channel")
def trial_conversion_pct(D, channel):
    c = D.cust_stats
    c = c[(c.acquisition_channel == channel) & c.trial_ts.notna() & (c.trial_ts <= REF)]
    d = c.first_order - c.trial_ts
    conv = (d >= pd.Timedelta(0)) & (d <= pd.Timedelta(days=30))
    return _pct(conv.sum(), len(c))


@recipe("Web conversion: among real customers with a web session from a traffic source, percentage with a valid order within 14 days after such a session.",
        source="traffic source")
def web_conversion_pct(D, source):
    s = D.sessions[D.sessions.traffic_source == source]
    m = s[["identity", "session_ts"]].merge(D.vorders[["identity", "order_ts"]], on="identity")
    d = m.order_ts - m.session_ts
    conv = set(m[(d >= pd.Timedelta(0)) & (d <= pd.Timedelta(days=14))].identity)
    ids = set(s.identity)
    return _pct(len(conv), len(ids))


@recipe("Lifetime valid-order net revenue for resolved customers having at least one eligible touch for a campaign AND at least one eligible support ticket of an issue type.",
        campaign="campaign", issue_type="ticket issue type")
def revenue_touch_and_ticket(D, campaign, issue_type):
    ids = set(D.touches[D.touches.campaign == campaign].identity) & set(D.tickets[D.tickets.issue_type == issue_type].identity)
    return D.vorders[D.vorders.identity.isin(ids)].net_eur.sum()


@recipe("For customers in a country who opened at least one support ticket, the product category with highest net revenue from valid order items before their first ticket.",
        country="country")
def top_category_before_first_ticket(D, country, mode="identity"):
    t = D.tickets
    if mode == "identity":
        first = t.groupby("identity").opened_ts.min()
        it = _items(D, country=country)
        ft = it.identity.map(first)
    else:
        first = t.groupby("customer_id").opened_ts.min()
        it = _items(D, country=country)
        ft = it.customer_id.map(first)
    return _top(it[ft.notna() & (it.order_ts < ft)].groupby("category").net_eur.sum())


@recipe("Using 30-day last-touch attribution, which campaign generated the highest attributed net revenue among touches of a marketing channel.",
        channel="marketing channel")
def top_attributed_campaign(D, channel):
    a = D.attributed[D.attributed.marketing_channel == channel]
    return _top(a.groupby("campaign").net_eur.sum())


@recipe("Average number of days between touch and order for attributed valid orders of a campaign (30-day last-touch attribution).",
        campaign="campaign")
def avg_days_touch_to_order(D, campaign):
    a = D.attributed[D.attributed.campaign == campaign]
    return NA if a.empty else float(((a.order_ts - a.touch_ts) / pd.Timedelta(days=1)).mean())


@recipe("Marketing channel return percent: 100 * (attributed order margin EUR - shipping cost EUR on those orders - all eligible touch cost EUR of the channel) / all eligible touch cost EUR of the channel.",
        channel="marketing channel")
def channel_marketing_roi_pct(D, channel):
    orders = set(D.attributed[D.attributed.marketing_channel == channel].order_id)
    margin = D.items[D.items.order_id.isin(orders)].margin_eur.sum()
    ship = D.shipments[D.shipments.order_id.isin(orders)].ship_cost_eur.sum()
    spend = D.touches[D.touches.marketing_channel == channel].cost_eur.sum()
    return _pct(margin - ship - spend, spend)


# ================= support / shipments =================
@recipe("Percentage of eligible support tickets with a given priority that breached SLA.", priority="urgent/high/medium/low")
def sla_breach_pct(D, priority):
    t = D.tickets[D.tickets.priority == priority]
    return _pct(t.breach.sum(), len(t))


@recipe("Among support tickets from high-value accounts in a segment, which issue type occurred most often.", segment="account segment")
def top_issue_high_value_accounts(D, segment):
    a = D.account_ltv
    acc = set(a[a.high_value & (a.segment == segment)].index)
    return _top(D.tickets[D.tickets.account_id.isin(acc)].groupby("issue_type").size().astype(float))


@recipe("Count of accounts in an industry with at least one late shipment and at least one SLA-breaching support ticket.",
        industry="account industry")
def accounts_late_and_breach(D, industry):
    a = D.account_ltv
    acc = set(a[a.industry == industry].index)
    late = set(D.shipments[D.shipments.late].account_id)
    br = set(D.tickets[D.tickets.breach].account_id)
    return len(acc & late & br)


@recipe("Net-revenue-weighted average delay in days for late shipments handled by a carrier (valid orders only).", carrier="carrier")
def carrier_weighted_delay(D, carrier):
    s = D.shipments[D.shipments.late & (D.shipments.carrier == carrier)]
    return NA if s.net_eur.sum() == 0 else (s.net_eur * s.delay_days).sum() / s.net_eur.sum()


@recipe("Which carrier has the highest net-revenue-weighted average delay for late shipments in a customer region/country.",
        country="country or null", region="region or null")
def top_delay_carrier(D, country=None, region=None):
    s = _geo(D.shipments[D.shipments.late], country, region)
    g = s.assign(wd=s.net_eur * s.delay_days).groupby("carrier")[["wd", "net_eur"]].sum()
    g = g[g.net_eur != 0]
    return _top(g.wd / g.net_eur)


@recipe("Among support tickets opened within 14 days after a late shipment delivery for the same resolved customer in a customer region, percentage that breached SLA.",
        region="region or null", country="country or null")
def tickets_after_late_delivery_breach_pct(D, region=None, country=None):
    s = _geo(D.shipments[D.shipments.late & D.shipments.delivered], country, region)
    m = D.tickets[["ticket_id", "identity", "opened_ts", "breach"]].merge(s[["identity", "deliv_ts"]], on="identity")
    d = m.opened_ts - m.deliv_ts
    m = m[(d >= pd.Timedelta(0)) & (d <= pd.Timedelta(days=14))].drop_duplicates("ticket_id")
    return _pct(m.breach.sum(), len(m))


# ================= procurement =================
def _fam(df, family):
    return df if (not family or str(family).lower().startswith("all")) else df[df.family == family]


@recipe("Total eligible purchase commitment EUR for suppliers in a country (purchase-month FX).", supplier_country="supplier country")
def procurement_commitment(D, supplier_country):
    return D.po_lines[D.po_lines.country == supplier_country].commit_eur.sum()


def _line_acc(D, on_time=False):
    r = D.receipts
    if on_time:
        r = r[r.received_ts <= r.promised_ts]
    return r.groupby("po_line_id").accepted_units.sum()


@recipe("Units remaining outstanding across eligible purchase lines for a warehouse.", warehouse="warehouse region")
def procurement_open_units(D, warehouse):
    l = D.po_lines[D.po_lines.warehouse_region == warehouse]
    acc = l.po_line_id.map(_line_acc(D)).fillna(0)
    return int((l.ordered_units - acc).clip(lower=0).sum())


@recipe("Percentage of ordered units on due eligible purchase lines accepted on time, for suppliers in a country.", supplier_country="supplier country")
def procurement_on_time_pct(D, supplier_country):
    l = D.po_lines[(D.po_lines.country == supplier_country) & (D.po_lines.promised_ts <= REF)]
    acc = l.po_line_id.map(_line_acc(D, on_time=True)).fillna(0)
    return _pct(np.minimum(l.ordered_units, acc).sum(), l.ordered_units.sum())


@recipe("Rejected-unit percentage across eligible posted goods receipts for suppliers in a country.", supplier_country="supplier country")
def procurement_rejection_pct(D, supplier_country):
    r = D.receipts[D.receipts.country == supplier_country]
    return _pct(r.rejected_units.sum(), (r.accepted_units + r.rejected_units).sum())


@recipe("Purchase price variance EUR for a product family (or 'All families').", family="product family or 'All families'")
def procurement_price_variance(D, family=None):
    l = _fam(D.po_lines, family)
    return (l.ordered_units * (l.unit_eur - l.std_eur)).sum()


@recipe("EUR cost of accepted units in eligible posted goods receipts for a warehouse (receipt-month FX).", warehouse="warehouse region")
def procurement_received_cost(D, warehouse):
    return D.receipts[D.receipts.warehouse_region == warehouse].cost_eur.sum()


@recipe("Supplier name with the highest rejection ratio among eligible suppliers in a country.", supplier_country="supplier country")
def procurement_worst_supplier(D, supplier_country):
    g = D.receipts[D.receipts.country == supplier_country].groupby("supplier_name")[["rejected_units", "accepted_units"]].sum()
    den = g.rejected_units + g.accepted_units
    g = g[den > 0]
    return _top(g.rejected_units / (g.rejected_units + g.accepted_units))


@recipe("Count of eligible goods receipt IDs for suppliers in a country.", supplier_country="supplier country")
def procurement_receipt_count(D, supplier_country):
    return int((D.receipts.country == supplier_country).sum())


@recipe("Supplier name with greatest accepted eligible receipt units for products in a family.", family="product family or 'All families'")
def procurement_top_supplier_units(D, family=None):
    return _top(_fam(D.receipts, family).groupby("supplier_name").accepted_units.sum().astype(float))


@recipe("(Final stage) Due shortfall value EUR for suppliers in a country.", supplier_country="supplier country")
def procurement_due_shortfall_value(D, supplier_country):
    l = D.po_lines[(D.po_lines.country == supplier_country) & (D.po_lines.promised_ts <= REF)]
    acc = l.po_line_id.map(_line_acc(D)).fillna(0)
    return ((l.ordered_units - acc).clip(lower=0) * l.unit_eur).sum()


# ================= inventory =================
def _wh(df, warehouse):
    return df if (not warehouse or str(warehouse).lower().startswith("all")) else df[df.warehouse_region == warehouse]


@recipe("Total reconciled end stock units for a warehouse at the reference time.", warehouse="warehouse region")
def inventory_on_hand(D, warehouse):
    return int(_wh(D.stock, warehouse).end_stock.sum())


@recipe("Standard-cost EUR value of reconciled stock for a warehouse at the reference.", warehouse="warehouse region")
def inventory_value(D, warehouse):
    return _wh(D.stock, warehouse).value_eur.sum()


@recipe("Count of product/warehouse pairs with end stock strictly below the reorder point, for a warehouse.", warehouse="warehouse region")
def inventory_below_reorder(D, warehouse):
    return int(_wh(D.stock, warehouse).below.sum())


@recipe("Total suggested replenishment units for a warehouse at the reference.", warehouse="warehouse region")
def inventory_reorder_units(D, warehouse):
    return int(_wh(D.stock, warehouse).reorder_units.sum())


@recipe("Sum of posted adjustment movement deltas after stocktake through reference for a warehouse.", warehouse="warehouse region")
def inventory_adjustment_units(D, warehouse):
    m = _wh(D.movements, warehouse)
    return int(m[m.movement_type == "adjustment"].units_delta.sum())


@recipe("(Final stage) Inventory issue coverage days for a warehouse: end stock / (Q4 sale_issue units / 92).", warehouse="warehouse region")
def inventory_issue_coverage_days(D, warehouse):
    m = _wh(D.movements, warehouse)
    m = m[(m.movement_type == "sale_issue") & (m.event_ts >= "2024-10-01") & (m.event_ts < "2025-01-01")]
    demand = abs(m.units_delta.sum())
    return NA if demand == 0 else _wh(D.stock, warehouse).end_stock.sum() / (demand / 92)


# ================= returns =================
@recipe("Percentage of units received in eligible completed return inspections that are restockable, for a refund reason.", refund_reason="refund reason")
def returns_restockable_pct(D, refund_reason):
    i = D.inspections[D.inspections.refund_reason == refund_reason]
    return _pct(i.restockable_units.sum(), i.units_received.sum())


@recipe("Total recovery value EUR from eligible completed return inspections for an original order-time category.", category="product category")
def returns_recovered_value(D, category):
    return D.inspections[D.inspections.category == category].recovery_eur.sum()


@recipe("Count of eligible approved refunds with no eligible inspection, for a refund reason.", refund_reason="refund reason")
def returns_uninspected(D, refund_reason):
    r = D.refunds[D.refunds.refund_reason == refund_reason]
    return int((~r.refund_id.isin(set(D.inspections.refund_id))).sum())


@recipe("(Final stage) Recovery-adjusted refund rate percent for an order-time category.", category="product category")
def returns_recovery_adjusted_rate(D, category):
    ref = D.refunds_cat[D.refunds_cat.category == category].refund_eur.sum()
    rec = D.inspections[D.inspections.category == category].recovery_eur.sum()
    return _pct(ref - rec, D.items[D.items.category == category].gross_eur.sum())


# ================= support cost / contracts =================
@recipe("Total eligible worklog labor cost EUR for a support issue type.", issue_type="ticket issue type")
def support_labor_cost(D, issue_type):
    return D.worklogs[D.worklogs.issue_type == issue_type].labor_eur.sum()


@recipe("Total eligible billable worklog hours for a support channel.", support_channel="support channel")
def support_billable_hours(D, support_channel):
    w = D.worklogs[(D.worklogs.support_channel == support_channel) & D.worklogs.is_billable]
    return w.minutes_worked.sum() / 60


@recipe("Issue type with highest eligible labor cost per eligible ticket in a country.", country="customer country")
def support_costliest_issue(D, country):
    n = D.tickets[D.tickets.country == country].groupby("issue_type").size()
    c = D.worklogs[D.worklogs.country == country].groupby("issue_type").labor_eur.sum().reindex(n.index).fillna(0)
    return _top(c / n)


@recipe("Count of eligible tickets of an issue type with no eligible worklog.", issue_type="ticket issue type")
def support_unlogged_tickets(D, issue_type):
    t = D.tickets[D.tickets.issue_type == issue_type]
    return int((~t.ticket_id.isin(set(D.worklogs.ticket_id))).sum())


@recipe("Percentage of eligible tickets covered by an active contract, for an account segment.", segment="account segment")
def contracts_ticket_coverage_pct(D, segment):
    t = D.tickets_cov[D.tickets_cov.segment == segment]
    return _pct(t.covered.sum(), len(t))


@recipe("Sum of fixed service credits for covered SLA-breached eligible tickets, for an account segment.", segment="account segment")
def contracts_service_credits(D, segment):
    return D.tickets_cov[D.tickets_cov.segment == segment].credit_due.sum()


@recipe("Total service cost: eligible worklog labor EUR plus covered breached-ticket credits, for an account segment.", segment="account segment")
def support_total_service_cost(D, segment):
    return D.worklogs[D.worklogs.segment == segment].labor_eur.sum() + contracts_service_credits(D, segment)


@recipe("Contracted quarterly fee run-rate at reference for an account segment.", segment="account segment")
def contracts_quarterly_run_rate(D, segment):
    c = D.contracts[(D.contracts.segment == segment) & (D.contracts.vf <= REF) & (REF < D.contracts.vt)]
    assert not c.account_id.duplicated().any()
    return c.quarterly_fee_eur.sum()


@recipe("(Final stage) Lifetime valid-item margin for a segment minus eligible labor cost and covered breached-ticket credits.", segment="account segment")
def support_adjusted_margin(D, segment):
    return D.items[D.items.segment == segment].margin_eur.sum() - support_total_service_cost(D, segment)


@recipe("(Final stage) Q4 support overage minutes for an account segment.", segment="account segment")
def contracts_q4_overage_minutes(D, segment):
    q4 = pd.Timestamp("2024-10-01")
    c = D.contracts[(D.contracts.segment == segment) & (D.contracts.vf <= q4) & (q4 < D.contracts.vt)].set_index("account_id")
    w = D.worklogs[(D.worklogs.worked_ts >= q4) & (D.worklogs.worked_ts < REF)].groupby("account_id").minutes_worked.sum()
    return (w.reindex(c.index).fillna(0) - c.included_support_minutes).clip(lower=0).sum()


# ================= scenario cases =================
@recipe("Fulfilment decision (hard_opt) case: minimum total baseline cost EUR.", case_id="case id like OI01")
def opt_baseline_cost(D, case_id):
    from hard_opt import solve_case
    return solve_case(D, case_id, counterfactual=False) / 100


@recipe("Fulfilment decision (hard_opt) case: change in minimum total cost EUR, counterfactual minus baseline.", case_id="case id like OI01")
def opt_cost_change(D, case_id):
    from hard_opt import solve_case
    return (solve_case(D, case_id, counterfactual=True) - solve_case(D, case_id, counterfactual=False)) / 100


@recipe("Flow simulation case: baseline net contribution EUR at horizon.", case_id="case id like FLOW_I01")
def flow_baseline(D, case_id):
    from flow import simulate
    return simulate(D, case_id, what_if=False) / 100


@recipe("Flow simulation case: change in net contribution EUR, what-if minus baseline.", case_id="case id like FLOW_I01")
def flow_change(D, case_id):
    from flow import simulate
    return (simulate(D, case_id, what_if=True) - simulate(D, case_id, what_if=False)) / 100


# ================= formatting =================
def format_answer(value, answer_type):
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if answer_type == "integer":
        assert abs(value - round(value)) < 1e-9, value
        return str(int(round(value)))
    return f"{round(float(value) + 0.0, 2):.2f}".replace("-0.00", "0.00")


def run(D, name, params, answer_type="number"):
    spec = RECIPES[name]
    kwargs = {k: v for k, v in (params or {}).items() if k in spec["params"] and v not in (None, "", "null")}
    return format_answer(spec["fn"](D, **kwargs), answer_type)
