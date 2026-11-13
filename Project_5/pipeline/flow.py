"""Deterministic replay of the stateful fulfilment simulations (flow_* tables).

Implements the rules in HARD_TASK_RULES.md: known-version selection, event ordering,
sticky reservations, dispatch capacity, cancellations, returns, rebates and what-if edits.
All money is integer cents.
"""
import pandas as pd

PRIORITY = {"CANCEL": 0, "RETURN": 1, "RECEIPT": 2, "RELEASE": 3, "DISPATCH": 4}
_CACHE = {}


def _select_known(df, key, cutoff):
    """Latest known version per key: max (recorded_step, version_id lexicographic) with recorded_step <= cutoff."""
    df = df[df.recorded_step <= cutoff].copy()
    df["version_id"] = df.version_id.astype(str)
    return df.sort_values(["recorded_step", "version_id"]).drop_duplicates(key, keep="last")


def simulate(D, case_id, what_if=False, trace=None):
    key = (str(D.dir), case_id, what_if)
    if key in _CACHE and trace is None:
        return _CACHE[key]
    case = D.raw("flow_cases").set_index("case_id").loc[case_id]
    horizon, cutoff = int(case.horizon_step), int(case.knowledge_cutoff_step)
    f = lambda n: (lambda df: df[df.case_id == case_id].copy())(D.raw(n))
    orders_df, events_df, contracts_df, stock_df = f("flow_orders"), f("flow_events"), f("flow_contracts"), f("flow_stock")

    # --- known versions (select before checking status / effective step)
    ev = _select_known(events_df, "event_id", cutoff)
    ev = ev[ev.status != "void"].copy()
    if what_if:
        for pre in ("first", "second"):
            eid, action = case[f"{pre}_event_id"], case[f"{pre}_action"]
            if pd.isna(eid) or pd.isna(action) or str(eid).strip() == "":
                continue
            mask = ev.event_id == eid
            assert mask.sum() == 1, f"{case_id}: what-if event {eid} not among selected events"
            val = int(case[f"{pre}_value"])
            if action == "SHIFT":
                ev.loc[mask, "effective_step"] += val
            elif action == "SET_UNITS":
                ev.loc[mask, "units"] = val
            else:
                raise ValueError(action)
    ev = ev[ev.effective_step <= horizon]

    con = _select_known(contracts_df, "customer_id", cutoff).set_index("customer_id")

    def terms(cust):
        if cust in con.index and con.loc[cust].status == "active":
            c = con.loc[cust]
            return int(c.priority), int(c.rebate_threshold_units), int(c.rebate_per_unit_cents)
        return 99, None, 0

    # --- state
    free = {(r.product_id, r.warehouse): int(r.opening_units) for r in stock_df.itertuples()}
    orders = {}
    for o in orders_df.itertuples():
        pr, thr, reb = terms(o.customer_id)
        orders[o.order_id] = dict(o=o, prio=pr, released=False, n=int(o.units),
                                  # unit state: W waiting, R reserved, C cancelled, D dispatched, X returned
                                  state=["W"] * int(o.units), disp_no=[None] * int(o.units), late=[False] * int(o.units))
    qkey = lambda oid: (orders[oid]["prio"], int(orders[oid]["o"].deadline_step), int(orders[oid]["o"].release_step), oid)
    money = dict(revenue=0, cost=0, recovered=0, late=0, cancel_fees=0, return_fees=0)
    dispatch_counter = [0]

    def allocate():
        for oid in sorted((k for k, v in orders.items() if v["released"]), key=qkey):
            od = orders[oid]
            k = (od["o"].product_id, od["o"].warehouse)
            for i in range(od["n"]):
                if free.get(k, 0) <= 0:
                    break
                if od["state"][i] == "W":
                    od["state"][i] = "R"
                    free[k] -= 1

    actions = [(int(o.release_step), PRIORITY["RELEASE"], o.order_id, "RELEASE", o) for o in orders_df.itertuples() if int(o.release_step) <= horizon]
    actions += [(int(e.effective_step), PRIORITY[e.event_type], e.event_id, e.event_type, e) for e in ev.itertuples()]
    actions.sort(key=lambda a: a[:3])

    for step, _, aid, kind, row in actions:
        if kind == "RELEASE":
            orders[row.order_id]["released"] = True
        elif kind == "RECEIPT":
            k = (row.product_id, row.warehouse)
            free[k] = free.get(k, 0) + int(row.units)
        elif kind == "CANCEL":
            od = orders.get(row.order_id)
            if od and od["released"]:
                left = int(row.units)
                for i in range(od["n"]):
                    if left <= 0:
                        break
                    if od["state"][i] in ("W", "R"):
                        if od["state"][i] == "R":
                            k = (od["o"].product_id, od["o"].warehouse)
                            free[k] = free.get(k, 0) + 1
                        od["state"][i] = "C"
                        money["cancel_fees"] += int(od["o"].cancellation_fee_cents)
                        left -= 1
        elif kind == "DISPATCH":
            capacity = int(row.units)
            dispatch_counter[0] += 1
            for oid in sorted((k for k, v in orders.items() if v["released"] and v["o"].warehouse == row.warehouse), key=qkey):
                od = orders[oid]
                for i in range(od["n"]):
                    if capacity <= 0:
                        break
                    if od["state"][i] == "R":
                        od["state"][i] = "D"
                        od["disp_no"][i] = dispatch_counter[0]
                        capacity -= 1
                        money["revenue"] += int(od["o"].unit_price_cents)
                        money["cost"] += int(od["o"].unit_cost_cents)
                        if step > int(od["o"].deadline_step):
                            money["late"] += int(od["o"].late_penalty_cents)
        elif kind == "RETURN":
            od = orders.get(row.order_id)
            if od:
                cand = sorted((i for i in range(od["n"]) if od["state"][i] == "D"), key=lambda i: (-od["disp_no"][i], -i))
                for i in cand[:int(row.units)]:
                    od["state"][i] = "X"
                    money["revenue"] -= int(od["o"].unit_price_cents)
                    money["return_fees"] += int(od["o"].return_fee_cents)
                    if str(row.restock).strip().lower() == "true":
                        k = (od["o"].product_id, od["o"].warehouse)
                        free[k] = free.get(k, 0) + 1
                        money["recovered"] += int(od["o"].unit_cost_cents)
        allocate()
        if trace is not None:
            trace.append((step, kind, aid, dict(money)))

    unfilled = sum(sum(s in ("W", "R") for s in od["state"]) * int(od["o"].unfilled_penalty_cents) for od in orders.values() if od["released"])
    retained = {}
    for od in orders.values():
        retained[od["o"].customer_id] = retained.get(od["o"].customer_id, 0) + sum(s == "D" for s in od["state"])
    rebate = 0
    for cust, n in retained.items():
        _, thr, reb = terms(cust)
        if thr is not None and n >= thr:
            rebate += n * reb
    total = (money["revenue"] - money["cost"] + money["recovered"] - money["late"] - unfilled
             + money["cancel_fees"] - money["return_fees"] - rebate)
    if trace is not None:
        trace.append(("END", dict(money, unfilled=unfilled, rebate=rebate, total=total)))
    _CACHE[key] = total
    return total
