"""Exact solver for the coupled fulfilment decision cases (hard_opt_* tables).

Solved exactly by branch-and-bound over used-supplier subsets; all money is integer cents.
`milp_check` is an independent SciPy/HiGHS formulation used to cross-validate every case.
"""

_CACHE = {}


def load_case(D, case_id, counterfactual):
    t = {n: D.raw(f"hard_opt_{n}") for n in ["jobs", "options", "stock", "dispatch", "suppliers", "changes"]}
    t = {n: df[df.case_id == case_id].copy() for n, df in t.items()}
    assert len(t["jobs"]), f"unknown case {case_id}"
    if counterfactual:
        for ch in t["changes"].itertuples():
            if ch.table_name == "stock":
                df, mask = t["stock"], (t["stock"].warehouse_region == ch.warehouse_region) & (t["stock"].product_id == ch.product_id)
            elif ch.table_name == "dispatch":
                df, mask = t["dispatch"], (t["dispatch"].warehouse_region == ch.warehouse_region) & (t["dispatch"].dispatch_day == int(ch.dispatch_day))
            elif ch.table_name == "suppliers":
                df, mask = t["suppliers"], t["suppliers"].supplier_id == ch.supplier_id
            else:
                raise ValueError(ch.table_name)
            assert mask.sum() == 1, f"change {ch.change_id} matches {mask.sum()} rows"
            df.loc[mask, ch.field] = int(float(ch.new_value))
    return t


def feasible_options(t):
    """Options passing the per-option rules, with their full cost (variable + lateness)."""
    jobs = t["jobs"].set_index("job_id")
    sup = t["suppliers"].set_index("supplier_id")
    stock = {(r.warehouse_region, r.product_id): int(r.available_units) for r in t["stock"].itertuples()}
    cap = {(r.warehouse_region, int(r.dispatch_day)): int(r.capacity_units) for r in t["dispatch"].itertuples()}
    out = []
    for o in t["options"].itertuples():
        j = jobs.loc[o.job_id]
        if o.supplier_id not in sup.index or int(sup.loc[o.supplier_id].active) != 1:
            continue
        s = sup.loc[o.supplier_id]
        day, q = int(o.dispatch_day), int(j.quantity)
        arrival = day + int(s.lead_days)
        if day < j.ready_day or arrival > j.last_day:
            continue
        if stock.get((o.warehouse_region, j.product_id), 0) < q or cap.get((o.warehouse_region, day), 0) < q:
            continue
        cost = int(o.variable_cost_cents) + q * int(j.late_penalty_cents_per_unit_day) * max(0, arrival - int(j.due_day))
        out.append(dict(job=o.job_id, wh=o.warehouse_region, prod=j.product_id, sup=o.supplier_id, day=day, q=q, cost=cost))
    return out, jobs, sup, stock, cap


def solve_case(D, case_id, counterfactual=False):
    """Minimum total cost in integer cents (exact branch-and-bound, cached)."""
    key = (str(D.dir), case_id, counterfactual)
    if key not in _CACHE:
        _CACHE[key] = branch_and_bound(D, case_id, counterfactual)
    return _CACHE[key]


def milp_check(D, case_id, counterfactual=False):
    """Independent formulation solved with SciPy/HiGHS MILP, used to cross-validate branch_and_bound."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    t = load_case(D, case_id, counterfactual)
    opts, jobs, sup, stock, cap = feasible_options(t)
    job_ids, sup_ids = list(jobs.index), list(sup.index)
    nx, nj, ns = len(opts), len(job_ids), len(sup_ids)
    n = nx + nj + ns  # x options, unfilled flags, used-supplier flags
    c = np.array([o["cost"] for o in opts] + [int(jobs.loc[j].unfilled_penalty_cents) for j in job_ids]
                 + [int(sup.loc[s].activation_cost_cents) for s in sup_ids], dtype=float)
    A, lo, hi = [], [], []

    def add(coefs, l, h):
        row = np.zeros(n)
        for i, v in coefs:
            row[i] += v
        A.append(row); lo.append(l); hi.append(h)

    for ji, j in enumerate(job_ids):
        add([(i, 1) for i, o in enumerate(opts) if o["job"] == j] + [(nx + ji, 1)], 1, 1)
    for k, units in stock.items():
        add([(i, o["q"]) for i, o in enumerate(opts) if (o["wh"], o["prod"]) == k], -np.inf, units)
    for k, units in cap.items():
        add([(i, o["q"]) for i, o in enumerate(opts) if (o["wh"], o["day"]) == k], -np.inf, units)
    for si, s in enumerate(sup_ids):
        mine = [(i, o["q"]) for i, o in enumerate(opts) if o["sup"] == s]
        add(mine + [(nx + nj + si, -int(sup.loc[s].minimum_units))], 0, np.inf)
        add(mine + [(nx + nj + si, -int(sup.loc[s].maximum_units))], -np.inf, 0)
        for i, _ in mine:
            add([(i, 1), (nx + nj + si, -1)], -np.inf, 0)
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"mip_rel_gap": 0})
    assert res.status == 0, res.message
    return int(round(res.fun))


def branch_and_bound(D, case_id, counterfactual=False):
    """Exact search: enumerate used-supplier subsets, DFS over jobs with pruning."""
    from itertools import combinations
    t = load_case(D, case_id, counterfactual)
    opts, jobs, sup, stock0, cap0 = feasible_options(t)
    job_ids = list(jobs.index)
    best = sum(int(j.unfilled_penalty_cents) for _, j in jobs.iterrows())
    sups = sorted({o["sup"] for o in opts})
    for r in range(1, len(sups) + 1):
        for subset in combinations(sups, r):
            act = sum(int(sup.loc[s].activation_cost_cents) for s in subset)
            if act >= best:
                continue
            by_job = {j: sorted([o for o in opts if o["job"] == j and o["sup"] in subset], key=lambda o: o["cost"]) for j in job_ids}
            pen = {j: int(jobs.loc[j].unfilled_penalty_cents) for j in job_ids}
            # only options cheaper than the penalty can help unless needed for supplier minimum -> keep all
            order = sorted(job_ids, key=lambda j: -pen[j])
            lb_rest = [0] * (len(order) + 1)
            for i in range(len(order) - 1, -1, -1):
                j = order[i]
                lb_rest[i] = lb_rest[i + 1] + min([pen[j]] + [o["cost"] for o in by_job[j]])
            stock, cap = dict(stock0), dict(cap0)
            load = {s: 0 for s in subset}
            mins = {s: int(sup.loc[s].minimum_units) for s in subset}
            maxs = {s: int(sup.loc[s].maximum_units) for s in subset}

            def dfs(i, cost):
                nonlocal best
                if cost + lb_rest[i] >= best:
                    return
                if i == len(order):
                    if all(load[s] >= mins[s] and load[s] > 0 for s in subset):
                        best = cost
                    return
                j = order[i]
                for o in by_job[j]:
                    ks, kc = (o["wh"], o["prod"]), (o["wh"], o["day"])
                    if stock[ks] >= o["q"] and cap[kc] >= o["q"] and load[o["sup"]] + o["q"] <= maxs[o["sup"]]:
                        stock[ks] -= o["q"]; cap[kc] -= o["q"]; load[o["sup"]] += o["q"]
                        dfs(i + 1, cost + o["cost"])
                        stock[ks] += o["q"]; cap[kc] += o["q"]; load[o["sup"]] -= o["q"]
                dfs(i + 1, cost + pen[j])

            dfs(0, act)
    return best
