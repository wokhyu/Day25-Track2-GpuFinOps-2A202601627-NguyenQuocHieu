"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30
RESERVED_RATE_COL = {"1yr": "reserved_1yr_hr", "3yr": "reserved_3yr_hr"}


def price_job(job, cat, policy: str = "new") -> dict:
    """Cost one workload under a purchasing policy.

    'base' is the lab's original policy: tier from duty cycle alone, a flat 5%/hr
    spot interruption rate for every GPU, and a 3-year reservation for anything
    steady. 'new' (Extension 1) prices spot with the GPU type's own interruption
    rate and sizes the commitment term to how much of the month the job actually
    stands up.
    """
    gtype = job["gpu_type"]
    ngpu = int(num(job["num_gpus"]))
    hpd = num(job["hours_per_day"])
    job_days = num(job["days"])
    interruptible = bool(int(num(job["interruptible"])))
    c = cat[gtype]
    gpu_hours = hpd * DAYS * ngpu
    od = num(c["on_demand_hr"])
    on_demand_cost = gpu_hours * od

    if policy == "base":
        tier = pricing.recommend_tier(hpd, interruptible)
        term = "3yr"
        rate = pricing.DEFAULT_INTERRUPT_RATE
    else:
        term = pricing.recommend_commit_term(job_days)
        reserved_hr = num(c[RESERVED_RATE_COL[term]]) if term in RESERVED_RATE_COL else None
        tier = pricing.recommend_tier(
            hpd, interruptible, gpu_type=gtype, job_days=job_days,
            spot_hr=num(c["spot_hr"]), on_demand_hr=od, reserved_hr=reserved_hr,
        )
        rate = pricing.interrupt_rate_for(gtype)
        if tier == "reserved" and term == "none":
            tier = "on_demand"  # nothing worth locking in for a short-lived job

    if tier == "spot":
        sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od, interrupt_rate=rate)
        opt_cost = sim["spot_cost"]
    elif tier == "reserved":
        opt_cost = gpu_hours * num(c[RESERVED_RATE_COL.get(term, "reserved_3yr_hr")])
    else:
        opt_cost = on_demand_cost

    return {"job_id": job["job_id"], "gpu_type": gtype, "tier": tier, "term": term,
            "interrupt_rate": rate, "gpu_hours": gpu_hours,
            "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
            "on_demand_raw": on_demand_cost, "optimized_raw": opt_cost}


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    recs = [price_job(j, cat, "new") for j in jobs]
    base_recs = [price_job(j, cat, "base") for j in jobs]
    on_demand_monthly = sum(r["on_demand_raw"] for r in recs)
    optimized_monthly = sum(r["optimized_raw"] for r in recs)
    base_optimized_monthly = sum(r["optimized_raw"] for r in base_recs)
    base_savings_pct = ((on_demand_monthly - base_optimized_monthly) / on_demand_monthly * 100
                        if on_demand_monthly else 0.0)

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'term':6}{'spot-risk':>10}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            risk = f"{r['interrupt_rate']:.0%}/hr" if r["tier"] == "spot" else "-"
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}{r['term']:6}{risk:>10}"
                  f"${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        # --- Extension 1: old policy vs new policy ---
        print("\n-- Policy comparison (Extension 1) --")
        print(f"  base policy (flat 5%/hr spot risk, always 3yr): ${base_optimized_monthly:,.0f}"
              f"  ({base_savings_pct:.1f}% saved)")
        print(f"  new  policy (per-GPU spot risk + sized term)  : ${optimized_monthly:,.0f}"
              f"  ({savings_pct:.1f}% saved)")
        print(f"  delta: ${base_optimized_monthly - optimized_monthly:+,.0f}/month"
              f"  ({savings_pct - base_savings_pct:+.1f} pp)")
        changed = [(b, n) for b, n in zip(base_recs, recs)
                   if (b["tier"], b["term"]) != (n["tier"], n["term"])]
        for b, n in changed:
            print(f"    {b['job_id']:18} {b['tier']}/{b['term']} -> {n['tier']}/{n['term']}"
                  f"  ${b['optimized']:,} -> ${n['optimized']:,}")

        print("\n-- Tier matrix: GPU type x duty cycle x interruptible --")
        print(f"  {'gpu':8}{'risk':>7}  " + "".join(f"{h:>13}" for h in ("3h/int", "8h/int", "20h/int", "20h/fixed", "24h/fixed")))
        for g in ("H100", "A100", "A10G", "L4"):
            c = cat[g]
            cells = []
            for hpd, inter in ((3, True), (8, True), (20, True), (20, False), (24, False)):
                t = pricing.recommend_tier(hpd, inter, gpu_type=g, spot_hr=num(c["spot_hr"]),
                                           on_demand_hr=num(c["on_demand_hr"]),
                                           reserved_hr=num(c["reserved_3yr_hr"]))
                cells.append(f"{t:>13}")
            print(f"  {g:8}{pricing.interrupt_rate_for(g):>6.0%}  " + "".join(cells))

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "base_optimized_monthly": round(base_optimized_monthly),
            "base_savings_pct": round(base_savings_pct, 1),
            "policy_delta_monthly": round(base_optimized_monthly - optimized_monthly, 2)}


if __name__ == "__main__":
    run()
