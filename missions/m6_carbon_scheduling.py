"""M6 — Carbon-aware scheduling (Extension 5): move the shiftable work, not all of it.

Interruptible jobs are already checkpointed and already tolerate being moved, so
they are the only part of the fleet that can chase a clean grid without a
migration project. This mission prices that move in three currencies at once:
grams of CO2, dollars of electricity, and milliseconds of user-visible latency.

Run: python missions/m6_carbon_scheduling.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

DAYS = 30
HOME_REGION = "us-east-1"

# One-way network latency from a US-east user population, ms. Illustrative, but
# the ordering is physics: Norway is ~6,000 km further away than Virginia.
REGION_LATENCY_MS = {
    "us-east-1": 5,
    "us-east-wa": 60,
    "us-west-2": 70,
    "europe-north1": 110,
    "europe-central2": 120,
}


def job_energy_kwh(job, cat) -> float:
    """Monthly energy for a workload: GPU-hours x board power (+ datacenter overhead)."""
    gpu_hours = num(job["hours_per_day"]) * DAYS * int(num(job["num_gpus"]))
    watts = num(cat[job["gpu_type"]]["watts"])
    pue = 1.15  # modern datacenter power-usage effectiveness
    return gpu_hours * watts * pue / 1000.0


def region_table(total_kwh: float) -> list:
    """Cost and carbon of the same energy in every region, plus a balanced score."""
    rows = []
    for region in sustainability.REGION_CARBON:
        gco2_kwh = sustainability.REGION_CARBON[region]
        usd_kwh = sustainability.REGION_PRICE_KWH[region]
        rows.append({
            "region": region,
            "usd_per_kwh": usd_kwh,
            "gco2_per_kwh": gco2_kwh,
            "power_usd": total_kwh * usd_kwh,
            "carbon_kg": total_kwh * gco2_kwh / 1000.0,
            "latency_ms": REGION_LATENCY_MS.get(region, 0),
        })
    # balanced score: normalise both axes to their best value, lower is better
    min_usd = min(r["usd_per_kwh"] for r in rows)
    min_co2 = min(r["gco2_per_kwh"] for r in rows)
    for r in rows:
        r["score"] = (r["usd_per_kwh"] / min_usd) + (r["gco2_per_kwh"] / min_co2)
    return rows


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    shiftable = [j for j in jobs if int(num(j["interruptible"]))]
    fixed = [j for j in jobs if not int(num(j["interruptible"]))]
    shift_kwh = sum(job_energy_kwh(j, cat) for j in shiftable)
    fixed_kwh = sum(job_energy_kwh(j, cat) for j in fixed)

    table = region_table(shift_kwh)
    by_region = {r["region"]: r for r in table}
    cleanest = min(table, key=lambda r: r["gco2_per_kwh"])
    cheapest = min(table, key=lambda r: r["usd_per_kwh"])
    balanced = min(table, key=lambda r: r["score"])
    home = by_region[HOME_REGION]

    carbon_saved_kg = home["carbon_kg"] - cleanest["carbon_kg"]
    power_saved_usd = home["power_usd"] - cleanest["power_usd"]
    # The real tension is not home-vs-clean (the clean grid happens to be cheaper
    # than home too) but clean-vs-cheapest: the cheapest grid is 3x dirtier.
    premium_usd = cleanest["power_usd"] - cheapest["power_usd"]
    premium_carbon_kg = cheapest["carbon_kg"] - cleanest["carbon_kg"]
    usd_per_tonne = premium_usd / max(premium_carbon_kg / 1000.0, 1e-9)

    if verbose:
        print("== M6 Carbon-aware Scheduling ==")
        print(f"shiftable (interruptible) jobs: {len(shiftable)}/{len(jobs)}"
              f"  ->  {shift_kwh:,.0f} kWh/month of the fleet's {shift_kwh + fixed_kwh:,.0f} kWh"
              f" ({shift_kwh / (shift_kwh + fixed_kwh) * 100:.0f}%)")

        print(f"\n-- Per-job carbon, {HOME_REGION} vs {cleanest['region']} --")
        print(f"  {'job':18}{'gpu':7}{'kWh/mo':>9}{'kgCO2 home':>12}{'kgCO2 clean':>13}{'saved':>9}")
        for j in shiftable:
            kwh = job_energy_kwh(j, cat)
            home_kg = kwh * home["gco2_per_kwh"] / 1000.0
            clean_kg = kwh * cleanest["gco2_per_kwh"] / 1000.0
            print(f"  {j['job_id']:18}{j['gpu_type']:7}{kwh:>9,.0f}{home_kg:>12,.1f}"
                  f"{clean_kg:>13,.1f}{home_kg - clean_kg:>9,.1f}")

        print(f"\n-- All 5 regions, for the {shift_kwh:,.0f} kWh of shiftable load --")
        print(f"  {'region':16}{'$/kWh':>8}{'gCO2/kWh':>10}{'power $/mo':>12}{'tCO2e/mo':>10}{'latency':>9}{'score':>8}")
        for r in sorted(table, key=lambda x: x["score"]):
            print(f"  {r['region']:16}{r['usd_per_kwh']:>8.3f}{r['gco2_per_kwh']:>10.0f}"
                  f"{r['power_usd']:>12,.0f}{r['carbon_kg'] / 1000.0:>10.2f}"
                  f"{r['latency_ms']:>7} ms{r['score']:>8.2f}")

        print(f"\n-- Verdict --")
        print(f"  cheapest power : {cheapest['region']} (${cheapest['usd_per_kwh']:.3f}/kWh,"
              f" ${cheapest['power_usd']:,.0f}/mo)")
        print(f"  cleanest grid  : {cleanest['region']} ({cleanest['gco2_per_kwh']:.0f} gCO2/kWh,"
              f" {cleanest['carbon_kg'] / 1000.0:.2f} tCO2e/mo)")
        print(f"  best balance   : {balanced['region']} (score {balanced['score']:.2f})")
        print(f"  moving all shiftable load {HOME_REGION} -> {cleanest['region']}:"
              f" -{carbon_saved_kg:,.0f} kgCO2e/mo"
              f" ({carbon_saved_kg / home['carbon_kg'] * 100:.0f}% less carbon)"
              f" and ${power_saved_usd:,.0f}/mo LESS electricity - both axes win")
        print(f"  real trade-off : {cleanest['region']} vs the cheapest grid {cheapest['region']}"
              f" costs ${premium_usd:,.0f}/mo more to avoid {premium_carbon_kg:,.0f} kgCO2e"
              f"  ->  ${usd_per_tonne:,.0f}/tonne CO2e")
        print(f"  latency cost   : +{cleanest['latency_ms'] - home['latency_ms']} ms one-way —"
              f" irrelevant for these jobs, they are batch training with no user waiting.")

    return {
        "shiftable_jobs": len(shiftable), "shiftable_kwh": round(shift_kwh, 1),
        "fleet_kwh": round(shift_kwh + fixed_kwh, 1),
        "home_region": HOME_REGION, "cleanest": cleanest["region"],
        "cheapest": cheapest["region"], "balanced": balanced["region"],
        "carbon_saved_kg": round(carbon_saved_kg, 1),
        "carbon_saved_pct": round(carbon_saved_kg / home["carbon_kg"] * 100, 1),
        "power_saved_usd": round(power_saved_usd, 2),
        "clean_premium_usd": round(premium_usd, 2),
        "clean_premium_carbon_kg": round(premium_carbon_kg, 1),
        "usd_per_tonne_co2e": round(usd_per_tonne, 2),
        "regions": table,
    }


if __name__ == "__main__":
    run()
