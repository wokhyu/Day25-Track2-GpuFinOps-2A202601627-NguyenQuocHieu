"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30
MBU_TARGET = 0.60   # a healthy memory-bound serving GPU (deck: H100-80GB batch-1)
MFU_TARGET = 0.40   # a healthy training GPU
VRAM_HEADROOM = 1.20  # keep 20% VRAM slack over the observed peak


def dollars_per_gb_vram(cat: dict) -> dict:
    """$/hr per GB of HBM for every GPU in the catalog — the memory-side unit price."""
    return {
        gtype: num(row["on_demand_hr"]) / max(1e-9, num(row["hbm_gb"]))
        for gtype, row in cat.items()
    }


def rightsize_by_mbu(summary: list, cat: dict) -> list:
    """Propose a cheaper GPU for every card whose real work fits a smaller part.

    A GPU is a right-sizing candidate when it is missing BOTH targets (MFU and
    MBU) — it is not compute-saturated and not bandwidth-saturated, so the part
    is simply larger than the workload. The replacement must still be able to
    deliver the *observed* throughput at target efficiency and hold the observed
    peak VRAM plus headroom; among the feasible parts we take the cheapest.
    """
    price_per_gb = dollars_per_gb_vram(cat)
    out = []
    for s in summary:
        cur = s["gpu_type"]
        if s["mbu"] >= MBU_TARGET or s["mfu"] >= MFU_TARGET:
            continue  # saturated on at least one axis: leave it alone
        need_bw = s["achieved_bw_tbs"] / MBU_TARGET      # BW the replacement must peak at
        need_tflops = s["achieved_tflops"] / MFU_TARGET  # FLOPs the replacement must peak at
        need_vram = s["peak_mem_gb"] * VRAM_HEADROOM
        cur_hr = num(cat[cur]["on_demand_hr"])
        feasible = [
            g for g, row in cat.items()
            if num(row["peak_bw_tbs"]) >= need_bw
            and num(row["peak_tflops_fp16"]) >= need_tflops
            and num(row["hbm_gb"]) >= need_vram
            and num(row["on_demand_hr"]) < cur_hr
        ]
        if not feasible:
            continue
        best = min(feasible, key=lambda g: num(cat[g]["on_demand_hr"]))
        new_hr = num(cat[best]["on_demand_hr"])
        out.append({
            "gpu_id": s["gpu_id"], "from": cur, "to": best,
            "from_hr": cur_hr, "to_hr": new_hr,
            "from_per_gb": price_per_gb[cur], "to_per_gb": price_per_gb[best],
            "peak_mem_gb": s["peak_mem_gb"], "mbu": s["mbu"], "mfu": s["mfu"],
            "monthly_savings": (cur_hr - new_hr) * 24 * DAYS,
            "savings_pct": (cur_hr - new_hr) / cur_hr * 100,
        })
    return out


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0,
                               "tflops": [], "bw": [], "peak_mem": 0.0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        a["tflops"].append(num(r["achieved_tflops"]))
        a["bw"].append(num(r["achieved_bw_tbs"]))
        a["peak_mem"] = max(a["peak_mem"], num(r["mem_used_gb"]))
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "achieved_tflops": round(sum(a["tflops"]) / len(a["tflops"]), 1),
            "achieved_bw_tbs": round(sum(a["bw"]) / len(a["bw"]), 3),
            "peak_mem_gb": round(a["peak_mem"], 1),
        })

    lies = metrics.flag_util_lies(summary)
    rightsize = rightsize_by_mbu(summary, cat)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        # --- Extension 2: right-sizing by MBU, priced per GB of VRAM ---
        print("\n-- $/GB-VRAM (memory-side unit price) --")
        pg = dollars_per_gb_vram(cat)
        for g in sorted(pg, key=lambda x: pg[x]):
            print(f"  {g:8} ${num(cat[g]['on_demand_hr']):>5.2f}/hr  {num(cat[g]['hbm_gb']):>4.0f} GB"
                  f"  ${pg[g]:.4f}/GB-hr  bw {num(cat[g]['peak_bw_tbs']):.2f} TB/s")
        print("\n-- Right-size candidates (MBU < 0.60 and MFU < 0.40) --")
        if not rightsize:
            print("  none")
        for c in rightsize:
            print(f"  {c['gpu_id']:14}{c['from']:>5} -> {c['to']:<5}"
                  f" peak VRAM {c['peak_mem_gb']:>5.1f} GB  MBU {c['mbu']:.3f}"
                  f"  ${c['from_hr']:.2f}->${c['to_hr']:.2f}/hr"
                  f"  save ${c['monthly_savings']:,.0f}/mo ({c['savings_pct']:.0f}%)")
        print(f"  TOTAL right-size savings: ${sum(c['monthly_savings'] for c in rightsize):,.0f}/month")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "rightsize": rightsize,
            "rightsize_monthly": round(sum(c["monthly_savings"] for c in rightsize), 2)}


if __name__ == "__main__":
    run()
