"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import datetime
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

CACHE_TTL_S = 300          # Anthropic default 5-minute cache lifetime
CACHE_TTL_LONG_S = 3600    # the 1-hour extended-TTL option
REASONING_CAP_FRAC = 0.03  # Extension 4: cap reasoning at 3% of traffic


def cache_reuse(rows, ttl_s: int = CACHE_TTL_S) -> dict:
    """Measure how often a cached prefix is actually read back before it expires.

    A prefix is approximated by (team, project, route_tier) — the things that
    share a system prompt. Within one TTL window the first request pays the write
    premium and the rest are reads, so reads-per-write is the number that decides
    whether the cache earns its keep.
    """
    groups = defaultdict(list)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            key = (r["team"], r["project"], r["route_tier"])
            groups[key].append(datetime.datetime.fromisoformat(r["ts"]))
    writes = reads = 0
    for stamps in groups.values():
        windows = {int(t.timestamp() // ttl_s) for t in stamps}
        writes += len(windows)
        reads += len(stamps) - len(windows)
    return {
        "prefixes": len(groups), "writes": writes, "reads": reads,
        "avg_reads_per_write": reads / writes if writes else 0.0,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")

    # --- Extension 3: only claim the cache discount if the cache pays for itself ---
    reuse = cache_reuse(rows, CACHE_TTL_S)
    reuse_long = cache_reuse(rows, CACHE_TTL_LONG_S)
    break_even = {
        tier: pricing.cache_break_even_reads(
            pricing.cache_write_premium(MODEL_PRICES[tier][0]),
            base_price_per_m=MODEL_PRICES[tier][0],
        )
        for tier in MODEL_PRICES
    }
    cache_pays = pricing.cache_is_worth_it(
        reuse["avg_reads_per_write"],
        pricing.cache_write_premium(MODEL_PRICES["large"][0]),
        base_price_per_m=MODEL_PRICES["large"][0],
    )

    base_cost = opt_cost = 0.0
    total_tokens = 0
    # --- Extension 4: reasoning vs non-reasoning, in dollars AND watt-hours ---
    split = {0: {"n": 0, "cost": 0.0, "tokens": 0, "wh": 0.0},
             1: {"n": 0, "cost": 0.0, "tokens": 0, "wh": 0.0}}
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        this_cost = pricing.request_cost(inp, out, pin, pout,
                                         cached_in=cached if cache_pays else 0,
                                         batch=is_batch)
        opt_cost += this_cost

        rflag = int(num(r["is_reasoning"]))
        b = split[rflag]
        b["n"] += 1
        b["cost"] += this_cost
        b["tokens"] += inp + out
        b["wh"] += sustainability.wh_per_query(inp + out, is_reasoning=bool(rflag))

    # what-if: demote the most expensive reasoning traffic above the cap to plain calls
    reasoning_rows = [r for r in rows if int(num(r["is_reasoning"]))]
    plain_rows = [r for r in rows if not int(num(r["is_reasoning"]))]
    median_plain_out = (sorted(int(num(r["output_tokens"])) for r in plain_rows)[len(plain_rows) // 2]
                        if plain_rows else 0)
    keep = int(len(rows) * REASONING_CAP_FRAC)
    demoted = sorted(reasoning_rows, key=lambda r: int(num(r["output_tokens"])),
                     reverse=True)[: max(0, len(reasoning_rows) - keep)]
    cap_cost_saved = cap_wh_saved = 0.0
    for r in demoted:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cached = int(num(r["cached_input_tokens"])) if cache_pays else 0
        is_batch = bool(int(num(r["is_batch"])))
        new_out = min(out, median_plain_out)
        cap_cost_saved += (pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
                           - pricing.request_cost(inp, new_out, pin, pout, cached_in=cached, batch=is_batch))
        cap_wh_saved += (sustainability.wh_per_query(inp + out, is_reasoning=True)
                         - sustainability.wh_per_query(inp + new_out, is_reasoning=False))

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- Cache economics (Extension 3) --")
        for tier, be in break_even.items():
            prem = pricing.cache_write_premium(MODEL_PRICES[tier][0])
            print(f"  {tier:6} input ${MODEL_PRICES[tier][0]:.2f}/1M  write premium ${prem:.3f}/1M"
                  f"  break-even = {be:.2f} reads")
        print(f"  measured @ {CACHE_TTL_S}s TTL : {reuse['prefixes']} prefixes, {reuse['writes']} writes,"
              f" {reuse['reads']} reads  ->  {reuse['avg_reads_per_write']:.2f} reads/write")
        print(f"  measured @ {CACHE_TTL_LONG_S}s TTL: {reuse_long['writes']} writes,"
              f" {reuse_long['reads']} reads  ->  {reuse_long['avg_reads_per_write']:.2f} reads/write")
        print(f"  cache_is_worth_it -> {cache_pays}"
              f"  (margin {reuse['avg_reads_per_write'] / break_even['large']:.1f}x over break-even)")

        print("\n-- Reasoning budget (Extension 4) --")
        tot_c = split[0]["cost"] + split[1]["cost"]
        tot_wh = split[0]["wh"] + split[1]["wh"]
        for flag, label in ((1, "reasoning"), (0, "plain")):
            b = split[flag]
            print(f"  {label:10} {b['n']:>5} req ({b['n']/len(rows)*100:>4.1f}% traffic)"
                  f"  ${b['cost']:>6.2f} ({b['cost']/tot_c*100:>4.1f}% cost)"
                  f"  {b['wh']:>9,.0f} Wh ({b['wh']/tot_wh*100:>4.1f}% energy)")
        print(f"  cap reasoning to {REASONING_CAP_FRAC:.0%} of traffic:"
              f" saves ${cap_cost_saved:.2f}/day and {cap_wh_saved:,.0f} Wh/day"
              f" ({cap_wh_saved/tot_wh*100:.1f}% of energy)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "requests": len(rows),
        "cache_reuse": reuse, "cache_reuse_long": reuse_long,
        "cache_break_even": {k: round(v, 3) for k, v in break_even.items()},
        "cache_pays": cache_pays,
        "reasoning": {
            "n": split[1]["n"], "traffic_pct": round(split[1]["n"] / len(rows) * 100, 1),
            "cost": round(split[1]["cost"], 2),
            "cost_pct": round(split[1]["cost"] / (split[0]["cost"] + split[1]["cost"]) * 100, 1),
            "wh": round(split[1]["wh"], 1),
            "wh_pct": round(split[1]["wh"] / (split[0]["wh"] + split[1]["wh"]) * 100, 1),
            "plain_wh": round(split[0]["wh"], 1),
            "plain_cost": round(split[0]["cost"], 2),
            "cap_frac": REASONING_CAP_FRAC,
            "cap_cost_saved_daily": round(cap_cost_saved, 2),
            "cap_wh_saved_daily": round(cap_wh_saved, 1),
        },
    }


if __name__ == "__main__":
    run()
