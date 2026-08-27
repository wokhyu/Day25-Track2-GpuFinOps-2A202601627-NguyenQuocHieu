"""Tests for the "Your Turn" extensions (student-written; the 15 lab tests are untouched)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m6_carbon_scheduling


# --- Extension 1: recommend_tier + commitment term ---------------------------

def test_interrupt_rate_is_per_gpu_type():
    assert pricing.interrupt_rate_for("H100") < pricing.interrupt_rate_for("A10G")
    assert pricing.interrupt_rate_for("nonexistent") == pricing.DEFAULT_INTERRUPT_RATE


def test_recommend_tier_keeps_documented_defaults():
    # extension must not break the original contract when no prices are supplied
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_recommend_tier_rejects_spot_when_rework_eats_the_discount():
    # H100: spot $1.50 vs a $1.40 3-year reservation -> steady tier wins at high duty
    assert pricing.recommend_tier(20, True, gpu_type="H100", spot_hr=1.5,
                                  on_demand_hr=2.5, reserved_hr=1.4) == "reserved"
    # same duty, but a deep spot discount still wins
    assert pricing.recommend_tier(20, True, gpu_type="H100", spot_hr=0.5,
                                  on_demand_hr=2.5, reserved_hr=1.4) == "spot"


def test_commit_term_scales_with_persistence():
    assert pricing.recommend_commit_term(30) == "3yr"
    assert pricing.recommend_commit_term(20) == "1yr"
    assert pricing.recommend_commit_term(5) == "none"


def test_new_policy_is_not_worse_than_the_base_policy():
    r3 = m3_purchasing.run(verbose=False)
    assert r3["savings_pct"] >= r3["base_savings_pct"]
    assert {r["tier"] for r in r3["recommendations"]} <= {"spot", "reserved", "on_demand"}


# --- Extension 2: right-sizing by MBU ----------------------------------------

def test_dollars_per_gb_vram_ranks_differently_from_dollars_per_hour():
    from missions._common import catalog_by_type, num
    cat = catalog_by_type()
    per_gb = m1_efficiency_audit.dollars_per_gb_vram(cat)
    cheapest_hr = min(cat, key=lambda g: num(cat[g]["on_demand_hr"]))
    cheapest_gb = min(per_gb, key=per_gb.get)
    assert cheapest_hr != cheapest_gb          # L4 is cheapest per hour, not per GB


def test_rightsize_candidates_are_feasible():
    from missions._common import catalog_by_type, num
    cat = catalog_by_type()
    r1 = m1_efficiency_audit.run(verbose=False)
    assert r1["rightsize_monthly"] > 0
    for c in r1["rightsize"]:
        tgt = cat[c["to"]]
        assert num(tgt["on_demand_hr"]) < c["from_hr"]                  # actually cheaper
        assert num(tgt["hbm_gb"]) >= c["peak_mem_gb"]                   # fits the workload
        assert c["mbu"] < m1_efficiency_audit.MBU_TARGET                # only slack GPUs


# --- Extension 3: cache_is_worth_it ------------------------------------------

def test_cache_break_even_is_price_independent():
    small = pricing.cache_break_even_reads(pricing.cache_write_premium(0.20), base_price_per_m=0.20)
    large = pricing.cache_break_even_reads(pricing.cache_write_premium(3.00), base_price_per_m=3.00)
    assert abs(small - large) < 1e-9
    assert abs(large - 0.25 / 0.90) < 1e-9


def test_cache_is_worth_it_needs_reuse():
    prem = pricing.cache_write_premium(3.00)
    assert pricing.cache_is_worth_it(5.0, prem, base_price_per_m=3.00) is True
    assert pricing.cache_is_worth_it(0.0, prem, base_price_per_m=3.00) is False


def test_longer_ttl_raises_measured_reuse():
    from missions._common import load_csv
    rows = load_csv("token_usage.csv")
    short = m2_inference_levers.cache_reuse(rows, 300)
    long = m2_inference_levers.cache_reuse(rows, 3600)
    assert long["avg_reads_per_write"] > short["avg_reads_per_write"]
    assert short["writes"] + short["reads"] == long["writes"] + long["reads"]


# --- Extension 4: reasoning budget -------------------------------------------

def test_reasoning_energy_share_dwarfs_its_traffic_share():
    r2 = m2_inference_levers.run(verbose=False)
    rr = r2["reasoning"]
    assert rr["traffic_pct"] < 15 and rr["wh_pct"] > 80     # few requests, most of the energy
    assert rr["wh_pct"] > rr["cost_pct"]                    # energy scales worse than price
    assert rr["cap_wh_saved_daily"] > 0 and rr["cap_cost_saved_daily"] > 0


# --- Extension 5: carbon-aware scheduling ------------------------------------

def test_cleanest_region_is_not_the_cheapest_one():
    r6 = m6_carbon_scheduling.run(verbose=False)
    assert r6["cleanest"] != r6["cheapest"]
    # the clean grid is also cheaper than home here; the premium only shows up
    # against the *cheapest* grid, which is dirtier
    assert r6["power_saved_usd"] > 0
    assert r6["clean_premium_usd"] > 0 and r6["usd_per_tonne_co2e"] > 0
    assert r6["carbon_saved_pct"] > 80


def test_only_interruptible_load_is_shifted():
    r6 = m6_carbon_scheduling.run(verbose=False)
    assert 0 < r6["shiftable_kwh"] < r6["fleet_kwh"]
