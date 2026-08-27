"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


SPOT_INTERRUPT_RATE = {
    # Per-hour chance a spot instance is reclaimed, by GPU type. Scarce
    # datacenter parts are reclaimed less often than commodity inference cards:
    # nobody outbids you for an H100 fleet on a whim, but A10G/L4 capacity is
    # thin and churns constantly.
    "B200": 0.02,
    "H200": 0.03,
    "H100": 0.03,
    "MI300X": 0.04,
    "A100": 0.05,
    "A10G": 0.12,
    "L4": 0.15,
}
DEFAULT_INTERRUPT_RATE = 0.05


def interrupt_rate_for(gpu_type: str | None) -> float:
    """Per-hour spot reclaim probability for a GPU type."""
    return SPOT_INTERRUPT_RATE.get(gpu_type or "", DEFAULT_INTERRUPT_RATE)


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
    spot_hr: float | None = None,
    on_demand_hr: float | None = None,
    reserved_hr: float | None = None,
) -> str:
    """Pick a purchasing tier from duty cycle, interruptibility and spot economics.

    Base policy (unchanged, still the fallback when no prices are supplied):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    Extension: 'spot' is no longer assumed to win just because a job can be
    checkpointed. When spot/on-demand rates are supplied the choice is priced
    out using the GPU type's own interruption rate, so a flaky commodity card
    with a thin spot discount can lose to the steady tier on rework alone.
    `job_days` is not used for the tier itself — commitment *length* is a
    separate decision, see `recommend_commit_term`.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    steady_tier = "reserved" if duty >= be else "on_demand"

    if interruptible and hours_per_day < 24:
        if spot_hr is None or on_demand_hr is None or hours_per_day <= 0:
            return "spot"  # no prices to reason with: keep the documented default
        steady_hr = on_demand_hr
        if steady_tier == "reserved" and reserved_hr is not None:
            steady_hr = reserved_hr
        sim = spot_checkpoint_cost(
            hours_per_day, spot_hr, on_demand_hr,
            interrupt_rate=interrupt_rate_for(gpu_type),
        )
        effective_spot_hr = sim["spot_cost"] / hours_per_day
        if effective_spot_hr < steady_hr:
            return "spot"
        # rework ate the discount — fall through to the steady tier
    return steady_tier


def recommend_commit_term(job_days: float, days_in_month: float = 30.0) -> str:
    """How long to commit for, given how much of the month the job actually stands up.

    A reservation is a bet that the *same shape* of demand persists for the whole
    term. Duty cycle answers "is it busy while it runs"; persistence answers "does
    it keep running at all". Only a job that is effectively always-on earns a 3-year
    lock — a half-month job that gets a 3-year commitment is buying idle capacity
    for 35 months.
    """
    persistence = max(0.0, job_days) / max(1e-9, days_in_month)
    if persistence >= 0.90:
        return "3yr"
    if persistence >= 0.50:
        return "1yr"
    return "none"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


# --- Extension 3: is the prompt cache actually paying for itself? -------------

CACHE_WRITE_MULTIPLIER = 1.25  # Anthropic 5-min cache write ~1.25x the base input rate


def cache_write_premium(base_price_per_m: float, write_multiplier: float = CACHE_WRITE_MULTIPLIER) -> float:
    """Extra $/1M-token paid to *populate* the cache, above the normal input rate."""
    return max(0.0, base_price_per_m * (write_multiplier - 1.0))


def cache_break_even_reads(
    write_cost_per_m: float,
    read_discount: float = 0.10,
    base_price_per_m: float = 1.0,
) -> float:
    """How many cache reads a written prefix needs before caching turns a profit.

    Writing costs a one-off premium; every subsequent read saves
    (1 - read_discount) x the base input rate. Break-even is the ratio.
    """
    savings_per_read = base_price_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return float("inf")
    return write_cost_per_m / savings_per_read


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    base_price_per_m: float = 1.0,
) -> bool:
    """True when the reads a prefix gets outrun the premium paid to cache it.

    Caching is not free: a written prefix that is read back once or zero times
    costs *more* than not caching at all. The lever is reuse, not the discount.
    """
    return avg_cache_reads > cache_break_even_reads(write_cost_per_m, read_discount, base_price_per_m)
