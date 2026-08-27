"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 analysis: str | None = None) -> str:
    """Return a markdown cost-optimization report."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    total_lever = sum(levers.values()) or 1.0
    lines[-2] = "| Lever | Savings (USD) | % of baseline | % of savings |"
    lines[-1] = "|---|---|---|---|"
    for name, amount in sorted(levers.items(), key=lambda kv: -kv[1]):
        share_base = amount / baseline_usd * 100 if baseline_usd else 0.0
        lines.append(f"| {name} | ${amount:,.0f} | {share_base:.1f}% | {amount / total_lever * 100:.1f}% |")
    lines.append(f"| **Total** | **${sum(levers.values()):,.0f}** | "
                 f"**{sum(levers.values()) / baseline_usd * 100 if baseline_usd else 0:.1f}%** | **100%** |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cleanest region: {sustainability.get('best_region', 'n/a')}"
            f" ({sustainability.get('carbon_saved_pct', 0):.0f}% less carbon on shiftable load,"
            f" {sustainability.get('carbon_saved_kg', 0):,.0f} kgCO2e/month)",
            f"- Cheapest power: {sustainability.get('cheapest_region', 'n/a')}"
            f"  |  best $/carbon balance: {sustainability.get('balanced_region', 'n/a')}",
            f"- Shiftable (interruptible) load: {sustainability.get('shiftable_kwh', 0):,.0f} kWh/month;"
            f" moving it to the clean region also saves ${sustainability.get('power_saved_usd', 0):,.0f}/month of electricity",
            f"- Cost of choosing clean over cheapest: ${sustainability.get('usd_per_tonne_co2e', 0):,.0f}/tonne CO2e avoided",
        ]
    if analysis:
        lines += ["", analysis.rstrip()]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, **kwargs) -> str:
    """Write a savings waterfall PNG: baseline, one step down per lever, optimized.

    Pass `baseline_usd` so the bridge starts at the real spend; without it the
    chart falls back to bridging the savings alone. No-op if matplotlib absent.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    ordered = sorted(levers.items(), key=lambda kv: -kv[1])
    baseline = kwargs.get("baseline_usd")
    if baseline is None:
        baseline = sum(v for _, v in ordered)
    labels = ["Baseline"] + [n for n, _ in ordered] + ["Optimized"]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    running = baseline
    ax.bar(0, baseline, color="#8c2f39", width=0.62)
    ax.text(0, baseline, f"${baseline:,.0f}", ha="center", va="bottom", fontsize=9)
    for i, (name, amount) in enumerate(ordered, start=1):
        ax.bar(i, -amount, bottom=running, color="#2e548a", width=0.62)
        ax.plot([i - 0.31 - 0.38, i - 0.31], [running, running], color="#999", lw=0.9, ls="--")
        ax.text(i, running - amount, f"-${amount:,.0f}", ha="center", va="top", fontsize=9)
        running -= amount
    ax.bar(len(ordered) + 1, running, color="#2f7a4f", width=0.62)
    ax.text(len(ordered) + 1, running, f"${running:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("USD / month")
    ax.set_title(f"NimbusAI monthly GPU spend: \${baseline:,.0f} -> \${running:,.0f} "
                 f"({(baseline - running) / baseline * 100:.0f}% saved)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
