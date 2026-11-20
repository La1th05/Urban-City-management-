"""5-seed baseline evaluation and figure generation for Pricing and Maintenance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.maintenance_baselines import run_maintenance_baselines
from agents.pricing_baselines import run_pricing_baselines


def main() -> None:
    seeds = (0, 1, 2, 3, 4)
    summary = {}
    summary.update(run_pricing_baselines(seeds))
    summary.update(run_maintenance_baselines(seeds))

    print("=== Stage 3 & 4 Baselines (synthetic, 5 seeds) ===")
    for name, (mean_r, std_r) in summary.items():
        print(f"{name:22s} | label=synthetic | return={mean_r:+.4f} +/- {std_r:.4f}")

    assert summary["Pricing: GridSearch"][0] >= summary["Pricing: Fixed"][0]
    assert summary["Maint: WorstFirst"][0] >= summary["Maint: DoNothing"][0]

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "pricing_maintenance_baselines.png"

    labels = list(summary.keys())
    means = [summary[k][0] for k in labels]
    stds = [summary[k][1] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=["#4c78a8", "#2b5c8f", "#9ecae9", "#f58518", "#e45756"])
    ax.set_ylabel("Episode Return (mean +/- std, 5 seeds)")
    ax.set_title("Stage 3 & 4 Baselines [synthetic]")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=15, ha="right")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {fig_path} (do not commit outputs/).")


if __name__ == "__main__":
    main()
