
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from agents.zoning_baselines import (
    run_zoning,
    random_policy,
    rule_based_policy
)

from agents.transit_baselines import (
    run_transit,
    random_policy as random_transit,
    greedy_policy
)


OUTPUT_DIR = Path("data/visualizations")

COLORS = [
    "#E8C36A",  # Residential
    "#9868C4",  # Commercial
    "#969696",  # Industrial
    "#4CAF50",  # Green
    "#DC8055",  # Mixed
    "#4285C5"   # Water
]

LABELS = [
    "Residential",
    "Commercial",
    "Industrial",
    "Green",
    "Mixed",
    "Water"
]


def visualize_zoning():

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    policies = [
        ("Random Policy", random_policy),
        ("Rule-based Policy", rule_based_policy)
    ]

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(
        np.arange(-0.5, 6.5), cmap.N
    )

    for ax, (name, policy) in zip(axes, policies):

        land, water, reward, steps = run_zoning(
            policy, seed=42
        )

        grid = np.where(water, 5, land)

        ax.imshow(grid, cmap=cmap, norm=norm)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    legend = [
        Patch(color=color, label=label)
        for color, label in zip(COLORS, LABELS)
    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=6
    )

    fig.suptitle("Synthetic City - Zoning Comparison")

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    fig.savefig(
        OUTPUT_DIR / "zoning_comparison.png",
        dpi=200
    )


def visualize_transit():

    random_result = run_transit(random_transit)
    greedy_result = run_transit(greedy_policy)

    results = [random_result, greedy_result]

    coverage = [
        100 * r["covered_demand"] / r["total_demand"]
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["Random", "Greedy"],
        coverage,
        color=["#648DB9", "#39966D"]
    )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Demand Coverage (%)")
    ax.set_title("Synthetic Transit Comparison")

    ax.bar_label(
        bars,
        labels=[f"{v:.1f}%" for v in coverage],
        padding=4
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "transit_comparison.png",
        dpi=200
    )


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visualize_zoning()
    visualize_transit()

    print("\nPASS: Visualizations generated")
    print("Saved to:", OUTPUT_DIR)

    plt.show()


if __name__ == "__main__":
    main()
