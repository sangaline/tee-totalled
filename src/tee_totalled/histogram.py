"""Histogram generation with seaborn for game results."""

import io
import logging
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def generate_histogram(scores: list[int]) -> tuple[bytes, dict[str, float]]:
    """
    Generate a histogram image and statistics for the given scores.

    Returns:
        Tuple of (PNG image bytes, statistics dict).
    """
    if not scores:
        raise ValueError("Cannot generate histogram with no scores")

    # Calculate statistics.
    stats = {
        "mean": mean(scores),
        "median": median(scores),
        "std": stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "count": len(scores),
    }

    # Set up the plot style.
    sns.set_theme(style="darkgrid", palette="deep")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Fixed bins: 1-10, 11-20, ..., 91-100.
    bin_edges = np.arange(0.5, 101.5, 10)
    sns.histplot(
        scores,
        bins=bin_edges,
        kde=len(scores) >= 3,
        color="#5DA5DA",
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    # Add vertical lines for mean and median.
    ax.axvline(
        stats["mean"],
        color="#F15854",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {stats["mean"]:.1f}',
    )
    ax.axvline(
        stats["median"],
        color="#60BD68",
        linestyle="-.",
        linewidth=2,
        label=f'Median: {stats["median"]:.1f}',
    )

    # Customize the plot.
    ax.set_xlabel("Offensiveness Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distribution of Offensiveness Scores",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Fixed 0-100 range with labeled bin centers.
    ax.set_xlim(0, 101)
    ax.set_xticks(np.arange(5.5, 100, 10))
    ax.set_xticklabels([f"{i}-{i+9}" for i in range(1, 100, 10)], fontsize=9)

    # Add legend with statistics.
    legend_text = (
        f"n = {stats['count']}\n"
        f"Mean = {stats['mean']:.1f}\n"
        f"Median = {stats['median']:.1f}\n"
        f"Std Dev = {stats['std']:.1f}"
    )
    ax.text(
        0.98,
        0.98,
        legend_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    ax.legend(loc="upper left")

    # Add subtle branding.
    fig.text(
        0.99,
        0.01,
        "TeeTotalledBot - Trust in TEE",
        fontsize=8,
        color="gray",
        ha="right",
        va="bottom",
        alpha=0.7,
    )

    plt.tight_layout()

    # Save to bytes.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    image_bytes = buf.getvalue()

    plt.close(fig)

    logger.debug(f"Generated histogram for {len(scores)} scores")

    return image_bytes, stats


def format_stats_message(stats: dict[str, float]) -> str:
    """Format statistics as a message string."""
    return (
        f"📊 *Game Statistics*\n\n"
        f"Participants: {int(stats['count'])}\n"
        f"Mean Score: {stats['mean']:.1f}\n"
        f"Median Score: {stats['median']:.1f}\n"
        f"Std Deviation: {stats['std']:.1f}\n"
        f"Range: {int(stats['min'])} - {int(stats['max'])}"
    )
