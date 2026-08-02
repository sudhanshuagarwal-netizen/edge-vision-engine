#!/usr/bin/env python3
"""
plot_orbital_results.py — Phase 3 plots for orbital edge simulation

Generates 4 portfolio-ready figures from orbital_sim_*.csv files:
  1. Estimated power over time (sunlight vs eclipse)
  2. Cumulative detections kept vs discarded
  3. Bandwidth used vs budget (rolling window)
  4. Skip rate / adaptive state + temperature

Usage (from tensorrt_opt/):
  python3 src/plot_orbital_results.py
  python3 src/plot_orbital_results.py --results-dir benchmarks/results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def find_latest_csvs(results_dir: Path):
    """Return dict regime -> path of the most recent orbital_sim_*.csv."""
    files = list(results_dir.glob("orbital_sim_*.csv"))
    if not files:
        raise FileNotFoundError(f"No orbital_sim_*.csv found in {results_dir}")

    by_regime = {}
    for f in files:
        # filename pattern: orbital_sim_{regime}_{timestamp}.csv
        parts = f.stem.split("_")
        if len(parts) >= 3:
            regime = parts[2]  # sunlight or eclipse
            if regime not in by_regime or f.stat().st_mtime > by_regime[regime].stat().st_mtime:
                by_regime[regime] = f
    return by_regime


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def plot_power(dfs: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"sunlight": "#F4A261", "eclipse": "#2A9D8F"}
    for regime, df in dfs.items():
        ax.plot(df["t"], df["est_power_w"],
                label=regime.upper(), color=colors.get(regime, "gray"),
                linewidth=1.6, alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated power (W)")
    ax.set_title("Estimated Power over Time — Sunlight vs Eclipse")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "plot_power_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_detections(dfs: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    colors = {"sunlight": "#F4A261", "eclipse": "#2A9D8F"}

    for ax, (regime, df) in zip(axes, dfs.items()):
        cum_kept = df["dets_kept"].cumsum()
        cum_disc = df["dets_discarded"].cumsum()
        ax.plot(df["t"], cum_kept, label="Kept", color=colors.get(regime), linewidth=1.8)
        ax.plot(df["t"], cum_disc, label="Discarded", color="gray", linewidth=1.4, linestyle="--")
        ax.set_title(f"{regime.upper()}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative detections")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cumulative Detections Kept vs Discarded", y=1.02)
    fig.tight_layout()
    path = out_dir / "plot_detections_cumulative.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bandwidth(dfs: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    budgets = {"sunlight": 2500.0, "eclipse": 800.0}
    colors = {"sunlight": "#F4A261", "eclipse": "#2A9D8F"}

    for regime, df in dfs.items():
        ax.plot(df["t"], df["bw_used_window_kb"],
                label=f"{regime.upper()} used", color=colors.get(regime),
                linewidth=1.6)
        ax.axhline(budgets.get(regime, 0), color=colors.get(regime),
                   linestyle=":", alpha=0.7, label=f"{regime.upper()} budget")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bandwidth in current 60 s window (KB)")
    ax.set_title("Rolling Bandwidth Usage vs Budget")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "plot_bandwidth.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_adaptive(dfs: dict, out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    colors = {"sunlight": "#F4A261", "eclipse": "#2A9D8F"}

    # Top: skip rate
    ax = axes[0]
    for regime, df in dfs.items():
        ax.step(df["t"], df["skip_rate"], where="post",
                label=regime.upper(), color=colors.get(regime), linewidth=1.6)
    ax.set_ylabel("Skip rate (1 = full)")
    ax.set_title("Adaptive Skip Rate & GPU Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 3.5)

    # Bottom: temperature
    ax = axes[1]
    for regime, df in dfs.items():
        ax.plot(df["t"], df["temp_gpu"],
                label=regime.upper(), color=colors.get(regime), linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU temp (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "plot_adaptive_temp.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  Saved {path}")


def print_summary_table(dfs: dict):
    print("\nQuick quantitative summary (from loaded CSVs):")
    print(f"{'Metric':<28} {'Sunlight':>12} {'Eclipse':>12}")
    print("-" * 54)
    for regime, df in dfs.items():
        pass  # just ensure order

    metrics = {}
    for regime, df in dfs.items():
        metrics[regime] = {
            "runtime_s": df["t"].iloc[-1] if len(df) else 0,
            "frames": len(df),
            "processed_pct": 100 * df["processed"].mean() if len(df) else 0,
            "mean_power": df["est_power_w"].mean(),
            "dets_kept": int(df["dets_kept"].sum()),
            "dets_seen": int(df["dets_seen"].sum()),
            "bw_total_kb": df["bw_sent_kb"].sum(),
        }

    rows = [
        ("Runtime (s)", "runtime_s", "{:.1f}"),
        ("Frames", "frames", "{:.0f}"),
        ("% processed", "processed_pct", "{:.1f}"),
        ("Mean power (W)", "mean_power", "{:.2f}"),
        ("Dets kept", "dets_kept", "{:.0f}"),
        ("Dets seen", "dets_seen", "{:.0f}"),
        ("Bandwidth sent (KB)", "bw_total_kb", "{:.0f}"),
    ]
    for label, key, fmt in rows:
        sun = metrics.get("sunlight", {}).get(key, float("nan"))
        ecl = metrics.get("eclipse", {}).get(key, float("nan"))
        print(f"{label:<28} {fmt.format(sun):>12} {fmt.format(ecl):>12}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="benchmarks/results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Looking for CSVs in {results_dir.resolve()}")

    latest = find_latest_csvs(results_dir)
    print("Using:")
    for regime, path in latest.items():
        print(f"  {regime}: {path.name}")

    dfs = {regime: load_df(path) for regime, path in latest.items()}
    if not dfs:
        raise SystemExit("No data loaded.")

    print("\nGenerating plots...")
    plot_power(dfs, results_dir)
    plot_detections(dfs, results_dir)
    plot_bandwidth(dfs, results_dir)
    plot_adaptive(dfs, results_dir)

    print_summary_table(dfs)
    print("\nDone. Plots are in", results_dir.resolve())


if __name__ == "__main__":
    main()