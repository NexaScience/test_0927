"""src/evaluate.py
Aggregates the results of all experiment variations and produces comparison
figures under <results-dir>/images/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

################################################################################
# Plot helpers                                                                  
################################################################################

def _make_bar(ax, names: List[str], values: List[float], title: str, ylabel: str):
    palette = sns.color_palette("Set2", len(names))
    bars = ax.bar(names, values, color=palette)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

################################################################################
# Main                                                                          
################################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all experimental runs and plot comparisons")
    parser.add_argument("--results-dir", required=True, help="Path where individual run folders live")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    image_dir = results_dir / "images"
    image_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load result JSONs
    # ------------------------------------------------------------------
    run_results: Dict[str, Dict] = {}
    for run_path in results_dir.iterdir():
        if run_path.is_dir() and (run_path / "results.json").exists():
            with open(run_path / "results.json", "r") as f:
                run_results[run_path.name] = json.load(f)

    if not run_results:
        print("{}")
        return

    names = list(run_results.keys())
    fid_values = [run_results[n]["metrics"].get("fid", None) for n in names]
    clip_values = [run_results[n]["metrics"].get("clip_score", None) for n in names]
    lat_values = [run_results[n]["metrics"].get("inference_time", None) for n in names]

    # ------------------------------------------------------------------
    # 2. Plot FID and CLIPScore
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if any(v is not None for v in fid_values):
        _make_bar(axes[0], names, [v if v is not None else 0 for v in fid_values], "FID (lower=better)", "FID")
    if any(v is not None for v in clip_values):
        _make_bar(axes[1], names, [v if v is not None else 0 for v in clip_values], "CLIPScore (higher=better)", "CLIPScore")
    plt.tight_layout()
    plt.savefig(image_dir / "quality_metrics.pdf", bbox_inches="tight")

    # ------------------------------------------------------------------
    # 3. Plot inference latency
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    _make_bar(ax, names, lat_values, "Inference latency", "seconds / image")
    plt.tight_layout()
    plt.savefig(image_dir / "inference_latency.pdf", bbox_inches="tight")

    # ------------------------------------------------------------------
    # 4. Print structured comparison summary
    # ------------------------------------------------------------------
    summary = {
        "best_fid_run": min([(v, n) for n, v in zip(names, fid_values) if v is not None], default=(None, None))[1],
        "best_clip_run": max([(v, n) for n, v in zip(names, clip_values) if v is not None], default=(None, None))[1],
        "fastest_run": min([(v, n) for n, v in zip(names, lat_values) if v is not None], default=(None, None))[1],
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
