"""src/evaluate.py
Aggregates results from multiple experiment variations, produces comparative
figures and prints summary statistics in structured JSON.
"""
from __future__ import annotations
import argparse
import json
import os
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate & compare experiment variations.")
    parser.add_argument("--results-dir", type=str, required=True, help="Root directory containing variation sub-directories.")
    return parser.parse_args()


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    runs = []
    for run_id in sorted(os.listdir(results_dir)):
        res_file = os.path.join(results_dir, run_id, "results.json")
        if os.path.isfile(res_file):
            with open(res_file, "r") as f:
                runs.append(json.load(f))
    return runs


def aggregate_metrics(runs: List[Dict[str, Any]]):
    summary = {}
    for run in runs:
        summary[run["run_id"]] = {
            "best_score": run["best_score"],
            "time_to_threshold": run["time_to_threshold"],
        }
    return summary


def plot_best_scores(runs: List[Dict[str, Any]], out_dir: str):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    run_ids = [r["run_id"] for r in runs]
    best_scores = [r["best_score"] for r in runs]
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=run_ids, y=best_scores)
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Best Compressed Score")
    ax.set_title("Best Score Comparison across Variations")
    for idx, val in enumerate(best_scores):
        ax.text(idx, val + 0.01, f"{val:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "best_score_comparison.pdf"), bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    runs = load_results(args.results_dir)
    if not runs:
        print("No result files found – nothing to evaluate.")
        return

    summary = aggregate_metrics(runs)
    # --------- Figures ---------
    plot_best_scores(runs, args.results_dir)

    print(json.dumps({"comparison": summary}, indent=2))


if __name__ == "__main__":
    main()
