# src/evaluate.py
"""Evaluates and compares results from all experiment variations.
Reads *results.json files and produces comparison figures + a JSON report.
This script is triggered by src.main once all training runs are complete.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ------------------------------------------------------------------------- #
# Utility
# ------------------------------------------------------------------------- #

def load_results(results_dir: Path) -> List[Dict]:
    results = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        res_file = run_dir / "results.json"
        if res_file.exists():
            with open(res_file) as f:
                results.append(json.load(f))
    return results


def aggregate_metrics(all_results: List[Dict]) -> pd.DataFrame:
    rows = []
    for res in all_results:
        row = {"run_id": res["run_id"]}
        metrics = res.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- #
# Figure generation helpers
# ------------------------------------------------------------------------- #

def barplot_metric(df: pd.DataFrame, metric: str, out_dir: Path):
    plt.figure(figsize=(6, 4))
    sns.barplot(x="run_id", y=metric, data=df)
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    # Annotate each bar with value
    for i, v in enumerate(df[metric]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.title(f"{metric} comparison")
    plt.tight_layout()
    fname = f"{metric}.pdf".replace(" ", "_")
    plt.savefig(out_dir / fname, bbox_inches="tight")
    plt.close()
    return fname


# ------------------------------------------------------------------------- #
# Main evaluation pipeline
# ------------------------------------------------------------------------- #

def evaluate(results_dir: Path):
    results_dir = Path(results_dir)
    out_img_dir = results_dir / "images"
    out_img_dir.mkdir(exist_ok=True, parents=True)

    all_results = load_results(results_dir)
    if len(all_results) == 0:
        raise RuntimeError(f"No results.json files found in {results_dir}")

    df = aggregate_metrics(all_results)

    # Identify numeric metrics (excluding run_id)
    metric_columns = [c for c in df.columns if c != "run_id"]
    generated_figures = []
    for metric in metric_columns:
        fname = barplot_metric(df, metric, out_img_dir)
        generated_figures.append(fname)

    # ------------------------------------------------------------------ #
    #  JSON summary printed to STDOUT                                   #
    # ------------------------------------------------------------------ #
    summary = {"best_by_metric": {}, "figures": generated_figures}
    for metric in metric_columns:
        if metric.startswith("loss"):
            best_run = df.loc[df[metric].idxmin(), "run_id"]
        else:
            best_run = df.loc[df[metric].idxmax(), "run_id"]
        summary["best_by_metric"][metric] = best_run

    print(json.dumps(summary, indent=2))


# ------------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Aggregate & compare experiment results")
    p.add_argument("--results-dir", type=str, required=True, help="Root directory holding experiment outputs")
    return p.parse_args()


def main():
    args = parse_args()
    evaluate(Path(args.results_dir))


if __name__ == "__main__":
    main()
