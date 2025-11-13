"""
evaluate.py – Aggregate & compare results of the run variations.
Reads all sub-directories in --results-dir that contain results.json, compiles
comparison tables & figures and writes them to stdout + images/.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FIG_TOPIC_FINAL_LOSS = "final_loss"

################################################################################
# -----------------------------  utilities  -----------------------------------#
################################################################################

def collect_results(results_dir: Path) -> List[Dict]:
    records = []
    for run_dir in results_dir.iterdir():
        file = run_dir / "results.json"
        if file.exists():
            with open(file, "r", encoding="utf-8") as fp:
                records.append(json.load(fp))
    return records

################################################################################
# --------------------------  figure helpers  ---------------------------------#
################################################################################

def plot_final_loss(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df, x="run_id", y="final_val_loss", hue="algorithm")
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Final Validation Loss")
    ax.set_title("Final Validation Loss Across Experiments")

    # annotate each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fname = f"{FIG_TOPIC_FINAL_LOSS}.pdf"
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(images_dir / fname, bbox_inches="tight")
    plt.close()
    return fname

################################################################################
# ------------------------------   main   -------------------------------------#
################################################################################

def main(results_dir: Path):
    records = collect_results(results_dir)
    if not records:
        raise RuntimeError(f"No results.json found under {results_dir}")

    df = pd.DataFrame(records)
    # ---------------------------------------------------------------- figures
    fig_files = []
    fig_files.append(plot_final_loss(df, results_dir))

    # --------------------------------------------------------- stdout outputs
    comparison = {
        "num_runs": len(records),
        "best_final_val_loss": df["final_val_loss"].min(),
        "worst_final_val_loss": df["final_val_loss"].max(),
        "figure_files": fig_files,
    }

    print("\n===== Cross-Run Comparison Summary =====")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()
    main(Path(args.results_dir))