# src/main.py
"""Main orchestrator script.
Reads a YAML configuration file (either smoke_test.yaml or full_experiment.yaml)
and sequentially executes every experiment variation by spawning src.train as a
sub-process.  After all runs finish it calls src.evaluate to aggregate results.
Structured logging to stdout/stderr + per-run log files is implemented via a
tee-like mechanism.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import yaml

# The directory in which this file resides
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
CONFIG_DIR = ROOT / "config"

TRAIN_MODULE = "src.train"
EVAL_MODULE = "src.evaluate"


# ------------------------------------------------------------------------- #
# Process helpers                                                           #
# ------------------------------------------------------------------------- #

def tee_stream(stream, *files):
    """Yields lines from stream while simultaneously writing to file handles."""
    for line in iter(stream.readline, b""):
        for f in files:
            f.write(line.decode())
        yield line.decode()


def run_subprocess(cmd: List[str], stdout_path: Path, stderr_path: Path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w") as so, open(stderr_path, "w") as se:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Stream STDOUT
        for line in tee_stream(proc.stdout, so, sys.stdout):
            pass
        # Stream STDERR
        for line in tee_stream(proc.stderr, se, sys.stderr):
            pass
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Sub-process {' '.join(cmd)} exited with code {proc.returncode}")


# ------------------------------------------------------------------------- #
# Orchestrator                                                              #
# ------------------------------------------------------------------------- #

def execute_runs(experiments: List[Dict], results_dir: Path):
    for exp in experiments:
        run_id = exp.get("run_id")
        if run_id is None:
            raise ValueError("Every experiment variation must have a 'run_id' field")
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Persist run-specific config to JSON (so train.py can read it)
        cfg_path = run_dir / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(exp, f, indent=2)

        # Build command
        cmd = [
            sys.executable,
            "-m",
            TRAIN_MODULE,
            "--config",
            str(cfg_path),
            "--results-dir",
            str(results_dir),
            "--run-id",
            run_id,
        ]
        print(f"\n=== Launching run '{run_id}' ===")
        run_subprocess(cmd, stdout_path=run_dir / "stdout.log", stderr_path=run_dir / "stderr.log")
        print(f"=== Run '{run_id}' completed ===\n")

    # After all runs: evaluate
    eval_cmd = [sys.executable, "-m", EVAL_MODULE, "--results-dir", str(results_dir)]
    run_subprocess(eval_cmd, stdout_path=results_dir / "evaluate_stdout.log", stderr_path=results_dir / "evaluate_stderr.log")


# ------------------------------------------------------------------------- #
# CLI                                                                       #
# ------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Auto-ASE experiment orchestrator")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke test defined in config/smoke_test.yaml")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment defined in config/full_experiment.yaml")
    p.add_argument("--results-dir", type=str, required=True, help="Directory where outputs will be saved")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_file = CONFIG_DIR / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")

    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    experiments = cfg.get("experiments")
    if not experiments:
        raise ValueError("Configuration file must contain 'experiments' list")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    execute_runs(experiments, results_dir)


if __name__ == "__main__":
    main()
