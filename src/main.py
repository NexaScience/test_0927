"""src/main.py
Orchestrates the execution of all experiment variations defined in a YAML file.
For each run it spawns src.train as a subprocess, tee-ing stdout/stderr into
both the console and per-run log files. After all runs complete, it calls
src.evaluate to aggregate and visualise the results.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

import yaml

################################################################################
# Utilities                                                                     
################################################################################

def _tee_stream(stream, log_file):
    """Mirrors a stream (stdout/stderr) into a log file and the parent stream."""
    for line in iter(stream.readline, b""):
        sys.stdout.buffer.write(line) if log_file.name.endswith("stdout.log") else sys.stderr.buffer.write(line)
        log_file.buffer.write(line)
    stream.close()


def _run_subprocess(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)

    with open(stdout_path, "wb") as fout, open(stderr_path, "wb") as ferr:
        threads = [
            threading.Thread(target=_tee_stream, args=(proc.stdout, fout), daemon=True),
            threading.Thread(target=_tee_stream, args=(proc.stderr, ferr), daemon=True),
        ]
        for t in threads:
            t.start()
        proc.wait()
        for t in threads:
            t.join()
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess {' '.join(cmd)} failed with exit code {proc.returncode}")

################################################################################
# Main                                                                          
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Run all experiments defined in a config YAML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run smoke_test.yaml")
    group.add_argument("--full-experiment", action="store_true", help="Run full_experiment.yaml")
    parser.add_argument("--results-dir", required=True, help="Where all outputs should be saved")
    args = parser.parse_args()

    cfg_path = Path("config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Sequentially run each variation
    # ------------------------------------------------------------------
    for run in cfg["runs"]:
        run_id = run["name"]
        print(f"==== Running experiment: {run_id} ====")
        run_dir = results_dir / run_id
        run_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--run-id",
            run_id,
            "--config-file",
            str(cfg_path),
            "--results-dir",
            str(results_dir),
        ]
        _run_subprocess(cmd, cwd=Path.cwd(), stdout_path=run_dir / "stdout.log", stderr_path=run_dir / "stderr.log")

    # ------------------------------------------------------------------
    # After all runs → aggregate / visualise
    # ------------------------------------------------------------------
    eval_cmd = [sys.executable, "-m", "src.evaluate", "--results-dir", str(results_dir)]
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
