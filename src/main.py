"""src/main.py
Experiment orchestrator: reads a config-file listing all variations, launches
train.py sequentially, collects logs, and finally invokes evaluate.py.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
import shutil
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment orchestrator.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run variations defined in smoke_test.yaml")
    group.add_argument("--full-experiment", action="store_true", help="Run variations defined in full_experiment.yaml")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory where all outputs will be saved.")
    return parser.parse_args()


def tee_subprocess(cmd, stdout_path: Path, stderr_path: Path):
    """Runs *cmd* and simultaneously writes stdout/stderr to file and console."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
        # Non-blocking read loop
        while True:
            out_line = proc.stdout.readline()
            err_line = proc.stderr.readline()
            if out_line:
                sys.stdout.write(out_line)
                f_out.write(out_line)
            if err_line:
                sys.stderr.write(err_line)
                f_err.write(err_line)
            if not out_line and not err_line and proc.poll() is not None:
                break
    return proc.returncode


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent  # project root
    config_path = (
        root / "config" / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiments = config["experiments"]
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Copy config for provenance
    shutil.copy(config_path, results_dir / config_path.name)

    for exp in experiments:
        run_id = exp["run_id"]
        run_cfg_path = results_dir / f"{run_id}_config.yaml"
        with open(run_cfg_path, "w") as f:
            yaml.safe_dump(exp, f)

        stdout_path = results_dir / run_id / "stdout.log"
        stderr_path = results_dir / run_id / "stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(run_cfg_path),
            "--results-dir",
            str(results_dir),
            "--run-id",
            run_id,
        ]
        print(f"===== Launching {run_id} =====")
        sys.stdout.flush()
        rc = tee_subprocess(cmd, stdout_path, stderr_path)
        if rc != 0:
            print(f"Experiment {run_id} failed with return-code {rc}")
            sys.exit(rc)
        print(f"===== Completed {run_id} =====\n")

    # -------------------- Post-hoc evaluation -----------------------
    eval_cmd = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--results-dir",
        str(results_dir),
    ]
    subprocess.check_call(eval_cmd)


if __name__ == "__main__":
    main()
