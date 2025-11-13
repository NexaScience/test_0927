"""
main.py – Experiment orchestrator.
Reads smoke_test.yaml or full_experiment.yaml, spawns src/train.py sequentially
for each run variation, captures logs, and finally launches src/evaluate.py.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import yaml

################################################################################
# -----------------------------  log helpers  ---------------------------------#
################################################################################

def tee_subprocess(cmd: List[str], stdout_path: Path, stderr_path: Path):
    """Run *cmd* while tee-ing stdout / stderr to the given files + parent console."""
    with open(stdout_path, "w", encoding="utf-8") as out_fp, open(stderr_path, "w", encoding="utf-8") as err_fp:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            if stdout_line:
                sys.stdout.write(stdout_line)
                out_fp.write(stdout_line)
            if stderr_line:
                sys.stderr.write(stderr_line)
                err_fp.write(stderr_line)
            if stdout_line == "" and stderr_line == "" and process.poll() is not None:
                break
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

################################################################################
# -----------------------------  orchestrator  --------------------------------#
################################################################################

def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def create_temp_run_config(run_cfg: Dict) -> Path:
    """Write *run_cfg* to a NamedTemporaryFile and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
    json.dump(run_cfg, tmp)
    tmp.flush()
    return Path(tmp.name)


def main(args):
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path("config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml")
    exp_cfg = load_yaml(cfg_path)

    runs = exp_cfg.get("experiments", [])
    if not runs:
        print(f"No experiments defined in {cfg_path}")
        sys.exit(1)

    for run in runs:
        run_id = run["run_id"]
        print(f"\n=== Launching run: {run_id} ===")
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        temp_cfg_path = create_temp_run_config(run)
        cmd = [
            sys.executable, "-m", "src.train",
            "--run-config", str(temp_cfg_path),
            "--results-dir", str(results_dir),
        ]
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        tee_subprocess(cmd, stdout_path, stderr_path)

    # ------------------------------------------------- post-hoc evaluation
    print("\n===== All runs finished, starting evaluation =====")
    eval_cmd = [
        sys.executable, "-m", "src.evaluate",
        "--results-dir", str(results_dir),
    ]
    tee_subprocess(
        eval_cmd,
        results_dir / "evaluation_stdout.log",
        results_dir / "evaluation_stderr.log",
    )


################################################################################
# --------------------------------  CLI  --------------------------------------#
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experimental pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run smoke_test.yaml")
    group.add_argument("--full-experiment", action="store_true", help="Run full_experiment.yaml")
    parser.add_argument("--results-dir", required=True, help="Directory where results are stored.")

    args_parsed = parser.parse_args()
    main(args_parsed)