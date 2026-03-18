"""Run a full experiment: train across curriculum phases then evaluate."""
import argparse
import subprocess
import sys
from pathlib import Path


PHASE_CONFIGS = {
    1: "configs/train/phase1.yaml",
    2: "configs/train/phase2.yaml",
    3: "configs/train/phase3.yaml",
    4: "configs/train/phase4.yaml",
}


def run_phase(phase: int, config_path: str, extra_args: list[str] | None = None):
    """Run a single training phase."""
    cmd = [sys.executable, "scripts/train.py", "--config", config_path]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"Phase {phase}: {config_path}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Training phases to run (default: 1 2 3)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--eval_after",
        action="store_true",
        help="Run evaluation after training",
    )
    parser.add_argument("--predictions_path", type=str, default=None)
    parser.add_argument("--gold_path", type=str, default=None)
    args = parser.parse_args()

    for phase in sorted(args.phases):
        if phase not in PHASE_CONFIGS:
            print(f"Warning: no config for phase {phase}, skipping")
            continue
        extra = ["--device", args.device]
        run_phase(phase, PHASE_CONFIGS[phase], extra)

    if args.eval_after and args.predictions_path and args.gold_path:
        print(f"\n{'='*60}")
        print("Running evaluation")
        print(f"{'='*60}")
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--predictions_path",
                args.predictions_path,
                "--gold_path",
                args.gold_path,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
