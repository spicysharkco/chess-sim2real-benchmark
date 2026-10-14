"""Serial execution of the ten preregistered upstream-mode C2 cells.

Start only after the common baseline has completed and passed retention. This
controller never changes hyperparameters or retries an incomplete point.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-run", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cq-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from tools.paper import lora_curve as curve
    from tools.paper.benchmark_io import finish_output, reserve_output, sha256
    from tools.paper.transfer_cohort import verify_receipt

    baseline = args.run / "results/release.dev_predictions.jsonl"
    reference = verify_receipt(args.run / "results/release.complete.json")
    if reference.get("mode") != "baseline" or not reference.get("accepted"):
        raise ValueError("matrix requires an accepted common reference baseline")
    baseline_hash = sha256(args.run / "results/release.complete.json")
    prefix = args.run / "matrix"
    reserve_output(prefix)
    specs = curve.registered_runs(False)
    plan = {
        "protocol_sha256": sha256(args.protocol),
        "controller_sha256": sha256(Path(__file__)),
        "runner_sha256": sha256(args.source / "tools/paper/lora_curve.py"),
        "runner_inputs_sha256": curve.harness_source_inputs(),
        "reference_receipt_sha256": baseline_hash,
        "recipe": curve.RECIPE,
        "runs": [spec.__dict__ for spec in specs],
        "train_head": False,
    }
    (args.run / "matrix.plan.json").write_text(json.dumps(plan, indent=2))
    logdir = args.run / "logs"
    logdir.mkdir(exist_ok=True)
    results = []
    for spec in specs:
        curve.verify_hashes(plan["runner_inputs_sha256"])
        if sha256(args.run / "results/release.complete.json") != baseline_hash:
            raise ValueError("reference receipt changed during matrix")
        if (args.run / "STOP_REQUESTED").exists():
            raise RuntimeError("author stop marker present; no next run launched")
        name = f"b{spec.budget}_{spec.arm}_s{spec.seed}"
        out = args.run / "results" / name
        command = [
            sys.executable,
            str(args.source / "tools/paper/lora_curve.py"),
            "train",
            "--run",
            str(args.data_run),
            "--cq-repo",
            str(args.cq_repo),
            "--data-root",
            str(args.data_root),
            "--checkpoint",
            str(args.checkpoint),
            "--out",
            str(out),
            "--workers",
            "4",
            "--budget",
            str(spec.budget),
            "--arm",
            spec.arm,
            "--seed",
            str(spec.seed),
            "--freeze-head",
            "--baseline-predictions",
            str(baseline),
        ]
        curve.parser().parse_args(command[2:])  # Validate before launching CUDA.
        print(json.dumps({"run_start": name}), flush=True)
        with (logdir / f"{name}.log").open("x") as log:
            subprocess.run(
                command,
                cwd=args.source,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        receipt_path = Path(f"{out}.complete.json")
        receipt = verify_receipt(receipt_path)
        if (
            receipt.get("run_spec") != spec.__dict__
            or receipt.get("recipe") != curve.RECIPE
        ):
            raise ValueError("completed point differs from preregistered recipe")
        result = {
            "run": name,
            "accepted": receipt["accepted"],
            "receipt_sha256": sha256(receipt_path),
        }
        results.append(result)
        with (args.run / "matrix.results.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps({"receipt_landed": result}), flush=True)
    if (
        sha256(args.protocol) != plan["protocol_sha256"]
        or sha256(Path(__file__)) != plan["controller_sha256"]
    ):
        raise ValueError("protocol/controller changed during matrix")
    curve.verify_hashes(plan["runner_inputs_sha256"])
    finish_output(
        prefix,
        {
            "status": "ten_fixed_cells_complete",
            "runs": results,
            "reference_receipt_sha256": baseline_hash,
            "release_ready": False,
        },
    )


if __name__ == "__main__":
    main()
