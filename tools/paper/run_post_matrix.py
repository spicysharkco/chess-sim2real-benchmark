"""Wait without CUDA, then run the fixed report -> timing -> residual chain."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paper.benchmark_io import finish_output, reserve_output, sha256
from tools.paper.lora_report import RUNS
from tools.paper.transfer_cohort import verify_receipt

STAGES = ("lora_report", "timing", "residual")
CONTRACTS = {
    "lora_report": (
        "lora_report.py",
        "lora_curve_report_complete",
        "LORA_REPORT_PROTOCOL_2026-09-10.md",
    ),
    "timing": (
        "timing_harness.py",
        "c4_timing_complete",
        "TIMING_PROTOCOL_2026-09-10.md",
    ),
    "residual": (
        "residual_shift.py",
        "c3_residual_complete",
        "RESIDUAL_SHIFT_PROTOCOL_2026-09-10.md",
    ),
}


def matrix_ready(root):
    """Require both the process exit and the complete ten-point receipt."""
    if (root / "STOP_REQUESTED").exists():
        raise RuntimeError("author stop requested")
    exit_file = root / "matrix_exit.txt"
    if not exit_file.exists():
        return False
    if exit_file.read_text().strip() != "0":
        raise RuntimeError("matrix process failed; downstream work is blocked")
    receipt_path = root / "matrix.complete.json"
    receipt = verify_receipt(receipt_path)
    if (
        receipt.get("status") != "ten_fixed_cells_complete"
        or [p["run"] for p in receipt.get("runs", [])] != RUNS
    ):
        raise ValueError("matrix is not the complete registered ten-point run")
    for point in receipt["runs"]:
        path = root / "results" / f"{point['run']}.complete.json"
        complete = verify_receipt(path)
        if (
            sha256(path) != point["receipt_sha256"]
            or complete["accepted"] != point["accepted"]
        ):
            raise ValueError("matrix point receipt changed")
    return True


def validate_plan(plan):
    if [stage.get("name") for stage in plan.get("stages", [])] != list(STAGES):
        raise ValueError("stage order must be report, timing, residual")
    if not plan.get("inputs_sha256"):
        raise ValueError("plan must bind scripts, protocols and configuration hashes")
    source = Path(plan["cwd"]).resolve()
    for stage in plan["stages"]:
        if (
            not isinstance(stage.get("command"), list)
            or not stage["command"]
            or not all(isinstance(x, str) and x for x in stage["command"])
            or not stage.get("receipt")
            or not stage.get("expected_status")
        ):
            raise ValueError("invalid stage command/receipt")
        script, status, protocol = CONTRACTS[stage["name"]]
        expected_script = source / "tools/paper" / script
        expected_protocol = source / "docs" / protocol
        command = stage["command"]
        if (
            len(command) < 4
            or Path(command[1]).resolve() != expected_script
            or stage["expected_status"] != status
            or str(expected_script) not in plan["inputs_sha256"]
            or str(expected_protocol) not in plan["inputs_sha256"]
        ):
            raise ValueError(
                "stage entrypoint/status/protocol does not match its registered meaning"
            )
        if stage["name"] == "timing" and command[2] != "timing-all":
            raise ValueError("timing stage must run the complete timing-all command")
        if command.count("--out") != 1:
            raise ValueError("each stage must name exactly one output prefix")
        out = Path(command[command.index("--out") + 1]).resolve()
        if Path(stage["receipt"]).resolve() != Path(f"{out}.complete.json"):
            raise ValueError("stage receipt must match its command output prefix")
        if stage["name"] != "timing" and (
            command.count("--protocol") != 1
            or Path(command[command.index("--protocol") + 1]).resolve()
            != expected_protocol
        ):
            raise ValueError("stage command is not bound to its registered protocol")
    check_inputs(plan)


def check_inputs(plan):
    for path, digest in plan["inputs_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"planned source/protocol changed: {path}")


def execute(matrix_root, plan_path, out, poll_seconds=60):
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    plan_hash = sha256(plan_path)
    controller_hash = sha256(Path(__file__))
    reserve_output(out)
    Path(f"{out}.plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps({"status": "waiting_for_matrix", "gpu_allocated": False}), flush=True
    )
    while not matrix_ready(matrix_root):
        if (out.parent / "STOP_REQUESTED").exists():
            raise RuntimeError("post-matrix stop requested")
        time.sleep(poll_seconds)
    matrix_hash = sha256(matrix_root / "matrix.complete.json")
    completed = []
    for stage in plan["stages"]:
        if (matrix_root / "STOP_REQUESTED").exists() or (
            out.parent / "STOP_REQUESTED"
        ).exists():
            raise RuntimeError("author stop requested; next stage not launched")
        check_inputs(plan)
        if sha256(plan_path) != plan_hash or sha256(Path(__file__)) != controller_hash:
            raise ValueError("post-matrix configuration/controller changed")
        if sha256(matrix_root / "matrix.complete.json") != matrix_hash:
            raise ValueError("completed matrix identity changed")
        name = stage["name"]
        print(json.dumps({"stage_start": name}), flush=True)
        with Path(f"{out}.{name}.log").open("x") as log:
            subprocess.run(
                stage["command"],
                cwd=plan["cwd"],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        path = Path(stage["receipt"])
        receipt = verify_receipt(path)
        if (
            receipt.get("status") != stage["expected_status"]
            or receipt.get("final_test_images_read") is not False
            or receipt.get("release_ready") is not False
        ):
            raise ValueError("stage completion or sealed-test/release boundary failed")
        completed.append({"stage": name, "receipt": str(path), "sha256": sha256(path)})
        with Path(f"{out}.results.jsonl").open("a") as stream:
            stream.write(json.dumps(completed[-1]) + "\n")
        print(json.dumps({"receipt_landed": completed[-1]}), flush=True)
    check_inputs(plan)
    finish_output(
        out,
        {
            "status": "post_matrix_complete",
            "release_ready": False,
            "final_test_images_read": False,
            "matrix_receipt_sha256": matrix_hash,
            "plan_sha256": plan_hash,
            "controller_sha256": controller_hash,
            "stages": completed,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    execute(args.matrix_run, args.plan, args.out)


if __name__ == "__main__":
    main()
