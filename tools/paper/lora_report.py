"""CPU-only, receipt-bound reporting after all ten fixed C2 points finish."""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paper.benchmark_io import finish_output, reserve_output, sha256
from tools.paper.transfer_cohort import verify_receipt

SYMBOLS = ".PNBRQKpnbrqk"
RUNS = [
    f"b{budget}_{arm}_s{seed}"
    for budget, seeds in [(500, [303]), (2000, [303, 304, 305]), (8000, [303])]
    for seed in seeds
    for arm in ["dark", "both"]
]


def expand(fen):
    ranks = fen.split()[0].split("/")
    if len(ranks) != 8:
        raise ValueError("FEN must contain eight ranks")
    cells = []
    for rank in ranks:
        row = []
        for char in rank:
            if char in "12345678":
                row.extend("." * int(char))
            elif char in SYMBOLS[1:]:
                row.append(char)
            else:
                raise ValueError("invalid FEN symbol")
        if len(row) != 8:
            raise ValueError("FEN rank does not cover eight squares")
        cells.extend(row)
    return cells


def rescore(rows, *, dev):
    """Independent FEN-string rescore, not trust in saved exact/count fields."""
    seen = set()
    group_fens = {}
    for row in rows:
        key = (
            (row["position_group"], row["factor_cell"], row["parity"])
            if dev
            else row["id"]
        )
        if key in seen:
            raise ValueError("duplicate prediction key")
        seen.add(key)
        if dev:
            if row["parity"] not in ("a1_dark", "a1_light"):
                raise ValueError("unknown parity")
            if row.get("partition", "dev") != "dev":
                raise ValueError("only development predictions are allowed")
            old = group_fens.setdefault(row["position_group"], row["gt"])
            if old != row["gt"]:
                raise ValueError("ground truth changes within a position group")
        correct = sum(a == b for a, b in zip(expand(row["gt"]), expand(row["pred"])))
        if type(row["squares_correct"]) is not int or correct != row["squares_correct"]:
            raise ValueError("stored square score disagrees with FEN rescore")
        if type(row["exact"]) is not bool or row["exact"] != (correct == 64):
            raise ValueError("stored exact score disagrees with FEN rescore")
    if not rows:
        raise ValueError("empty predictions")
    return {
        "n": len(rows),
        "exact_boards": sum(r["exact"] for r in rows),
        "exact_pct": 100 * sum(r["exact"] for r in rows) / len(rows),
        "per_square_pct": 100
        * sum(r["squares_correct"] for r in rows)
        / (64 * len(rows)),
    }


def require_metrics(actual, saved):
    for key, value in actual.items():
        if key not in saved or not math.isclose(
            value, saved[key], rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(f"aggregate metric mismatch: {key}")


def validate_dev_inventory(rows):
    by_group = defaultdict(set)
    for row in rows:
        by_group[row["position_group"]].add((row["factor_cell"], row["parity"]))
    expected = next(iter(by_group.values()), set())
    if (
        len(rows) != 16000
        or len(by_group) != 400
        or len(expected) != 40
        or any(cells != expected for cells in by_group.values())
    ):
        raise ValueError("expected the complete 400-group, 40-view development matrix")
    cells = {cell for cell, _ in expected}
    if len(cells) != 20 or expected != {
        (cell, parity) for cell in cells for parity in ["a1_dark", "a1_light"]
    }:
        raise ValueError("development matrix is not fully parity-paired")


def load_point(results, name):
    prefix = results / name
    receipt_path = Path(f"{prefix}.complete.json")
    receipt = verify_receipt(receipt_path)
    if (
        receipt.get("status") != "c2_complete"
        or receipt.get("final_test_images_read") is not False
        or receipt.get("scored_partitions") != ["dev", "chessred_test"]
    ):
        raise ValueError("invalid point completion or sealed-test boundary")
    dev = [
        json.loads(line)
        for line in Path(f"{prefix}.dev_predictions.jsonl").read_text().splitlines()
    ]
    public = [
        json.loads(line)
        for line in Path(f"{prefix}.chessred_test_predictions.jsonl")
        .read_text()
        .splitlines()
    ]
    metrics = json.loads(Path(f"{prefix}.metrics.json").read_text())
    require_metrics(rescore(dev, dev=True), metrics["dev"]["overall"])
    require_metrics(rescore(public, dev=False), metrics["chessred_test"])
    validate_dev_inventory(dev)
    if len(public) != 2129:
        raise ValueError("ChessReD inventory must contain 2,129 test images")
    for parity, saved in metrics["dev"]["by_parity"].items():
        require_metrics(
            rescore([r for r in dev if r["parity"] == parity], dev=True), saved
        )
    for cell_parity, saved in metrics["dev"]["by_factor_cell_parity"].items():
        cell, parity = cell_parity.split("/")
        require_metrics(
            rescore(
                [r for r in dev if r["factor_cell"] == cell and r["parity"] == parity],
                dev=True,
            ),
            saved,
        )
    retained = metrics["chessred_test"]["exact_pct"] >= 99.12
    if (
        receipt.get("accepted") is not retained
        or receipt.get("public_regression_pass") is not retained
    ):
        raise ValueError(
            "retention flag disagrees with independently rescored exact accuracy"
        )
    return receipt, metrics, dev, public


def report(run, out, protocol):
    matrix_path = run / "matrix.complete.json"
    matrix = verify_receipt(matrix_path)
    if (
        matrix.get("status") != "ten_fixed_cells_complete"
        or [x["run"] for x in matrix["runs"]] != RUNS
    ):
        raise ValueError("all ten registered points must be complete before reporting")
    inputs = {
        str(matrix_path.resolve()): sha256(matrix_path),
        str(protocol.resolve()): sha256(protocol),
        str(Path(__file__).resolve()): sha256(Path(__file__)),
    }
    results = run / "results"
    base_receipt, base_metrics, baseline, base_public = load_point(results, "release")
    base_hash = sha256(results / "release.complete.json")
    if matrix["reference_receipt_sha256"] != base_hash or not base_receipt["accepted"]:
        raise ValueError("matrix baseline binding or retention failed")
    base_index = {
        (r["position_group"], r["factor_cell"], r["parity"]): r for r in baseline
    }
    base_public_index = {r["id"]: r["gt"] for r in base_public}
    points = {"release": {"metrics": base_metrics, "receipt_sha256": base_hash}}
    for entry in matrix["runs"]:
        name = entry["run"]
        receipt_path = results / f"{name}.complete.json"
        receipt, metrics, rows, public = load_point(results, name)
        if (
            sha256(receipt_path) != entry["receipt_sha256"]
            or receipt["accepted"] != entry["accepted"]
        ):
            raise ValueError("matrix/point receipt mismatch")
        spec = receipt["run_spec"]
        if (
            f"b{spec['budget']}_{spec['arm']}_s{spec['seed']}" != name
            or spec["train_head"] is not False
            or receipt["recipe"] != base_receipt["recipe"]
        ):
            raise ValueError("point differs from frozen recipe")
        for field in [
            "chessqueries_revision",
            "base_checkpoint_sha256",
            "public_data_inputs_sha256",
        ]:
            if receipt[field] != base_receipt[field]:
                raise ValueError(f"point baseline provenance mismatch: {field}")
        for path, digest in base_receipt["model_inputs_sha256"].items():
            if receipt["model_inputs_sha256"].get(path) != digest:
                raise ValueError(
                    "point and baseline do not bind the same input images and source"
                )
        index = {(r["position_group"], r["factor_cell"], r["parity"]): r for r in rows}
        if set(index) != set(base_index) or any(
            index[k]["gt"] != old["gt"] for k, old in base_index.items()
        ):
            raise ValueError("development identity or ground truth changed")
        if {r["id"]: r["gt"] for r in public} != base_public_index:
            raise ValueError("public test identity or ground truth changed")
        paired = metrics["paired_vs_release"]
        gain = sum(r["exact"] and not base_index[k]["exact"] for k, r in index.items())
        lost = sum(not r["exact"] and base_index[k]["exact"] for k, r in index.items())
        require_metrics(
            {
                "gained": gain,
                "lost": lost,
                "exact_delta_pp": metrics["dev"]["overall"]["exact_pct"]
                - base_metrics["dev"]["overall"]["exact_pct"],
                "square_delta_pp": metrics["dev"]["overall"]["per_square_pct"]
                - base_metrics["dev"]["overall"]["per_square_pct"],
            },
            paired,
        )
        points[name] = {
            "metrics": metrics,
            "receipt_sha256": entry["receipt_sha256"],
            "accepted": receipt["accepted"],
            "fit": receipt["fit"],
        }
        inputs[str(receipt_path.resolve())] = entry["receipt_sha256"]
    inputs[str((results / "release.complete.json").resolve())] = base_hash
    reserve_output(out)
    lines = [
        "# Fixed-budget LoRA adaptation curve",
        "",
        "All ten preregistered points are reported; no best-seed selection.",
        "Evaluation: 16,000 dev cell-images, **400 independent position groups**, both parities; ChessReD n=2,129.",
        "The 100 final-test position groups remain sealed. `release_ready=false`.",
        "",
        "## Main result",
        "",
        "| Point | Dark exact % | Light exact % | Dark square % | Light square % | ChessReD exact % | Retention |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, point in points.items():
        m = point["metrics"]
        d, l = [m["dev"]["by_parity"][p] for p in ["a1_dark", "a1_light"]]
        flag = (
            "reference"
            if name == "release"
            else ("pass" if point["accepted"] else "FAIL")
        )
        lines.append(
            f"| {name} | {d['exact_pct']:.3f} | {l['exact_pct']:.3f} | {d['per_square_pct']:.3f} | {l['per_square_pct']:.3f} | {m['chessred_test']['exact_pct']:.3f} | {flag} |"
        )
    lines += [
        "",
        "## Paired development change versus release",
        "",
        "Whole-position bootstrap: 2,000 draws, seed 20260913; saved preregistered 95% intervals.",
        "Point estimates and gained/lost counts independently recomputed from FEN predictions; intervals retained from the bound run artifacts.",
        "| Point | Exact delta pp [95% CI] | Square delta pp [95% CI] | Gained / lost cell-images |",
        "|---|---|---|---|",
    ]
    for name in RUNS:
        b = points[name]["metrics"]["paired_vs_release"]
        lines.append(
            f"| {name} | {b['exact_delta_pp']:+.3f} [{b['exact_ci95_pp'][0]:+.3f}, {b['exact_ci95_pp'][1]:+.3f}] | {b['square_delta_pp']:+.3f} [{b['square_ci95_pp'][0]:+.3f}, {b['square_ci95_pp'][1]:+.3f}] | {b['gained']} / {b['lost']} |"
        )
    lines += [
        "",
        "## Seed variation at 2,000 groups",
        "",
        "Mean ± sample SD across the three fixed training seeds, not a confidence interval or three independent datasets.",
        "| Arm | Dark exact % | Light exact % |",
        "|---|---:|---:|",
    ]
    for arm in ["dark", "both"]:
        values = [
            [
                points[f"b2000_{arm}_s{seed}"]["metrics"]["dev"]["by_parity"][p][
                    "exact_pct"
                ]
                for seed in [303, 304, 305]
            ]
            for p in ["a1_dark", "a1_light"]
        ]
        lines.append(
            f"| {arm} | "
            + " | ".join(
                f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" for v in values
            )
            + " |"
        )
    lines += [
        "",
        "## Every factor cell × parity",
        "",
        "Entries are exact % / per-square %. No cell is omitted because it failed.",
    ]
    for parity in ["a1_dark", "a1_light"]:
        lines += [
            "",
            f"### {parity}",
            "",
            "| Cell | " + " | ".join(points) + " |",
            "|---|" + "---:|" * len(points),
        ]
        for cell in sorted(base_metrics["dev"]["by_factor_cell"]):
            values = [
                p["metrics"]["dev"]["by_factor_cell_parity"][f"{cell}/{parity}"]
                for p in points.values()
            ]
            lines.append(
                f"| {cell} | "
                + " | ".join(
                    f"{v['exact_pct']:.2f}/{v['per_square_pct']:.2f}" for v in values
                )
                + " |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Budgets count positions: dark uses N distinct images; both uses 2N. Both receive 9,000 presentations and 1,500 updates, so distinct-image coverage is not matched.",
        "- All runs use encoder-only rank-8 LoRA and frozen readout. Fixed final checkpoint; no early stopping, hyperparameter sweep, or selector.",
        "- The retention gate is ChessReD exact ≥99.12%; passing is not a general-superiority or no-cost-invariance claim. Failed retention points remain in every table.",
        "- Development cell-images reuse positions and three reference configurations; pooled scores weight the registered cells, not unique images or independent boards.",
        "- C1/dev is synthetic development data. Neither these scores nor public retention establish unseen-real-world robustness, SOTA, or final-test performance.",
        "- Training/evaluation wall times are not isolated per-image latency; use the separate C4 measurement.",
        "",
        "## Receipt identities",
        "",
    ]
    lines += [
        f"- `{name}`: `{point['receipt_sha256']}`" for name, point in points.items()
    ]
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    Path(f"{out}.json").write_text(json.dumps(points, indent=2, allow_nan=False) + "\n")
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError("report source changed during aggregation")
    finish_output(
        out,
        {
            "status": "lora_curve_report_complete",
            "release_ready": False,
            "final_test_images_read": False,
            "gpu_used": False,
            "inputs_sha256": inputs,
            "fixed_points": RUNS,
            "independent_fen_rescore": True,
            "bootstrap_intervals": "retained from bound run metrics",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    report(args.run, args.out, args.protocol)


if __name__ == "__main__":
    main()
