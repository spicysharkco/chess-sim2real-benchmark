"""Bounded, paired checkerboard-parity and query-colour inference experiment.

Protocol: docs/PARITY_PROBE_2026-09-09.md. Uses an isolated renderer/asset snapshot,
never modifies source renders/checkpoints, and refuses an occupied GPU.
"""

import argparse
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .benchmark_io import (
        finish_output,
        preflight,
        read_jsonl,
        reserve_output,
        sha256,
        split_rows,
    )
    from .fen_pools import cells, labels_to_fen, placement
except ImportError:
    from benchmark_io import (
        finish_output,
        preflight,
        read_jsonl,
        reserve_output,
        sha256,
        split_rows,
    )
    from fen_pools import cells, labels_to_fen, placement

SPLITS = (
    "bench_elev_08_15",
    "bench_elev_15_25",
    "bench_elev_25_45",
    "bench_elev_45_60",
)
CHECKPOINT_SHA = "6151bdd98fbe25f32080c097eba7ae75808b5615160e4867240931085fc762e5"
OLD_PARITY = b"(r+c)%2===1?lm:dm"
NEW_PARITY = b"(r+c)%2===1?dm:lm"


def swap_parity(source):
    pattern = rb"(\(r\+c\)%2===1\s*\?\s*)lm(\s*:\s*)dm(?![A-Za-z_0-9])"
    if len(re.findall(pattern, source)) != 1:
        raise ValueError("expected exactly one original checkerboard assignment")
    return re.sub(pattern, lambda match: match[1] + b"dm" + match[2] + b"lm", source)


def select_positions(rows, seed=20260909):
    selected = []
    for band, split in enumerate(SPLITS):
        rng = random.Random(seed + band)
        population = sorted(
            (r for r in rows if r["split"] == split), key=lambda r: r["id"]
        )
        for quadrant in range(4):
            candidates = [
                r
                for r in population
                if quadrant * 90 <= r["azimuth_deg"] < (quadrant + 1) * 90
            ]
            if len(candidates) < 12:
                raise ValueError(f"insufficient rows in {split}, quadrant {quadrant}")
            for index, row in enumerate(rng.sample(candidates, 12)):
                selected.append(
                    dict(row, probe_wave="pilot" if index < 6 else "confirmation")
                )
    return selected


def prepare(run, source_root, freeze):
    inventory = preflight(source_root, freeze, list(SPLITS))
    selected = select_positions(inventory["rows"])
    metadata = {
        (split, r["id"]): r for split in SPLITS for r in split_rows(source_root / split)
    }
    rendering = run / "src/rendering"
    source = (rendering / "renderer_geometry.html").read_bytes()
    treatment = swap_parity(source)
    prefix = run / "prepare"
    reserve_output(prefix)
    with (rendering / "renderer_parity_swap.html").open("xb") as stream:
        stream.write(treatment)
    rows = []
    for index, frozen in enumerate(selected):
        split, original_id = frozen["split"], frozen["id"]
        row = dict(metadata[(split, original_id)])
        if (
            sha256(rendering / "renderer_geometry.html")
            != row["source_sha256"]["renderer_geometry.html"]
        ):
            raise ValueError(
                "renderer snapshot differs from recorded original renderer"
            )
        row.update(
            id=f"{index:06d}",
            probe_source_split=split,
            probe_source_id=original_id,
            probe_wave=frozen["probe_wave"],
            probe_original_sha256=frozen["sha256"],
        )
        rows.append(row)
    with Path(f"{prefix}.manifest.jsonl").open("x") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)
    plan = dict(
        n=len(rows),
        seed=20260909,
        source_root=str(source_root.resolve()),
        freeze=str(freeze.resolve()),
        freeze_sha256=inventory["freeze_sha256"],
        checkpoint_sha256=CHECKPOINT_SHA,
        render_sources_sha256={
            str(p.relative_to(rendering)): sha256(p)
            for p in sorted(rendering.rglob("*"))
            if p.is_file()
        },
    )
    with Path(f"{prefix}.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
    finish_output(prefix, dict(status="prepared_not_evaluated", **plan))
    print(
        json.dumps(dict(prepared=len(rows), waves=["pilot", "confirmation"])),
        flush=True,
    )


def validate_pairs(run):
    if (run / "prepare.incomplete").exists() or not (
        run / "prepare.complete.json"
    ).is_file():
        raise ValueError("preparation is incomplete")
    plan = json.loads((run / "prepare.json").read_text())
    for relative, digest in plan["render_sources_sha256"].items():
        if sha256(run / "src/rendering" / relative) != digest:
            raise ValueError(f"render source changed: {relative}")
    rows = read_jsonl(run / "prepare.manifest.jsonl")
    expected = {r["id"] for r in rows}
    if len(rows) != 192 or len(expected) != 192:
        raise ValueError("the fixed protocol requires 192 unique paired positions")
    rendered = {
        variant: {r["id"]: r for r in split_rows(run / f"render_{variant}")}
        for variant in ("control", "parity")
    }
    if any(set(values) != expected for values in rendered.values()):
        raise ValueError("incomplete paired render population")
    hashes = {}
    for row in rows:
        control, parity = rendered["control"][row["id"]], rendered["parity"][row["id"]]
        if control["renderer"] != parity["renderer"] or any(
            control[field] != parity[field] or control[field] != row[field]
            for field in (
                "fen",
                "camera",
                "piece_geometry",
                "corners_px",
                "a8_image_corner",
            )
        ):
            raise ValueError(f"non-parity factors differ for {row['id']}")
        original = (
            Path(plan["source_root"])
            / row["probe_source_split"]
            / "images"
            / f"{row['probe_source_id']}.png"
        )
        for variant in ("control", "parity"):
            path = run / f"render_{variant}/images/{row['id']}.png"
            digest = sha256(path)
            if digest != rendered[variant][row["id"]]["image_sha256"]:
                raise ValueError(f"render image hash mismatch: {path}")
            hashes[f"{variant}/{row['id']}"] = digest
        if (
            hashes[f"control/{row['id']}"] != row["probe_original_sha256"]
            or sha256(original) != row["probe_original_sha256"]
        ):
            raise ValueError(
                f"control is not byte-identical to frozen original: {row['id']}"
            )
        if hashes[f"control/{row['id']}"] == hashes[f"parity/{row['id']}"]:
            raise ValueError("parity treatment did not change the image")
    return rows, hashes


def gpu_idle(allow_pid=None):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    others = [
        line.strip()
        for line in output.splitlines()
        if line.strip() and line.strip() != str(allow_pid)
    ]
    if others:
        raise RuntimeError(f"GPU 0 occupied; refusing to interfere: {others}")


def metrics(rows):
    n = len(rows)
    wrong = sum(r["n_wrong"] for r in rows)
    occupied = sum(r["occupied_n"] for r in rows)
    return dict(
        n=n,
        exact_boards=sum(r["n_wrong"] == 0 for r in rows),
        exact_pct=100 * sum(r["n_wrong"] == 0 for r in rows) / n,
        per_square_pct=100 * (64 * n - wrong) / (64 * n),
        occupied_square_pct=100 * sum(r["occupied_correct"] for r in rows) / occupied,
    )


def paired_effect(control, treatment):
    if set(control) != set(treatment):
        raise ValueError("paired populations differ")
    groups = defaultdict(list)
    gained = lost = 0
    for key, a in control.items():
        b = treatment[key]
        ac, bc = a["n_wrong"] == 0, b["n_wrong"] == 0
        gained += bc and not ac
        lost += ac and not bc
        groups[a["source_split"]].append(int(bc) - int(ac))
    rng = np.random.default_rng(20260909)
    samples = np.zeros(2000)
    for values in groups.values():
        samples += rng.choice(values, size=(2000, len(values)), replace=True).mean(
            axis=1
        ) / len(groups)
    return dict(
        n=len(control),
        gained=int(gained),
        lost=int(lost),
        exact_delta_pp=100 * (gained - lost) / len(control),
        paired_bootstrap_95_pct=(100 * np.percentile(samples, [2.5, 97.5])).tolist(),
    )


def evaluate(run, out, batch_size):
    rows, hashes = validate_pairs(run)
    checkpoint = run / "upstream/checkpoint.safetensors"
    if sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("wrong release checkpoint")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("this bounded probe requires CUDA_VISIBLE_DEVICES=0 on render-host")
    if batch_size not in (1, 2, 4):
        raise ValueError("batch size must be 1, 2, or 4")
    gpu_idle()
    reserve_output(out)
    import sys

    import torch
    from torchvision.io import read_image

    sys.path.insert(0, str(run / "upstream/minimal_reproduction"))
    from eval import load_model

    from data import eval_transform

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    model = load_model(checkpoint, "cuda")
    buffers = [
        (name, value)
        for name, value in model.named_buffers()
        if name.endswith("color_idx")
    ]
    if len(buffers) != 1 or tuple(buffers[0][1].shape) != (64,):
        raise ValueError("unexpected query-colour buffer contract")
    buffer_name, color_buffer = buffers[0]
    original_colors = color_buffer.detach().clone()
    if set(original_colors.cpu().tolist()) != {0, 1}:
        raise ValueError("query-colour buffer is not binary")
    transform = eval_transform(644)
    predictions = []
    with open(f"{out}.predictions.jsonl", "x") as stream, torch.inference_mode():
        try:
            for wave in ("pilot", "confirmation"):
                selected = [r for r in rows if r["probe_wave"] == wave]
                for query in ("default", "flipped"):
                    color_buffer.copy_(
                        original_colors if query == "default" else 1 - original_colors
                    )
                    for image_variant in ("control", "parity"):
                        gpu_idle(os.getpid())
                        condition = f"{image_variant}_{query}"
                        current = []
                        for start in range(0, len(selected), batch_size):
                            batch = selected[start : start + batch_size]
                            images = torch.stack(
                                [
                                    transform(
                                        read_image(
                                            str(
                                                run
                                                / f"render_{image_variant}/images/{r['id']}.png"
                                            )
                                        ).float()
                                    )
                                    for r in batch
                                ]
                            ).to("cuda")
                            outputs = model(images).argmax(-1).cpu().tolist()
                            for row, output in zip(batch, outputs):
                                pred = labels_to_fen(output)
                                gt = placement(row["fen"])
                                g, p = cells(gt), cells(pred)
                                record = dict(
                                    id=row["id"],
                                    wave=wave,
                                    condition=condition,
                                    source_split=row["probe_source_split"],
                                    source_id=row["probe_source_id"],
                                    gt=gt,
                                    pred=pred,
                                    n_wrong=sum(a != b for a, b in zip(g, p)),
                                    occupied_n=sum(a != "." for a in g),
                                    occupied_correct=sum(
                                        a != "." and a == b for a, b in zip(g, p)
                                    ),
                                )
                                current.append(record)
                                predictions.append(record)
                                stream.write(json.dumps(record) + "\n")
                            stream.flush()
                        print(
                            json.dumps(
                                dict(wave=wave, condition=condition, **metrics(current))
                            ),
                            flush=True,
                        )
        finally:
            color_buffer.copy_(original_colors)
    _, after_hashes = validate_pairs(run)
    if hashes != after_hashes or sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("inputs changed during inference")
    summary = {}
    for wave in ("pilot", "confirmation"):
        grouped = {
            condition: [
                r
                for r in predictions
                if r["wave"] == wave and r["condition"] == condition
            ]
            for condition in (
                "control_default",
                "parity_default",
                "control_flipped",
                "parity_flipped",
            )
        }
        summary[wave] = dict(
            conditions={key: metrics(value) for key, value in grouped.items()},
            primary_effect=paired_effect(
                {r["id"]: r for r in grouped["control_default"]},
                {r["id"]: r for r in grouped["parity_default"]},
            ),
        )
    with open(f"{out}.json", "x") as stream:
        json.dump(summary, stream, indent=2)
    finish_output(
        out,
        dict(
            status="completed_exploratory_probe",
            positions=len(rows),
            image_forwards=len(predictions),
            checkpoint_sha256=CHECKPOINT_SHA,
            query_buffer=buffer_name,
            original_color_indices=original_colors.cpu().tolist(),
            image_sha256=hashes,
            tool_sha256=sha256(__file__),
            batch_size=batch_size,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            summary=summary,
        ),
    )
    print("completed", f"{out}.complete.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "validate", "evaluate"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.source_root is None or args.freeze is None:
            parser.error("prepare requires --source-root and --freeze")
        prepare(args.run, args.source_root, args.freeze)
    elif args.mode == "validate":
        rows, _ = validate_pairs(args.run)
        print(
            json.dumps(
                dict(validated_pairs=len(rows), controls_identical_to_frozen=True)
            )
        )
    else:
        if args.out is None:
            parser.error("evaluate requires --out")
        evaluate(args.run, args.out, args.batch_size)


if __name__ == "__main__":
    main()
