#!/usr/bin/env python3
"""Why does a recogniser trained on real photos score 0.00 on our renders?

The ChessQueries release checkpoint gets 98.5 exact on ChessCog, which is also synthetic, and 0.00 on
benchmark v1. So the shift is not "synthetic vs photographic" — something about OUR rendering
distribution breaks it. This isolates the first candidate: framing and background.

Their eval path squashes the whole 1920x1080 frame to 644x644. Our renders put the board in roughly
half the frame width on a flat (38,38,46) ground, so after the squash the board is small and sits in a
field of constant colour, unlike ChessCog's tight, textured scenes. Benchmark v1 ships exact corner
ground truth, so we can hand the model the board and nothing else and see what comes back.

Five input variants over the same images, same checkpoint, same decode:
  full      the frame as-is (their eval path; the 0.00 baseline)
  bbox      axis-aligned crop to the corner bounding box, with --pad margin
  warp      GT perspective warp in canonical a8 orientation (oracle diagnostic)
  square    GT bbox padded to a square (changes framing AND aspect ratio)
  letterbox entire frame padded to a square (aspect-ratio control, no GT)

  python tools/paper/crop_diagnostic.py --root ~/bench_v1/renders --freeze ~/bench_v1/FREEZE_2026-09-09.jsonl --splits bench_elev_45_60 \
      --repo-cq .../minimal_reproduction --ckpt .../chessqueries-vitL14-644-joint.safetensors \
      --limit 150 --out tools/reports/paper/crop_diag_2026-09-09
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from .benchmark_io import (
        finish_output,
        preflight,
        reserve_output,
        sha256,
        verify_unchanged,
    )
    from .fen_pools import cells, labels_to_fen, placement
    from .score_benchmark import model_identity
except ImportError:
    from benchmark_io import (
        finish_output,
        preflight,
        reserve_output,
        sha256,
        verify_unchanged,
    )
    from fen_pools import cells, labels_to_fen, placement
    from score_benchmark import model_identity

VARIANTS = ("full", "bbox", "square", "letterbox", "warp")


def variant_image(bgr, corners, kind, pad, a8_image_corner=None):
    """GT-assisted crops; square/letterbox use explicit black padding, never clipping alone."""
    if kind not in VARIANTS:
        raise ValueError(f"unknown input variant: {kind}")
    if not math.isfinite(pad) or not 0 <= pad <= 1:
        raise ValueError("pad must be finite and between 0 and 1")
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) == 0:
        raise ValueError("expected a nonempty three-channel BGR image")
    h, w = bgr.shape[:2]
    if kind == "full":
        return bgr
    if kind == "letterbox":
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        return cv2.copyMakeBorder(
            bgr,
            top,
            side - h - top,
            left,
            side - w - left,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    pts = np.asarray(corners, dtype=np.float32)
    if (
        pts.shape != (4, 2)
        or not np.isfinite(pts).all()
        or not cv2.isContourConvex(pts)
        or abs(cv2.contourArea(pts)) < 1e-6
    ):
        raise ValueError("corners must be a finite, nondegenerate cyclic convex quad")
    if kind == "warp":
        names = ("tl", "tr", "br", "bl")
        if a8_image_corner not in names:
            raise ValueError("GT warp requires the physical a8 cyclic corner index")
        pts = np.roll(pts, -names.index(a8_image_corner), axis=0)
        side = 644
        dst = np.array(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
            dtype=np.float32,
        )
        return cv2.warpPerspective(
            bgr, cv2.getPerspectiveTransform(pts, dst), (side, side)
        )
    x0, y0 = pts.min(axis=0).astype(float)
    x1, y1 = pts.max(axis=0).astype(float)
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0, x1, y1 = x0 - mx, y0 - my, x1 + mx, y1 + my
    if kind == "square":
        side = math.ceil(max(x1 - x0, y1 - y0))
        if side > 3 * max(h, w):
            raise ValueError("square crop exceeds supported image extent")
        x0 = math.floor((x0 + x1 - side) / 2)
        y0 = math.floor((y0 + y1 - side) / 2)
        x1, y1 = x0 + side, y0 + side
    else:
        x0, y0, x1, y1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
    left, top, right, bottom = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if right <= left or bottom <= top:
        raise ValueError("crop does not intersect the image")
    crop = bgr[top:bottom, left:right]
    if kind == "square":
        crop = cv2.copyMakeBorder(
            crop,
            top - y0,
            y1 - bottom,
            left - x0,
            x1 - right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--repo-cq", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument(
        "--limit", type=int, default=150, help="first sorted IDs per split; 0 = all"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if len(a.variants) != len(set(a.variants)):
        ap.error("duplicate variants")
    if not math.isfinite(a.pad) or not 0 <= a.pad <= 1:
        ap.error("pad must be finite and between 0 and 1")
    inventory = preflight(a.root, a.freeze, a.splits, a.limit)
    a.system = "chessqueries"
    receipt = {key: value for key, value in inventory.items() if key != "rows"}
    receipt.update(
        model_inputs_sha256=model_identity(a),
        command=sys.argv,
        variants=a.variants,
        pad=a.pad,
        padding="constant black",
        tool_sha256={
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("benchmark_io.py"),
                Path(__file__).with_name("fen_pools.py"),
            )
        },
    )
    out = Path(a.out)
    reserve_output(out)
    import torch

    sys.path.insert(0, a.repo_cq)
    from eval import load_model

    from data import eval_transform

    torch.set_float32_matmul_precision("high")
    model = load_model(Path(a.ckpt), "cuda")
    tf = eval_transform(644)

    def predict(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        with torch.no_grad():
            return labels_to_fen(
                model(tf(t)[None].to("cuda")).argmax(-1)[0].cpu().tolist()
            )

    results = []
    for split in inventory["splits"]:
        rows = [r for r in inventory["rows"] if r["split"] == split]
        for kind in a.variants:
            ex = sq = 0
            t0 = time.time()
            with open(f"{out}.{split}.{kind}.tsv", "x") as per_board:
                per_board.write("id\tgt\tpred\tn_wrong\tinput_width\tinput_height\n")
                for r in rows:
                    image = a.root / split / "images" / f"{r['id']}.png"
                    transformed = variant_image(
                        cv2.imread(str(image)),
                        r["corners_px"],
                        kind,
                        a.pad,
                        a8_image_corner=r["a8_image_corner"],
                    )
                    pred = placement(predict(transformed))
                    nw = sum(x != y for x, y in zip(cells(r["fen"]), cells(pred)))
                    ex += nw == 0
                    sq += 64 - nw
                    h, w = transformed.shape[:2]
                    per_board.write(f"{r['id']}\t{r['fen']}\t{pred}\t{nw}\t{w}\t{h}\n")
            n = len(rows)
            row = dict(
                split=split,
                variant=kind,
                n=n,
                exact_boards=ex,
                correct_squares=sq,
                total_squares=64 * n,
                exact=round(100 * ex / n, 2),
                per_square=round(100 * sq / (64 * n), 2),
                uses_ground_truth=kind not in ("full", "letterbox"),
                ms_per_img=round(1000 * (time.time() - t0) / n, 1),
            )
            results.append(row)
            print(json.dumps(row), flush=True)
    with open(f"{out}.json", "x") as stream:
        json.dump(results, stream, indent=2)
    with open(f"{out}.md", "x") as f:
        f.write("# ChessQueries input diagnostic (not deployment accuracy)\n\n")
        f.write(
            f"pad={a.pad}; {inventory['selected_n']}/{inventory['frozen_n']} frozen frames. "
            "Square and letterbox use black padding; bbox/square/warp use GT corners. "
            "Warp also supplies canonical a8 orientation.\n\n"
        )
        f.write(
            "| split | variant | n | exact % | per-square % |\n|---|---|---:|---:|---:|\n"
        )
        for r in results:
            f.write(
                f"| {r['split']} | {r['variant']} | {r['n']} | {r['exact']} | {r['per_square']} |\n"
            )
    verify_unchanged(a.root, inventory, receipt["model_inputs_sha256"])
    finish_output(out, dict(receipt, results=results, status="complete_for_selection"))
    print("completed", f"{out}.complete.json")


if __name__ == "__main__":
    main()
