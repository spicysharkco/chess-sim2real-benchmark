#!/usr/bin/env python3
"""Annotate a benchmark v1 split: exact corner ground truth from the camera, plus the colour
detector's agreement as a measured statistic rather than a gate.

`rendering/annotate_orientation.py` is the training-lane annotator: it reprojects the Three.js camera,
compares against `src/board.py::find_corners`, and RAISES when the two disagree, which is right for
training data built to be detectable. A benchmark deliberately contains views where a detector fails
(low elevation, heavy occlusion), so failing the set is the wrong response — the failure rate is one
of the things the benchmark measures.

For every row this writes:
  corners/<id>.txt        four "x,y" lines, clockwise from min(x+y), playing plane, rounded to 0.001px
  metadata.jsonl          + a8_image_corner, corners_px, detector_corners_px, detector_error_px
and prints a per-split summary: detector found / missed / disagreed, and the error quantiles.

  python tools/paper/annotate_benchmark.py --data ~/bench_v1/renders/bench_elev_08_15 [--repo .]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CORNER_NAMES = ("tl", "tr", "br", "bl")


def load_helpers(repo: Path):
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "rendering"))
    import annotate_orientation as ao  # noqa: E402

    import board  # noqa: E402

    return board, ao


def ground_truth(row, board, ao):
    """Validate camera/slab bounds separately from the four playing-plane corners."""
    name, projected, bounds_error = ao.orientation_for(row)
    ordered = board.order_corners(projected)
    return dict(
        a8_image_corner=name,
        a8_corner_index=CORNER_NAMES.index(name),
        corners_px=np.asarray(ordered, dtype=float).round(3).tolist(),
        corner_order="clockwise_from_min_x_plus_y",
        corner_surface="playing_plane_y0",
        projection_bounds_max_abs_error=bounds_error,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="validate projection without image reads or writes",
    )
    a = ap.parse_args()
    board, ao = load_helpers(a.repo)

    meta = a.data / "metadata.jsonl"
    rows = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
    if a.limit < 0:
        ap.error("limit must be nonnegative")
    selected = rows[: a.limit] if a.limit else rows
    # All projection checks precede any writes; --limit must not truncate metadata.
    projections = [ground_truth(row, board, ao) for row in selected]
    if a.check_only:
        print(
            json.dumps(
                dict(
                    n=len(selected),
                    max_bounds_error=max(
                        (p["projection_bounds_max_abs_error"] for p in projections),
                        default=None,
                    ),
                )
            )
        )
        return
    (a.data / "corners").mkdir(exist_ok=True)

    errors, missed = [], 0
    for row, projection in zip(selected, projections):
        row.update(projection)
        ordered = np.asarray(row["corners_px"], dtype=float)
        (a.data / "corners" / f"{row['id']}.txt").write_text(
            "".join(f"{x:.3f},{y:.3f}\n" for x, y in row["corners_px"])
        )

        img = cv2.imread(str(a.data / "images" / f"{row['id']}.png"))
        detected = (
            board.find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if img is not None
            else None
        )
        if detected is None:
            row["detector_corners_px"] = None
            row["detector_error_px"] = None
            missed += 1
            continue
        det = board.order_corners(np.asarray(detected, dtype=np.float32))
        # order_corners is cyclic; take the best rotation, as the training annotator does
        err = min(
            float(np.linalg.norm(det - np.roll(ordered, s, axis=0), axis=1).mean())
            for s in range(4)
        )
        row["detector_corners_px"] = np.asarray(det, dtype=float).round(3).tolist()
        row["detector_error_px"] = round(err, 3)
        errors.append(err)

    meta.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    e = np.array(errors) if errors else np.array([np.nan])
    summary = dict(
        split=a.data.name,
        n=len(selected),
        detector_missed=missed,
        detector_found=len(errors),
        err_median=round(float(np.nanmedian(e)), 2),
        err_p90=round(float(np.nanpercentile(e, 90)), 2),
        err_max=round(float(np.nanmax(e)), 2),
        over_25px=int((e > 25).sum()),
    )
    (a.data / "corner_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
