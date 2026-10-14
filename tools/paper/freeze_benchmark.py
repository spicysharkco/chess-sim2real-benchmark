#!/usr/bin/env python3
"""Freeze a benchmark v1 render tree: per-image md5 manifest, factor levels, and honest detector stats.

Two jobs, one pass over the already-annotated splits (no re-render, no re-detect):

1. FREEZE. `<out>/FREEZE_<date>.jsonl` gets one row per image: split, id, md5, sha256 of the PNG,
   the placement FEN, the four ground-truth corners, and the factor levels (elevation, azimuth,
   framing, geometry asset, palette) so a leaderboard entry can be scored per factor.
   `<out>/FREEZE_<date>.json` gets the split-level totals and the freeze-tool revision (NOT the render
   revision). Per-row source hashes and exclusion fingerprints are retained in JSONL.
   Existing prefixes are never overwritten; a completion receipt is written last.

2. DETECTOR STATS, stated correctly. The first pass reported a mean corner error, which is dominated
   by a single divergent vertex: `src/board.py::find_corners` intersects the four edge lines, and when
   two are near-parallel one intersection shoots off the canvas (observed: x = -35981 on a frame whose
   other three corners are within 3 px). So report, per split:
     - `found`            the detector returned a quad at all
     - `divergent`        at least one vertex lands outside a generous canvas margin -> unusable quad
     - `median_px`/`p90`  corner error over the frames whose quad is NOT divergent
   That separates "slightly off" from "one vertex diverged", which are different failures with
   different fixes, and it is the number the paper can defend.

  python tools/paper/freeze_benchmark.py --root ~/bench_v1/renders --out ~/bench_v1
"""

import argparse
import datetime
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from .benchmark_io import (
        finish_output,
        load_freeze,
        png_dimensions,
        reserve_output,
        split_rows,
    )
    from .fen_pools import cells, sym_key
except ImportError:
    from benchmark_io import (
        finish_output,
        load_freeze,
        png_dimensions,
        reserve_output,
        split_rows,
    )
    from fen_pools import cells, sym_key

MARGIN = 2.0  # a vertex further than 2x the image size outside the frame is divergent, not "off"


def grid_transfer_error(gt, detected):
    """Mean 9x9 grid transfer error in square units, after cyclic alignment.

    Project the true board grid to the image, then map it back with the detected
    homography. Unlike an image-edge denominator, this accounts for foreshortening.
    None means the detected quad/homography cannot support this metric, not zero error.
    """
    if detected is None:
        return None
    gt, detected = np.asarray(gt, np.float32), np.asarray(detected, np.float32)
    for quad in (gt, detected):
        if (
            quad.shape != (4, 2)
            or not np.isfinite(quad).all()
            or not cv2.isContourConvex(quad)
            or abs(cv2.contourArea(quad)) < 1e-6
        ):
            return None
    detected = min(
        (np.roll(detected, k, axis=0) for k in range(4)),
        key=lambda quad: float(np.linalg.norm(gt - quad, axis=1).mean()),
    )
    canonical = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32)
    grid = np.array([[x, y, 1] for y in range(9) for x in range(9)], dtype=float)
    transform = cv2.getPerspectiveTransform(
        detected, canonical
    ) @ cv2.getPerspectiveTransform(canonical, gt)
    mapped = grid @ transform.T
    denom = mapped[:, 2]
    if (
        not np.isfinite(mapped).all()
        or np.any(abs(denom) < 1e-8)
        or not (np.all(denom > 0) or np.all(denom < 0))
    ):
        return None  # A projective horizon intersects the board.
    return float(
        np.linalg.norm(mapped[:, :2] / denom[:, None] - grid[:, :2], axis=1).mean()
    )


def digest(path: Path):
    h5, h256 = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h5.update(chunk)
            h256.update(chunk)
    return h5.hexdigest(), h256.hexdigest()


def git_rev(repo: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, type=Path, help="directory of rendered splits"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument(
        "--name", default=None, help="fresh basename; never overwrite an earlier freeze"
    )
    a = ap.parse_args()
    date = datetime.date.today().isoformat()
    lo_x, hi_x = -MARGIN * a.width, (1 + MARGIN) * a.width
    lo_y, hi_y = -MARGIN * a.height, (1 + MARGIN) * a.height

    if a.width <= 0 or a.height <= 0:
        ap.error("image dimensions must be positive")
    name = a.name or f"FREEZE_{date}"
    if Path(name).name != name or name in (".", ".."):
        ap.error("name must be a basename")
    directories = sorted(
        p
        for p in a.root.iterdir()
        if p.is_dir()
        and any(
            (p / child).exists() for child in ("metadata.jsonl", "images", "labels")
        )
    )
    if not directories or (a.root / "INCOMPLETE").exists():
        raise ValueError("empty or incomplete benchmark root")
    populations = [(directory, split_rows(directory)) for directory in directories]
    seen = set()
    exclusion_snapshots = Counter()
    for _, rows in populations:
        for row in rows:
            key = sym_key(row["fen"])
            if key in seen:
                raise ValueError("duplicate exact/symmetry-equivalent FEN in freeze")
            seen.add(key)
            if row.get("piece_count") != 64 - cells(row["fen"]).count("."):
                raise ValueError(f"{row['id']}: piece_count does not match FEN")
            sources = row.get("source_sha256")
            if (
                not isinstance(sources, dict)
                or not sources
                or any(
                    not isinstance(v, str)
                    or len(v) != 64
                    or any(c not in "0123456789abcdef" for c in v)
                    for v in sources.values()
                )
            ):
                raise ValueError(f"{row['id']}: missing/invalid render source hashes")
            count, fingerprint = (
                row.get("excluded_fen_count"),
                row.get("excluded_fen_sha256"),
            )
            if (
                type(count) is not int
                or count < 0
                or not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in fingerprint)
            ):
                raise ValueError(f"{row['id']}: missing/invalid exclusion fingerprint")
            exclusion_snapshots[(count, fingerprint)] += 1
    prefix = a.out / name
    reserve_output(prefix)
    splits, total = {}, 0
    with open(f"{prefix}.jsonl", "x") as rows_out:
        for split_dir, rows in populations:
            clean, clean_rel, divergent, missing = [], [], 0, 0
            grid_errors, grid_invalid = [], 0
            for r in rows:
                png = split_dir / "images" / f"{r['id']}.png"
                md5, sha = digest(png)
                if sha != r.get("image_sha256"):
                    raise ValueError(
                        f"{split_dir.name}/{r['id']}: rendered image hash mismatch"
                    )
                if png_dimensions(png) != [a.width, a.height]:
                    raise ValueError(
                        f"{split_dir.name}/{r['id']}: unexpected image dimensions"
                    )
                # pixel error is NOT comparable across framings or elevations: a board that occupies
                # less of the frame trivially yields smaller pixel errors. Normalise by the board's own
                # mean projected edge length so the axes can be compared at all.
                gt = np.asarray(r["corners_px"], dtype=float)
                if grid_transfer_error(gt, gt) is None:
                    raise ValueError(
                        "ground-truth corners must be a finite nondegenerate cyclic convex quad"
                    )
                edge = float(np.linalg.norm(gt - np.roll(gt, 1, axis=0), axis=1).mean())
                if edge <= 0:
                    raise ValueError("degenerate ground-truth corners")
                error = r.get("detector_error_px")
                if error is not None and (not np.isfinite(error) or error < 0):
                    raise ValueError("detector error must be finite and nonnegative")
                rel = (
                    None
                    if r.get("detector_error_px") is None
                    else r["detector_error_px"] / edge
                )
                det = r.get("detector_corners_px")
                if det is None:
                    missing += 1
                    status = "not_found"
                else:
                    d = np.asarray(det, dtype=float)
                    if d.shape != (4, 2) or not np.isfinite(d).all() or error is None:
                        raise ValueError(
                            "detected quad must be finite 4x2 with a recorded error"
                        )
                    if (
                        (d[:, 0] < lo_x).any()
                        or (d[:, 0] > hi_x).any()
                        or (d[:, 1] < lo_y).any()
                        or (d[:, 1] > hi_y).any()
                    ):
                        divergent += 1
                        status = "divergent_vertex"
                    else:
                        clean.append(r["detector_error_px"])
                        clean_rel.append(rel)
                        status = "ok"
                geom = r.get("piece_geometry") or {}
                grid_error = grid_transfer_error(gt, det)
                if status == "ok":
                    if grid_error is None:
                        grid_invalid += 1
                    else:
                        grid_errors.append(grid_error)
                rows_out.write(
                    json.dumps(
                        {
                            **r,  # Preserve actual renderer/source hashes, palettes, camera and exclusion lineage.
                            "split": split_dir.name,
                            "id": r["id"],
                            "md5": md5,
                            "sha256": sha,
                            "image_size": [a.width, a.height],
                            "fen": r["fen"],
                            "piece_count": r["piece_count"],
                            "corners_px": r["corners_px"],
                            "a8_image_corner": r["a8_image_corner"],
                            "elevation_deg": round(r["camera"]["elevation_deg"], 4),
                            "azimuth_deg": round(r["camera"]["azimuth_deg"], 4),
                            "board_width_fraction": round(
                                r["camera"]["target_board_width_fraction"], 4
                            ),
                            "geometry_asset": geom.get("asset_identity")
                            or geom.get("source"),
                            "geometry_license": geom.get("license"),
                            "board_edge_px": round(edge, 2),
                            "detector_status": status,
                            "detector_error_px": r.get("detector_error_px"),
                            "detector_error_rel": None
                            if rel is None
                            else round(rel, 5),
                            "grid_transfer_error_squares": grid_error,
                        },
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                total += 1
            c = np.array(clean)
            cr = np.array(clean_rel)
            splits[split_dir.name] = dict(
                n=len(rows),
                detector_not_found=missing,
                detector_divergent_vertex=divergent,
                detector_usable=len(clean),
                detector_failure_rate_pct=round(
                    100 * (missing + divergent) / len(rows), 3
                ),
                error_population="conditional on detected quad within the stated canvas margin",
                grid_transfer_invalid_usable=grid_invalid,
                grid_transfer_population="usable quads with finite homography over the full 9x9 board grid",
                grid_transfer_median_squares=float(np.median(grid_errors))
                if grid_errors
                else None,
                grid_transfer_p90_squares=float(np.percentile(grid_errors, 90))
                if grid_errors
                else None,
                median_px=round(float(np.median(c)), 2) if clean else None,
                p90_px=round(float(np.percentile(c, 90)), 2) if clean else None,
                max_px=round(float(np.max(c)), 2) if clean else None,
                median_rel_pct=round(100 * float(np.median(cr)), 3) if clean else None,
                p90_rel_pct=round(100 * float(np.percentile(cr, 90)), 3)
                if clean
                else None,
                elevation_band=[
                    round(min(r["camera"]["elevation_deg"] for r in rows), 1),
                    round(max(r["camera"]["elevation_deg"] for r in rows), 1),
                ],
                geometry=(rows[0].get("piece_geometry") or {}).get("asset_identity")
                or (rows[0].get("piece_geometry") or {}).get("source"),
                license=(rows[0].get("piece_geometry") or {}).get("license"),
            )
            print(
                json.dumps(
                    dict(split=split_dir.name, **splits[split_dir.name]),
                    allow_nan=False,
                )
            )
    load_freeze(
        Path(f"{prefix}.jsonl"), allow_incomplete=True
    )  # Validate before committing the receipt.

    meta = dict(
        frozen=date,
        images=total,
        splits=splits,
        schema_version=2,
        freeze_tool_repo_commit=git_rev(a.repo),
        freeze_tool_sources_sha256={
            p.name: digest(p)[1]
            for p in (
                Path(__file__),
                Path(__file__).with_name("benchmark_io.py"),
                Path(__file__).with_name("fen_pools.py"),
            )
        },
        exclusion_snapshots=[
            dict(count=count, sha256=fingerprint, rows=n)
            for (count, fingerprint), n in sorted(exclusion_snapshots.items())
        ],
        image_size=[a.width, a.height],
        render_provenance="per-row source_sha256; freeze-tool checkout is NOT a renderer revision",
        release_ready=False,
        unchecked_release_gates=[
            "complete training-source exclusion audit including learned priors",
            "archived source/assets matching recorded hashes and license grants",
            "independent playing-surface corner fixtures",
        ],
        corner_ground_truth="projected playing-plane corners, rounded to 0.001px; slab bounds are a separate check",
        divergent_definition=f"any detected vertex beyond {MARGIN}x the frame outside the canvas",
    )
    with Path(f"{prefix}.json").open("x") as stream:
        json.dump(meta, stream, indent=1, allow_nan=False)
        stream.write("\n")
    finish_output(
        prefix,
        dict(
            status="freeze_complete",
            release_ready=False,
            images=total,
            splits=list(splits),
        ),
    )
    print(f"\n{total} images across {len(splits)} splits -> {prefix}.jsonl")


if __name__ == "__main__":
    main()
