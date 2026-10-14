#!/usr/bin/env python3
"""Annotate rendered metadata with a8's detected-board image corner.

This reproduces the Three.js camera projection and then calls the repository's
corner ordering function.  It does not infer orientation from azimuth alone.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import board  # noqa: E402


WIDTH = 1920
HEIGHT = 1080
FOV_DEGREES = 35.0
TARGET = np.array([0.0, 0.28, 0.0], dtype=np.float64)
CANONICAL_NAMES = ("a8", "h8", "h1", "a1")
CANONICAL_WORLD = np.array(
    [[-4.0, 0.0, -4.0], [4.0, 0.0, -4.0], [4.0, 0.0, 4.0], [-4.0, 0.0, 4.0]],
    dtype=np.float64,
)
BOUND_EDGE = 4.003
BOUND_WORLD = np.array(
    [
        [x, y, z]
        for y in (0.0, -0.10)
        for x in (-BOUND_EDGE, BOUND_EDGE)
        for z in (-BOUND_EDGE, BOUND_EDGE)
    ],
    dtype=np.float64,
)
CORNER_NAMES = ("tl", "tr", "br", "bl")


def camera_axes(camera, renderer):
    azimuth = math.radians(camera["azimuth_deg"])
    elevation = math.radians(camera["elevation_deg"])
    roll = math.radians(renderer["actual_roll_deg"])
    distance = renderer["distance"]
    horizontal = distance * math.cos(elevation)
    position = np.array(
        [
            horizontal * math.sin(azimuth),
            distance * math.sin(elevation),
            horizontal * math.cos(azimuth),
        ],
        dtype=np.float64,
    )
    backward = position - TARGET
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 1.0, 0.0]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    # Object3D.rotateZ applies a local-axis post-rotation to the lookAt basis.
    rolled_right = right * math.cos(roll) + up * math.sin(roll)
    rolled_up = -right * math.sin(roll) + up * math.cos(roll)
    return position, rolled_right, rolled_up, backward


def project(points, camera, renderer):
    position, right, up, backward = camera_axes(camera, renderer)
    relative = np.asarray(points, dtype=np.float64) - position
    camera_x = relative @ right
    camera_y = relative @ up
    camera_z = relative @ backward
    tangent = math.tan(math.radians(FOV_DEGREES) / 2.0)
    ndc_x = camera_x / (-camera_z * tangent * (WIDTH / HEIGHT))
    ndc_y = camera_y / (-camera_z * tangent)
    offset_y = renderer.get("image_offset_y_px", 0.0)
    if isinstance(offset_y, bool) or not isinstance(offset_y, (int, float)) or not math.isfinite(offset_y) or abs(offset_y) > HEIGHT / 2:
        raise ValueError("invalid recorded vertical projection offset")
    return np.column_stack(((ndc_x + 1.0) * WIDTH / 2.0, (1.0 - ndc_y) * HEIGHT / 2.0 + offset_y))


def orientation_for(row):
    canonical_pixels = project(CANONICAL_WORLD, row["camera"], row["renderer"])
    ordered = board.order_corners(canonical_pixels)
    distances = np.linalg.norm(ordered - canonical_pixels[0], axis=1)
    a8_index = int(np.argmin(distances))
    if distances[a8_index] > 1e-3:
        raise ValueError(f"projected a8 did not survive corner ordering: {row['id']}")

    bounds_pixels = project(BOUND_WORLD, row["camera"], row["renderer"])
    calculated = {
        "left": float(bounds_pixels[:, 0].min() / WIDTH),
        "top": float(bounds_pixels[:, 1].min() / HEIGHT),
        "right": float(bounds_pixels[:, 0].max() / WIDTH),
        "bottom": float(bounds_pixels[:, 1].max() / HEIGHT),
    }
    stored = row["renderer"]["projected_board_bounds"]
    bounds_error = max(abs(calculated[key] - stored[key]) for key in calculated)
    if bounds_error > 1e-9:
        raise ValueError(
            f"Three.js projection mismatch for {row['id']}: max bounds error {bounds_error}"
        )
    return CORNER_NAMES[a8_index], canonical_pixels, bounds_error


def align_projected_to_detected(canonical_pixels, detected):
    """Map physical a8 to the detector's cyclic start vertex.

    `board.order_corners` is cyclic, but its x+y start extremum can differ
    between an exact projection and a slightly biased detected quadrilateral.
    Choose the unique cyclic correspondence instead of assuming both arrays
    selected the same start vertex.
    """
    projected_ordered = board.order_corners(canonical_pixels)
    detected = np.asarray(detected, dtype=np.float32)
    if detected.shape != (4, 2) or not np.isfinite(detected).all():
        raise ValueError("detector did not return four finite board corners")
    candidates = []
    for shift in range(4):
        aligned = np.roll(projected_ordered, shift, axis=0)
        errors = np.linalg.norm(detected - aligned, axis=1)
        candidates.append((float(errors.mean()), shift, errors, aligned))
    candidates.sort(key=lambda item: item[0])
    best, second = candidates[:2]
    edges = np.linalg.norm(
        projected_ordered - np.roll(projected_ordered, 1, axis=0), axis=1
    )
    minimum_edge = float(edges.min())
    if best[0] > max(25.0, 0.45 * minimum_edge):
        raise ValueError(
            f"detector/projected correspondence error {best[0]:.2f}px is too large"
        )
    if second[0] - best[0] < max(5.0, 0.08 * minimum_edge):
        raise ValueError(
            "detector/projected cyclic correspondence is ambiguous: "
            f"best={best[0]:.2f}px second={second[0]:.2f}px"
        )
    a8_distances = np.linalg.norm(best[3] - canonical_pixels[0], axis=1)
    a8_index = int(np.argmin(a8_distances))
    if a8_distances[a8_index] > 1e-3:
        raise ValueError("aligned projection lost the physical a8 vertex")
    return {
        "a8_image_corner": CORNER_NAMES[a8_index],
        "cyclic_shift": best[1],
        "corner_errors_px": best[2].tolist(),
        "mean_corner_error_px": best[0],
        "second_best_mean_error_px": second[0],
        "minimum_projected_edge_px": minimum_edge,
        "detected_corners_px": detected.tolist(),
    }


def annotate(path):
    original = path.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    if not rows:
        raise ValueError(f"empty metadata: {path}")
    board_hash = hashlib.sha256((ROOT / "src/board.py").read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for row in rows:
        projected_a8, projected, bounds_error = orientation_for(row)
        image_path = path.parent / "images" / f"{row['id']}.png"
        image_bytes = image_path.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if image_hash != row["image_sha256"]:
            raise ValueError(f"image hash differs from metadata for {row['id']}")
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode image for orientation mapping: {row['id']}")
        detected = board.find_corners(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if detected is None:
            raise ValueError(f"detector found no corners for orientation mapping: {row['id']}")
        correspondence = align_projected_to_detected(projected, detected)
        old_annotation = row.get("projected_orientation_annotation") or row.get(
            "orientation_annotation"
        )
        if old_annotation:
            row["projected_orientation_annotation"] = old_annotation
        row["projected_a8_image_corner"] = projected_a8
        row["a8_image_corner"] = correspondence["a8_image_corner"]
        row["orientation_annotation"] = {
            "method": "threejs-camera-projection+src.board.order_corners",
            "mapping": "unique-cyclic-correspondence-to-detected-image-ordered-quad",
            "canonical_world_corners": dict(zip(CANONICAL_NAMES, CANONICAL_WORLD.tolist())),
            "projected_canonical_corners_px": dict(zip(CANONICAL_NAMES, projected.tolist())),
            "camera_target": TARGET.tolist(),
            "image_offset_y_px": row["renderer"].get("image_offset_y_px", 0.0),
            "vertical_fov_deg": FOV_DEGREES,
            "image_size": [WIDTH, HEIGHT],
            "src_board_py_sha256": board_hash,
            "annotation_script_sha256": script_hash,
            "detector_input_image_sha256": image_hash,
            "stored_bounds_max_abs_error": bounds_error,
            **correspondence,
        }
    encoded = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    manifest = b"".join(
        (
            json.dumps(
                {
                    "id": row["id"],
                    "a8_image_corner": row["a8_image_corner"],
                    "projected_a8_image_corner": row["projected_a8_image_corner"],
                    "orientation_annotation": row["orientation_annotation"],
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        for row in rows
    )
    manifest_path = path.parent / "manifest.jsonl"
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_temporary.write_bytes(manifest)
    manifest_temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "input_sha256": hashlib.sha256(original).hexdigest(),
                "output_sha256": hashlib.sha256(encoded).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "a8_corner_counts": {
                    name: sum(row["a8_image_corner"] == name for row in rows)
                    for name in CORNER_NAMES
                },
                "maximum_projection_bounds_error": max(
                    row["orientation_annotation"]["stored_bounds_max_abs_error"]
                    for row in rows
                ),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/synth/angles", help="render output directory"
    )
    args = parser.parse_args()
    annotate(args.data / "metadata.jsonl")


if __name__ == "__main__":
    main()
