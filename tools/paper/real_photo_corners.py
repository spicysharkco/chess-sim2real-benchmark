#!/usr/bin/env python3
"""Click the four board corners on each real photograph, and check the result.

Corner order and names follow the render manifests (`canonical_corners_px` in
rendering/benchmark_v2.js): a8, h8, h1, a1, as the *sheet* labels them for that photograph. For
the rotated-board (a1_light) frames the physical board has turned, but the squares keep the
sheet's algebraic names, so click the corner next to the sheet's a8 square first, then h8, h1, a1.

  uv run --with matplotlib --with pillow python tools/paper/real_photo_corners.py annotate \\
      --photos <dir of jpg> --out <dir>/corners.jsonl [--redo]
  uv run --with pillow python tools/paper/real_photo_corners.py check \\
      --photos <dir of jpg> --corners <dir>/corners.jsonl

`annotate` skips photographs that already have a row, so a session can be resumed. Press a key
other than a click to skip the current photograph (it is recorded as `skipped` and fails `check`).
`check` exits 3 unless every decodable photograph has one hash-matched, finite, convex,
in-bounds quad. Recorded dimensions must match the decoded image. A skipped row may be superseded
when annotation resumes; duplicate completed rows are ambiguous and fail the check.
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import sys
from pathlib import Path

CORNER_NAMES = ("a8", "h8", "h1", "a1")
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic")


def list_photos(directory):
    directory = Path(directory).expanduser()
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def validate_quad(points, width, height):
    """Return None if the quad is usable, otherwise the reason it is not."""
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or width <= 0
        or height <= 0
    ):
        return "image dimensions must be positive integers"
    if not isinstance(points, (list, tuple)):
        return "corners must be a sequence"
    if len(points) != 4:
        return f"expected 4 points, got {len(points)}"
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return "each corner must be a two-number coordinate pair"
        x, y = point
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return "non-numeric coordinate"
        if isinstance(x, bool) or isinstance(y, bool):
            return "non-numeric coordinate"
        if x != x or y != y or abs(x) == float("inf") or abs(y) == float("inf"):  # noqa: PLR0124 - avoid integer-to-float coercion
            return "non-finite coordinate"
        if not (0 <= x < width and 0 <= y < height):
            return f"point ({x:.1f}, {y:.1f}) outside the {width}x{height} image"
    # every consecutive cross product must have the same sign: convex, non-degenerate
    signs = []
    for index in range(4):
        ax, ay = points[index]
        bx, by = points[(index + 1) % 4]
        cx, cy = points[(index + 2) % 4]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) < 1e-6:
            return "degenerate quad (collinear corners)"
        signs.append(cross > 0)
    if len(set(signs)) != 1:
        return "quad is not convex (corners clicked out of order?)"
    return None


def read_rows(path, raw_bytes=None):
    path = Path(path).expanduser()
    rows = {}
    problems = []
    if raw_bytes is None and not path.exists():
        return rows, problems
    if raw_bytes is None:
        raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return rows, [f"{path.name}: annotation file is not valid UTF-8: {exc}"]
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}:{line_number}: malformed JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            problems.append(f"{path.name}:{line_number}: row must be an object")
            continue
        name = row.get("file")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "\\" in name
        ):
            problems.append(f"{path.name}:{line_number}: invalid file name {name!r}")
            continue
        if name in rows and not rows[name].get("skipped"):
            problems.append(
                f"{path.name}:{line_number}: duplicate completed row for {name}"
            )
        rows[name] = row
    return rows, problems


def load_rows(path):
    """Load resumable annotation rows; malformed or ambiguous history is fatal."""
    rows, problems = read_rows(path)
    if problems:
        raise SystemExit("\n".join(problems))
    return rows


def decoded_size(data):
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.size


def refuse_annotation_alias(out_path, photos):
    output = Path(out_path).expanduser()
    for photo in photos:
        same_resolved = output.resolve(strict=False) == photo.resolve(strict=False)
        same_inode = (
            output.exists() and photo.exists() and os.path.samefile(output, photo)
        )
        if same_resolved or same_inode:
            raise SystemExit(
                f"annotation output {out_path} aliases a photograph; refusing to overwrite it"
            )


def check(photos_dir, corners_path):
    try:
        import PIL  # noqa: F401
    except ImportError:
        return {
            "photographs": 0,
            "annotated": 0,
            "photograph_sha256": {},
            "corners_sha256": None,
            "validation_scope": "structural_only",
            "problems": ["Pillow is required to decode photographs"],
            "clean": False,
        }
    try:
        photos = list_photos(photos_dir)
    except OSError as exc:
        return {
            "photographs": 0,
            "annotated": 0,
            "photograph_sha256": {},
            "corners_sha256": None,
            "validation_scope": "structural_only",
            "problems": [str(exc)],
            "clean": False,
        }
    corners_path = Path(corners_path).expanduser()
    try:
        corners_bytes = corners_path.read_bytes()
    except FileNotFoundError:
        corners_bytes = None
    corners_sha256 = sha256_bytes(corners_bytes) if corners_bytes is not None else None
    rows, problems = read_rows(corners_path, corners_bytes)
    photo_hashes = {}
    if not photos:
        problems.append(f"no photographs in {photos_dir}")
    for photo in photos:
        if photo.is_symlink() or not photo.is_file():
            problems.append(
                f"{photo.name}: photograph is not a regular non-symlink file"
            )
            continue
        try:
            photo_bytes = photo.read_bytes()
        except OSError as exc:
            problems.append(f"{photo.name}: cannot read photograph: {exc}")
            continue
        photo_sha256 = sha256_bytes(photo_bytes)
        photo_hashes[photo.name] = photo_sha256
        row = rows.get(photo.name)
        if row is None:
            problems.append(f"{photo.name}: no corner row")
            continue
        if row.get("skipped"):
            problems.append(f"{photo.name}: skipped")
            continue
        if row.get("sha256") != photo_sha256:
            problems.append(f"{photo.name}: file changed since annotation")
            continue
        corners = row.get("canonical_corners_px")
        if not isinstance(corners, dict):
            problems.append(f"{photo.name}: canonical_corners_px is not an object")
            continue
        if set(corners) != set(CORNER_NAMES):
            problems.append(
                f"{photo.name}: corner names are {sorted(map(str, corners))}"
            )
            continue
        recorded_size = row.get("image_size")
        if (
            not isinstance(recorded_size, (list, tuple))
            or len(recorded_size) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in recorded_size
            )
        ):
            problems.append(f"{photo.name}: recorded image_size is invalid")
            continue
        try:
            width, height = decoded_size(photo_bytes)
        except (OSError, ValueError) as exc:
            problems.append(f"{photo.name}: cannot decode photograph: {exc}")
            continue
        if list(recorded_size) != [width, height]:
            problems.append(
                f"{photo.name}: recorded image_size {list(recorded_size)} does not match decoded "
                f"size {[width, height]}"
            )
            continue
        points = [corners[name] for name in CORNER_NAMES]
        reason = validate_quad(points, width, height)
        if reason:
            problems.append(f"{photo.name}: {reason}")
            continue
        if sha256_of(photo) != photo_sha256:
            problems.append(f"{photo.name}: file changed during corner check")
    photo_names = {photo.name for photo in photos}
    for name in rows:
        if name not in photo_names:
            problems.append(f"{name}: row without a photograph")
    if corners_bytes is not None:
        try:
            corners_stable = sha256_of(corners_path) == corners_sha256
        except OSError:
            corners_stable = False
        if not corners_stable:
            problems.append(
                f"{corners_path.name}: annotation file changed during corner check"
            )
    return {
        "photographs": len(photos),
        "annotated": sum(
            1 for p in photos if p.name in rows and not rows[p.name].get("skipped")
        ),
        "photograph_sha256": photo_hashes,
        "corners_sha256": corners_sha256,
        "validation_scope": "structural_only",
        "problems": problems,
        "clean": not problems,
    }


def annotate(photos_dir, out_path, redo, annotator):
    out_path = Path(out_path).expanduser()
    photos = list_photos(photos_dir)
    if not photos:
        raise SystemExit(f"no photographs in {photos_dir}")
    refuse_annotation_alias(out_path, photos)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for corner annotation") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required for corner annotation") from exc

    rows = {} if redo else load_rows(out_path)
    todo = [p for p in photos if p.name not in rows or rows[p.name].get("skipped")]
    print(f"{len(photos)} photographs, {len(todo)} to annotate", file=sys.stderr)
    with out_path.open("a" if not redo else "w") as handle:
        for photo in todo:
            if photo.is_symlink() or not photo.is_file():
                raise SystemExit(
                    f"{photo}: photograph is not a regular non-symlink file"
                )
            photo_bytes = photo.read_bytes()
            photo_sha256 = sha256_bytes(photo_bytes)
            try:
                image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
                image.load()
            except (OSError, ValueError) as exc:
                raise SystemExit(f"{photo}: cannot decode photograph: {exc}") from exc
            width, height = image.size
            figure, axis = plt.subplots(figsize=(12, 8))
            axis.imshow(image)
            axis.set_title(
                f"{photo.name}: click a8, h8, h1, a1 (sheet names); any key skips"
            )
            axis.set_axis_off()
            clicks = plt.ginput(4, timeout=0, show_clicks=True)
            plt.close(figure)
            row = {
                "file": photo.name,
                "sha256": photo_sha256,
                "image_size": [width, height],
                "annotator": annotator,
                "annotated_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
            if sha256_of(photo) != photo_sha256:
                row["skipped"] = True
                row["reason"] = "file changed during annotation"
                print(
                    f"{photo.name}: file changed during annotation; skipped",
                    file=sys.stderr,
                )
            elif len(clicks) < 4:
                row["skipped"] = True
                print(f"{photo.name}: skipped", file=sys.stderr)
            else:
                points = [(float(x), float(y)) for x, y in clicks]
                reason = validate_quad(points, width, height)
                if reason:
                    row["skipped"] = True
                    row["reason"] = reason
                    print(f"{photo.name}: {reason}; re-run to redo it", file=sys.stderr)
                else:
                    row["canonical_corners_px"] = {
                        name: [round(x, 1), round(y, 1)]
                        for name, (x, y) in zip(CORNER_NAMES, points)
                    }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("annotate")
    a.add_argument("--photos", required=True)
    a.add_argument("--out", required=True)
    a.add_argument(
        "--redo", action="store_true", help="discard existing rows and start over"
    )
    a.add_argument("--annotator", default="author")
    c = sub.add_parser("check")
    c.add_argument("--photos", required=True)
    c.add_argument("--corners", required=True)
    args = parser.parse_args()
    if args.command == "annotate":
        annotate(args.photos, args.out, args.redo, args.annotator)
        report = check(args.photos, args.out)
    else:
        report = check(args.photos, args.corners)
    print(json.dumps(report, indent=2))
    return 0 if report["clean"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
