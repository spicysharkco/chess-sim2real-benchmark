"""Frozen-population checks and no-clobber receipts shared by paper CLIs.

No ML imports: these gates run before loading a checkpoint or allocating CUDA.
An incomplete marker is deliberately retained after failure; use a new prefix.
"""

import gzip
import hashlib
import json
import math
import platform
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .fen_pools import placement, sym_key
except ImportError:  # Direct script invocation from tools/paper/.
    from fen_pools import placement, sym_key


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def png_dimensions(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"{path}: missing PNG dimensions")
    return list(struct.unpack(">II", header[16:24]))


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected nonempty JSONL objects")
    return rows


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{6}", value):
        raise ValueError(f"invalid image id: {value!r}")
    return value


def split_rows(directory):
    """Check one render population without trusting counts or completion logs."""
    directory = Path(directory)
    if (directory / "INCOMPLETE").exists():
        raise ValueError(f"{directory.name}: incomplete render")
    if not (directory / "metadata.jsonl").is_file():
        raise ValueError(f"{directory.name}: missing metadata.jsonl")
    rows = read_jsonl(directory / "metadata.jsonl")
    ids = [valid_id(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{directory.name}: duplicate metadata IDs")
    for child, suffix in (("images", ".png"), ("labels", ".txt")):
        actual = {path.stem for path in (directory / child).glob(f"*{suffix}")}
        if actual != set(ids):
            raise ValueError(f"{directory.name}: {child} IDs differ from metadata")
    for row in rows:
        row["fen"] = placement(row["fen"])
        label = placement((directory / "labels" / f"{row['id']}.txt").read_text())
        if label != row["fen"]:
            raise ValueError(f"{directory.name}/{row['id']}: label/metadata mismatch")
    return sorted(rows, key=lambda row: row["id"])


def load_freeze(path, *, allow_incomplete=False):
    prefix = Path(str(path).removesuffix(".gz")).with_suffix("")
    if not allow_incomplete and Path(f"{prefix}.incomplete").exists():
        raise ValueError(f"incomplete freeze: {prefix}")
    rows = read_jsonl(path)
    keys, positions = set(), set()
    for row in rows:
        split = row.get("split")
        if not isinstance(split, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", split
        ):
            raise ValueError(f"invalid split name: {split!r}")
        key = (split, valid_id(row.get("id")))
        if key in keys:
            raise ValueError(f"duplicate frozen key: {key}")
        keys.add(key)
        if row.get("a8_image_corner") not in ("tl", "tr", "br", "bl"):
            raise ValueError(f"{key}: invalid physical a8 corner index")
        quad = row.get("corners_px")
        if (
            not isinstance(quad, list)
            or len(quad) != 4
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in point
                )
                for point in quad
            )
        ):
            raise ValueError(f"{key}: invalid frozen corners")
        edges = [
            (quad[(i + 1) % 4][0] - quad[i][0], quad[(i + 1) % 4][1] - quad[i][1])
            for i in range(4)
        ]
        crosses = [
            edge[0] * edges[(i + 1) % 4][1] - edge[1] * edges[(i + 1) % 4][0]
            for i, edge in enumerate(edges)
        ]
        if not (all(c > 1e-6 for c in crosses) or all(c < -1e-6 for c in crosses)):
            raise ValueError(f"{key}: frozen corners are not a cyclic convex quad")
        row["fen"] = placement(row["fen"])
        position = sym_key(row["fen"])
        if position in positions:
            raise ValueError(f"duplicate exact/symmetry-equivalent frozen FEN: {key}")
        positions.add(position)
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))):
            raise ValueError(f"{key}: missing/invalid image SHA-256")
    return sorted(rows, key=lambda row: (row["split"], row["id"]))


def preflight(root, freeze, splits=None, limit=0):
    if limit < 0:
        raise ValueError("limit must be nonnegative (0 means all rows)")
    root, freeze = Path(root), Path(freeze)
    rows = load_freeze(freeze)
    all_splits = {row["split"] for row in rows}
    requested = all_splits if splits is None else set(splits)
    if not requested or requested - all_splits:
        raise ValueError(
            f"unknown/empty requested splits: {sorted(requested - all_splits)}"
        )
    if splits is not None and len(splits) != len(requested):
        raise ValueError("duplicate requested splits")
    if (root / "INCOMPLETE").exists():
        raise ValueError("incomplete benchmark root")
    selected = []
    for split in sorted(requested):
        expected = [row for row in rows if row["split"] == split]
        directory = root / split
        actual = {row["id"]: row for row in split_rows(directory)}
        if set(actual) != {row["id"] for row in expected}:
            raise ValueError(f"{split}: render IDs differ from frozen inventory")
        # Validate the entire selected split even when inference will be limited.
        for row in expected:
            key = f"{split}/{row['id']}"
            if actual[row["id"]]["fen"] != row["fen"]:
                raise ValueError(f"{key}: frozen FEN differs from render labels")
            image = directory / "images" / f"{row['id']}.png"
            digest = sha256(image)
            if digest != row["sha256"]:
                raise ValueError(f"{key}: frozen image SHA-256 mismatch")
            recorded = actual[row["id"]].get("image_sha256")
            if recorded is not None and digest != recorded:
                raise ValueError(f"{key}: metadata image SHA-256 mismatch")
            if "image_size" in row and png_dimensions(image) != row["image_size"]:
                raise ValueError(f"{key}: frozen image dimensions mismatch")
            for field in ("corners_px", "a8_image_corner"):
                if field in row and row[field] != actual[row["id"]].get(field):
                    raise ValueError(f"{key}: frozen {field} differs from metadata")
        selected.extend(expected[:limit] if limit else expected)
    return dict(
        rows=selected,
        freeze=str(freeze.resolve()),
        freeze_sha256=sha256(freeze),
        frozen_n=len(rows),
        selected_n=len(selected),
        splits=sorted(requested),
        limit=limit,
        benchmark_complete=len(selected) == len(rows),
        selection_sha256=hashlib.sha256(
            json.dumps(
                [(r["split"], r["id"], r["sha256"]) for r in selected],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )


def reserve_output(prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if prefix.exists() or any(
        p.name.startswith(prefix.name + ".") for p in prefix.parent.iterdir()
    ):
        raise FileExistsError(
            f"output prefix already used: {prefix}; choose a new prefix"
        )
    with Path(f"{prefix}.incomplete").open("x") as stream:
        stream.write(
            "Not a completed result. Retain on failure; do not reuse this prefix.\n"
        )


def verify_unchanged(root, inventory, model_inputs):
    """Recheck inputs before publication; a concurrent copy/edit invalidates the run."""
    current = preflight(
        root, inventory["freeze"], inventory["splits"], inventory["limit"]
    )
    if any(
        current[key] != inventory[key] for key in ("freeze_sha256", "selection_sha256")
    ):
        raise ValueError("frozen population changed during inference")
    if any(sha256(path) != digest for path, digest in model_inputs.items()):
        raise ValueError("model/source inputs changed during inference")


def finish_output(prefix, receipt):
    prefix = Path(prefix)
    if not Path(f"{prefix}.incomplete").is_file():
        raise ValueError("cannot complete an unreserved output prefix")
    artifacts = {
        p.name: sha256(p)
        for p in sorted(prefix.parent.iterdir())
        if p.name.startswith(prefix.name + ".")
        and p.suffix != ".incomplete"
        and p.is_file()
    }
    if not artifacts:
        raise ValueError("cannot complete an empty result")
    environment = dict(python=platform.python_version(), host=platform.node())
    for name in ("numpy", "cv2", "torch", "torchvision"):
        module = sys.modules.get(name)
        if module is not None:
            environment[name] = getattr(module, "__version__", "unknown")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        environment.update(
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
        )
    with Path(f"{prefix}.complete.json").open("x") as stream:
        json.dump(
            dict(
                receipt,
                artifacts_sha256=artifacts,
                environment=environment,
                completed_at=datetime.now(timezone.utc).isoformat(),
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    Path(f"{prefix}.incomplete").unlink()
