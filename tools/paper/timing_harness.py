#!/usr/bin/env python3
"""C4 preregistered end-to-end latency harness (one model per process).

CPU preflight validates the complete C1 development inventory and the complete
C2 matrix before any model import or CUDA allocation.  There is intentionally no
partition argument and no final-test override.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.paper import lora_curve
from tools.paper.benchmark_io import (
    finish_output,
    png_dimensions,
    read_jsonl,
    reserve_output,
    sha256,
)
from tools.paper.fen_pools import placement
from tools.paper.transfer_cohort import verify_receipt

SELECTION_SEED = 20260914
SCENES = 250
TIMED_IMAGES = 500
WARMUP_IMAGES = 30
POWER_LIMIT_W = 270.0
PARITIES = ("a1_dark", "a1_light")
C2_FROZEN_LORA_SHA256 = (
    "[redacted: withheld digest]"
)
PROTOCOL_PATH = ROOT / "docs/TIMING_PROTOCOL_2026-09-10.md"
PRIVATE_VARIANTS = {
    "private_full": "full",
    "private_raw": "raw",
    "private_warp_high": "warp_high",
    "private_warp_low": "warp_low",
}


def c2_alias(budget: int, arm: str, seed: int) -> str:
    return f"lora_{arm}_n{budget}_s{seed}"


C2_ALIASES = tuple(
    c2_alias(run.budget, run.arm, run.seed) for run in lora_curve.registered_runs(False)
)
MODEL_ALIASES = ("cq_release", *PRIVATE_VARIANTS, *C2_ALIASES)


def _write_json(path: Path, value: Mapping) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _paired_scenes(rows: Sequence[Mapping]) -> dict[tuple[str, str], list[Mapping]]:
    scenes: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        if row.get("partition") != "dev":
            raise ValueError("timing population may contain only C1 dev rows")
        key = (str(row.get("position_group")), str(row.get("factor_cell")))
        if not all(key) or row.get("parity") not in PARITIES:
            raise ValueError("invalid timing scene identity")
        scenes[key].append(row)
    for key, pair in scenes.items():
        if len(pair) != 2 or {row["parity"] for row in pair} != set(PARITIES):
            raise ValueError(f"scene lacks one parity mate: {key}")
    return scenes


def select_timing_rows(
    rows: Sequence[Mapping], *, scenes: int = SCENES, seed: int = SELECTION_SEED
) -> tuple[list[Mapping], dict]:
    """Select balanced factor-cell scenes, always retaining both parity mates."""
    if scenes != SCENES or seed != SELECTION_SEED:
        raise ValueError("timing scene count and selection seed are preregistered")
    paired = _paired_scenes(rows)
    by_cell: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in paired:
        by_cell[key[1]].append(key)
    cells = sorted(by_cell)
    if not cells:
        raise ValueError("empty C1 development population")
    rng = random.Random(seed)
    extras = list(cells)
    rng.shuffle(extras)
    base, remainder = divmod(scenes, len(cells))
    quotas = {cell: base + int(cell in set(extras[:remainder])) for cell in cells}
    chosen = []
    for cell in cells:
        candidates = sorted(by_cell[cell])
        rng.shuffle(candidates)
        if len(candidates) < quotas[cell]:
            raise ValueError(f"factor cell {cell} cannot meet its timing quota")
        chosen.extend(candidates[: quotas[cell]])
    if len(chosen) != scenes or len(set(chosen)) != scenes:
        raise ValueError("timing scene selection count/uniqueness mismatch")
    selected = [
        row for key in chosen for row in sorted(paired[key], key=lambda r: r["parity"])
    ]
    groups = Counter(key[0] for key in chosen)
    summary = {
        "selection_seed": seed,
        "scenes": scenes,
        "images": len(selected),
        "factor_cell_scenes": dict(sorted(Counter(key[1] for key in chosen).items())),
        "unique_position_groups": len(groups),
        "repeated_position_groups": sum(count - 1 for count in groups.values()),
        "max_scenes_per_position_group": max(groups.values()),
    }
    if (
        len(selected) != TIMED_IMAGES
        or max(summary["factor_cell_scenes"].values())
        - min(summary["factor_cell_scenes"].values())
        > 1
    ):
        raise ValueError(
            "timing selection is not the registered balanced 500-image set"
        )
    return selected, summary


def _c1_receipt(root: Path) -> tuple[dict, Path]:
    path = root / "benchmark.complete.json"
    receipt = verify_receipt(path)
    if (
        receipt.get("status") != "schema_v2_frozen"
        or receipt.get("benchmark_schema") != 2
        or receipt.get("final_test_sealed") is not True
        or receipt.get("images") != 20_000
    ):
        raise ValueError("expected complete sealed 20,000-image C1 receipt")
    return receipt, path


def validate_dev_inventory(root: Path) -> tuple[list[dict], str]:
    """Hash every dev image/label without reading a final-test image."""
    from tools.paper.benchmark_v2 import load_dev

    root = Path(root).resolve()
    _c1_receipt(root)
    rows = load_dev(root / "benchmark.jsonl")
    if len(rows) != 16_000 or any(row.get("partition") != "dev" for row in rows):
        raise ValueError("expected exactly 16,000 C1 development rows")
    digest_parts = []
    for row in rows:
        image, label = _resolve(root, row["image"]), _resolve(root, row["label"])
        image_hash, label_hash = sha256(image), sha256(label)
        if image_hash != row.get("sha256") or label_hash != row.get("label_sha256"):
            raise ValueError(f"C1 development file hash mismatch: {row.get('id')}")
        if png_dimensions(image) != [1920, 1080]:
            raise ValueError(
                f"C1 development image dimensions changed: {row.get('id')}"
            )
        digest_parts.append((row["id"], image_hash, label_hash))
    import hashlib

    digest = hashlib.sha256(
        json.dumps(digest_parts, separators=(",", ":")).encode()
    ).hexdigest()
    return rows, digest


def prepare_selection(c1_root: Path, out: Path) -> None:
    rows, inventory_hash = validate_dev_inventory(c1_root)
    selected, summary = select_timing_rows(rows)
    c1_receipt, c1_receipt_path = _c1_receipt(Path(c1_root).resolve())
    reserve_output(out)
    manifest = []
    for order, row in enumerate(selected):
        manifest.append(
            {
                "order": order,
                "id": row["id"],
                "scene_id": f"{row['position_group']}/{row['factor_cell']}",
                "position_group": row["position_group"],
                "factor_cell": row["factor_cell"],
                "parity": row["parity"],
                "partition": "dev",
                "fen": placement(row["fen"]),
                "image": str(_resolve(Path(c1_root).resolve(), row["image"])),
                "image_sha256": row["sha256"],
                "label": str(_resolve(Path(c1_root).resolve(), row["label"])),
                "label_sha256": row["label_sha256"],
            }
        )
    _write_jsonl(Path(f"{out}.jsonl"), manifest)
    _write_json(Path(f"{out}.summary.json"), summary)
    finish_output(
        out,
        {
            "status": "timing_selection_locked",
            "release_ready": False,
            "c1_root": str(Path(c1_root).resolve()),
            "c1_receipt_sha256": sha256(c1_receipt_path),
            "c1_design_receipt_sha256": c1_receipt["design_receipt_sha256"],
            "dev_inventory_sha256": inventory_hash,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            **summary,
        },
    )


def load_selection(path: Path, c1_root: Path) -> tuple[list[dict], dict, str]:
    path = Path(path).resolve()
    if path.suffix != ".jsonl":
        raise ValueError("selection must be a timing .jsonl manifest")
    prefix = path.with_suffix("")
    receipt = verify_receipt(Path(f"{prefix}.complete.json"))
    rows, inventory_hash = validate_dev_inventory(c1_root)
    c1_by_id = {row["id"]: row for row in rows}
    c1_receipt_path = Path(c1_root).resolve() / "benchmark.complete.json"
    if (
        receipt.get("status") != "timing_selection_locked"
        or receipt.get("selection_seed") != SELECTION_SEED
        or receipt.get("scenes") != SCENES
        or receipt.get("images") != TIMED_IMAGES
        or receipt.get("c1_root") != str(Path(c1_root).resolve())
        or receipt.get("c1_receipt_sha256") != sha256(c1_receipt_path)
        or receipt.get("dev_inventory_sha256") != inventory_hash
        or receipt.get("protocol_sha256") != sha256(PROTOCOL_PATH)
        or not isinstance(inventory_hash, str)
        or len(inventory_hash) != 64
    ):
        raise ValueError("timing selection receipt is stale or invalid")
    selected = read_jsonl(path)
    if len(selected) != TIMED_IMAGES or [row.get("order") for row in selected] != list(
        range(TIMED_IMAGES)
    ):
        raise ValueError("timing manifest order/count mismatch")
    _paired_scenes(selected)
    for row in selected:
        original = c1_by_id.get(row["id"])
        if original is None or any(
            row[field] != original[field]
            for field in ("position_group", "factor_cell", "parity")
        ):
            raise ValueError("timing manifest differs from current C1 metadata")
        if row["fen"] != placement(original["fen"]):
            raise ValueError("timing manifest ground truth differs from C1")
        if (
            row["image"] != str(_resolve(Path(c1_root).resolve(), original["image"]))
            or row["image_sha256"] != original["sha256"]
            or row["label"] != str(_resolve(Path(c1_root).resolve(), original["label"]))
            or row["label_sha256"] != original["label_sha256"]
        ):
            raise ValueError("timing manifest file identity differs from C1")
        for field in ("image", "label"):
            if sha256(row[field]) != row[f"{field}_sha256"]:
                raise ValueError(f"timing {field} changed")
    return selected, receipt, inventory_hash


def verify_selected_inputs(selected: Sequence[Mapping]) -> None:
    for row in selected:
        for field in ("image", "label"):
            if sha256(row[field]) != row[f"{field}_sha256"]:
                raise ValueError(f"timing {field} changed")


def verify_full_dev_unchanged(c1_root: Path, expected_sha256: str) -> None:
    _, current = validate_dev_inventory(c1_root)
    if current != expected_sha256:
        raise ValueError("complete C1 development inventory changed during timing")


def _find_digest(receipt: Mapping, suffix: str) -> str:
    matches = [
        value
        for key, value in receipt.get("model_inputs_sha256", {}).items()
        if key.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"C2 receipt lacks one bound {suffix}")
    return matches[0]


def validate_c2_matrix(
    matrix_root: Path, baseline_receipt_path: Path, c1_root: Path | None = None
) -> dict[str, dict]:
    """Require all ten completed train points, including completed negative points."""
    baseline_receipt_path = Path(baseline_receipt_path).resolve()
    baseline = verify_receipt(baseline_receipt_path)
    baseline_predictions = baseline_receipt_path.with_name(
        baseline_receipt_path.name.removesuffix(".complete.json")
        + ".dev_predictions.jsonl"
    )
    if (
        baseline.get("status") != "c2_complete"
        or baseline.get("mode") != "baseline"
        or baseline.get("accepted") is not True
        or baseline.get("final_test_images_read") is not False
        or not baseline_predictions.is_file()
        or _find_digest(baseline, "/source/tools/paper/lora_curve.py")
        != C2_FROZEN_LORA_SHA256
    ):
        raise ValueError("invalid C2 release baseline receipt")
    if c1_root is not None:
        c1_receipt_path = Path(c1_root).resolve() / "benchmark.complete.json"
        if baseline.get("c1_root") != str(Path(c1_root).resolve()) or _find_digest(
            baseline, str(c1_receipt_path)
        ) != sha256(c1_receipt_path):
            raise ValueError(
                "C2 release baseline is not bound to the current C1 receipt"
            )
    expected = {
        (run.budget, run.arm, run.seed): c2_alias(run.budget, run.arm, run.seed)
        for run in lora_curve.registered_runs(False)
    }
    matrix_root = Path(matrix_root).resolve()
    exit_path = matrix_root / "matrix_exit.txt"
    if not exit_path.is_file() or exit_path.read_text().strip() != "0":
        raise ValueError("C2 matrix controller lacks an exit-0 record")
    matrix_path = matrix_root / "matrix.complete.json"
    matrix = verify_receipt(matrix_path)
    expected_names = {key: f"b{key[0]}_{key[1]}_s{key[2]}" for key in expected}
    listed = {row.get("run"): row for row in matrix.get("runs", [])}
    if (
        matrix.get("status") != "ten_fixed_cells_complete"
        or matrix.get("reference_receipt_sha256") != sha256(baseline_receipt_path)
        or len(matrix.get("runs", [])) != 10
        or len(listed) != 10
        or [row.get("run") for row in matrix.get("runs", [])]
        != list(expected_names.values())
        or (matrix_root / "matrix.incomplete").exists()
    ):
        raise ValueError("C2 matrix lacks its exact successful terminal receipt")
    found = {}
    for key, alias in expected.items():
        name = expected_names[key]
        path = matrix_root / "results" / f"{name}.complete.json"
        receipt = verify_receipt(path)
        spec = receipt.get("run_spec") or {}
        if (
            receipt.get("mode") != "train"
            or receipt.get("status") != "c2_complete"
            or spec.get("train_head") is not False
            or (spec.get("budget"), spec.get("arm"), spec.get("seed")) != key
            or receipt.get("recipe") != lora_curve.RECIPE
            or receipt.get("base_checkpoint_sha256")
            != lora_curve.BASE_CHECKPOINT_SHA256
            or receipt.get("chessqueries_revision") != lora_curve.CQ_REVISION
            or receipt.get("c1_root") != baseline.get("c1_root")
            or receipt.get("final_test_images_read") is not False
            or receipt.get("scored_partitions") != ["dev", "chessred_test"]
            or len(receipt.get("adapter_tensor_names", []))
            != lora_curve.VIT_L_LORA_TENSORS
            or _find_digest(receipt, "/source/tools/paper/lora_curve.py")
            != C2_FROZEN_LORA_SHA256
        ):
            raise ValueError(f"invalid completed C2 point receipt: {path}")
        if listed[name].get("receipt_sha256") != sha256(path):
            raise ValueError(
                f"matrix listing digest differs from point receipt: {name}"
            )
        if listed[name].get("accepted") is not receipt.get("accepted"):
            raise ValueError(f"matrix listing acceptance differs from point: {name}")
        if _find_digest(receipt, str(baseline_receipt_path)) != sha256(
            baseline_receipt_path
        ):
            raise ValueError("C2 point is not bound to the release baseline receipt")
        if _find_digest(receipt, str(baseline_predictions)) != sha256(
            baseline_predictions
        ):
            raise ValueError("C2 point is not bound to release baseline predictions")
        adapter_names = [
            name
            for name in receipt.get("artifacts_sha256", {})
            if name.endswith(".adapter.safetensors")
        ]
        if len(adapter_names) != 1:
            raise ValueError("C2 point lacks exactly one final adapter artifact")
        adapter = path.parent / adapter_names[0]
        found[key] = {
            "alias": alias,
            "receipt_path": path,
            "receipt": receipt,
            "adapter": adapter,
        }
    return {value["alias"]: value for value in found.values()}


def validate_lora_report_gate(path: Path, matrix_root: Path) -> dict:
    receipt = verify_receipt(Path(path))
    matrix_path = Path(matrix_root).resolve() / "matrix.complete.json"
    if (
        receipt.get("status") != "lora_curve_report_complete"
        or receipt.get("release_ready") is not False
        or receipt.get("final_test_images_read") is not False
        or receipt.get("fixed_points")
        != [
            f"b{run.budget}_{run.arm}_s{run.seed}"
            for run in lora_curve.registered_runs(False)
        ]
        or receipt.get("inputs_sha256", {}).get(str(matrix_path)) != sha256(matrix_path)
    ):
        raise ValueError("C2 report gate is incomplete or bound to another matrix")
    return receipt


def decode_rgb(path: Path):
    import cv2
    import numpy as np

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"image decode failed: {path}")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"decoded RGB contract failed: {path}")
    return rgb


class _DecodedCV2:
    """Submission cv2 proxy replacing only its already-excluded file decode."""

    def __init__(self, base):
        self._base = base
        self.rgb = None

    def __getattr__(self, name):
        return getattr(self._base, name)

    def imread(self, _path, _flags):
        if self.rgb is None:
            raise RuntimeError("decoded RGB was not supplied")
        return self.rgb

    def cvtColor(self, image, code):
        if image is self.rgb and code == self._base.COLOR_BGR2RGB:
            return image
        return self._base.cvtColor(image, code)


def decoded_submission_callable(module):
    proxy = _DecodedCV2(module.cv2)
    module.cv2 = proxy

    def predict(rgb):
        proxy.rgb = rgb
        try:
            return module.predict("<predecoded-rgb>")
        finally:
            proxy.rgb = None

    return predict


def _import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def private_source_hashes(submission: Path) -> dict[str, str]:
    paths = [path for path in sorted(Path(submission).rglob("*")) if path.is_file()]
    if not paths or not (Path(submission) / "predict.py").is_file():
        raise ValueError("private ensemble source is incomplete")
    return {str(path.resolve()): sha256(path) for path in paths}


def validate_private_receipt(
    path: Path, submission: Path, expected: Mapping[str, object] | None = None
) -> dict:
    """Bind the extracted private ensemble to its extraction receipt.

    The archive identity, member names and member digests come from the receipt
    itself.  The receipt is accepted only if every file it lists hashes to the
    recorded value and the extracted directory holds exactly the listed files.
    ``expected`` optionally pins ``receipt_sha256``, ``archive_sha256``,
    ``archive_md5`` or ``archive_bytes`` from outside the receipt.
    """
    path = Path(path)
    receipt = verify_receipt(path)
    pins = dict(expected or {})
    if "receipt_sha256" in pins and sha256(path) != pins.pop("receipt_sha256"):
        raise ValueError("private ensemble receipt SHA-256 mismatch")
    archive_bytes = receipt.get("archive_bytes")
    if (
        receipt.get("status") != "private_ensemble_identity_verified"
        or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("archive_sha256")))
        or not re.fullmatch(r"[0-9a-f]{32}", str(receipt.get("archive_md5")))
        or isinstance(archive_bytes, bool)
        or not isinstance(archive_bytes, int)
        or archive_bytes <= 0
        or receipt.get("extracted_root") != str(Path(submission).resolve())
        or any(receipt.get(key) != value for key, value in pins.items())
    ):
        raise ValueError("private ensemble extraction receipt identity mismatch")
    if not (Path(submission) / "predict.py").is_file():
        raise ValueError("private ensemble extracted root is incomplete")
    artifacts = receipt["artifacts_sha256"]
    member_prefix = "private_ensemble/"
    names = {
        name[len(member_prefix) :]
        for name in artifacts
        if name.startswith(member_prefix)
    }
    actual = {
        str(file.relative_to(submission))
        for file in Path(submission).rglob("*")
        if file.is_file()
    }
    if (
        set(artifacts)
        != {"private_ensemble.identity.json"}
        | {f"{member_prefix}{name}" for name in names}
        or "predict.py" not in names
        or actual != names
    ):
        raise ValueError("private ensemble extracted file inventory mismatch")
    model_names = {name for name in names if name.startswith("model/")}
    if set(receipt.get("model_files_sha256", {})) != model_names:
        raise ValueError("private ensemble model file inventory mismatch")
    for name in model_names:
        if receipt["model_files_sha256"][name] != artifacts[f"{member_prefix}{name}"]:
            raise ValueError("private ensemble model digest inventories disagree")
    return receipt


def make_private_predictor(submission: Path, alias: str):
    from tools.paper.member_eval import make_variant

    variant = PRIVATE_VARIANTS[alias]
    scratch, _ = make_variant(str(submission), variant)
    os.environ["BUDGET_S"] = "1000000000"
    sys.path.insert(0, str(scratch))
    module = _import_path(f"timing_{alias}", scratch / "predict.py")
    module._load()
    return decoded_submission_callable(module), scratch


def make_cq_predictor(cq_repo: Path, checkpoint: Path, adapter: Path | None = None):
    import numpy as np
    import torch

    data, evaluation, upstream_lora = lora_curve.load_upstream(
        cq_repo, os.environ["CHESSQUERIES_DATA_ROOT"]
    )
    model = evaluation.load_model(Path(checkpoint), "cpu")
    if adapter is not None:
        from safetensors.torch import load_file

        lora_curve.configure_lora(
            model,
            upstream_lora.inject_lora,
            train_head=False,
            enforce_vit_l_contract=True,
        )
        tensors = load_file(str(adapter), device="cpu")
        expected = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if set(tensors) != expected or len(tensors) != lora_curve.VIT_L_LORA_TENSORS:
            raise ValueError("C2 adapter tensor boundary mismatch")
        if any(
            tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()
            for tensor in tensors.values()
        ):
            raise ValueError("C2 adapter contains non-fp32 or non-finite tensors")
        all_state_names = set(model.state_dict())
        state = model.load_state_dict(tensors, strict=False)
        if state.unexpected_keys or set(state.missing_keys) != all_state_names - set(
            tensors
        ):
            raise ValueError("C2 adapter could not be loaded exactly")
    lora_curve.prepare_baseline_for_evaluation(model, "cuda:0")
    transform = data.eval_transform(644)

    def predict(rgb):
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        tensor = transform(tensor)[None].to("cuda:0")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(tensor)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite ChessQueries logits")
        return lora_curve._fen(logits.argmax(-1)[0].cpu().tolist())

    return predict, None


def gpu_compute_pids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {int(line) for line in result.stdout.splitlines() if line.strip()}


def assert_no_external_gpu_process() -> None:
    others = gpu_compute_pids() - {os.getpid()}
    if others:
        raise RuntimeError(f"external GPU 0 compute process detected: {sorted(others)}")


def hardware_metadata() -> dict:
    fields = (
        "name",
        "driver_version",
        "memory.total",
        "power.limit",
        "utilization.gpu",
        "utilization.memory",
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != len(fields):
        raise ValueError("unexpected nvidia-smi hardware response")
    return dict(zip(fields, values))


def percentile(values: Sequence[float], probability: float) -> float:
    if (
        not values
        or not 0 <= probability <= 1
        or any(not math.isfinite(v) or v < 0 for v in values)
    ):
        raise ValueError("timings must be finite nonnegative values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_times(seconds: Sequence[float]) -> dict:
    if len(seconds) != TIMED_IMAGES:
        raise ValueError(f"expected {TIMED_IMAGES} timed images")
    return {
        "images": len(seconds),
        "median_ms": 1000 * statistics.median(seconds),
        "p90_ms": 1000 * percentile(seconds, 0.90),
    }


def _verify_hash_map(inputs: Mapping[str, str]) -> None:
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"timing input changed: {path}")


def run_timing(args) -> None:
    if (
        socket.gethostname().split(".")[0] != "render-host"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
    ):
        raise RuntimeError("timing runs require host render-host with CUDA_VISIBLE_DEVICES=0")
    selected, selection_receipt, inventory_hash = load_selection(
        args.selection, args.c1_root
    )
    matrix = validate_c2_matrix(args.matrix_root, args.baseline_receipt, args.c1_root)
    validate_lora_report_gate(args.lora_report_receipt, args.matrix_root)
    validate_private_receipt(args.submission_receipt, args.submission)
    inputs = {
        str(Path(args.selection).resolve()): sha256(args.selection),
        str(Path(args.selection).with_suffix(".complete.json").resolve()): sha256(
            Path(args.selection).with_suffix(".complete.json")
        ),
        str(Path(args.baseline_receipt).resolve()): sha256(args.baseline_receipt),
        str(Path(__file__).resolve()): sha256(__file__),
        str(PROTOCOL_PATH.resolve()): sha256(PROTOCOL_PATH),
        str(ROOT / "tools/paper/lora_curve.py"): sha256(
            ROOT / "tools/paper/lora_curve.py"
        ),
        str(ROOT / "tools/paper/member_eval.py"): sha256(
            ROOT / "tools/paper/member_eval.py"
        ),
        str(Path(args.submission_receipt).resolve()): sha256(args.submission_receipt),
        str(Path(args.lora_report_receipt).resolve()): sha256(args.lora_report_receipt),
        str(Path(args.matrix_root).resolve() / "matrix.complete.json"): sha256(
            Path(args.matrix_root).resolve() / "matrix.complete.json"
        ),
        str(Path(args.matrix_root).resolve() / "matrix_exit.txt"): sha256(
            Path(args.matrix_root).resolve() / "matrix_exit.txt"
        ),
    }
    inputs.update(lora_curve.validate_chessqueries_source(args.cq_repo))
    inputs[str(Path(args.checkpoint).resolve())] = sha256(args.checkpoint)
    if (
        inputs[str(Path(args.checkpoint).resolve())]
        != lora_curve.BASE_CHECKPOINT_SHA256
    ):
        raise ValueError("ChessQueries release checkpoint hash mismatch")
    if args.model_alias in PRIVATE_VARIANTS:
        inputs.update(private_source_hashes(args.submission))
    if args.model_alias in C2_ALIASES:
        point = matrix[args.model_alias]
        inputs[str(point["receipt_path"])] = sha256(point["receipt_path"])
        inputs[str(point["adapter"])] = sha256(point["adapter"])
    reserve_output(args.out)
    assert_no_external_gpu_process()
    hardware_before = hardware_metadata()
    if not math.isclose(
        float(hardware_before["power.limit"]), POWER_LIMIT_W, rel_tol=0, abs_tol=0.05
    ):
        raise RuntimeError(f"GPU 0 power limit must be fixed at {POWER_LIMIT_W:.0f} W")
    scratch = None
    try:
        if args.model_alias in PRIVATE_VARIANTS:
            predict, scratch = make_private_predictor(args.submission, args.model_alias)
        elif args.model_alias == "cq_release":
            predict, _ = make_cq_predictor(args.cq_repo, args.checkpoint)
        else:
            predict, _ = make_cq_predictor(
                args.cq_repo, args.checkpoint, matrix[args.model_alias]["adapter"]
            )
        import torch

        assert_no_external_gpu_process()
        for index, row in enumerate(selected[:WARMUP_IMAGES]):
            if index % 10 == 0:
                assert_no_external_gpu_process()
            rgb = decode_rgb(row["image"])
            predict(rgb)
            torch.cuda.synchronize()
        timings = []
        predictions = Path(f"{args.out}.times.jsonl")
        with predictions.open("x") as stream:
            for index, row in enumerate(selected):
                if index % 25 == 0:
                    assert_no_external_gpu_process()
                rgb = decode_rgb(row["image"])
                torch.cuda.synchronize()
                started = time.perf_counter()
                predicted = placement(predict(rgb))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise FloatingPointError("non-finite per-image timing")
                timings.append(elapsed)
                stream.write(
                    json.dumps(
                        {
                            "order": index,
                            "id": row["id"],
                            "scene_id": row["scene_id"],
                            "factor_cell": row["factor_cell"],
                            "parity": row["parity"],
                            "seconds": elapsed,
                            "prediction": predicted,
                        },
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
        assert_no_external_gpu_process()
        hardware_after = hardware_metadata()
        summary = summarize_times(timings)
        _write_json(Path(f"{args.out}.summary.json"), summary)
        _verify_hash_map(inputs)
        verify_selected_inputs(selected)
        verify_full_dev_unchanged(args.c1_root, inventory_hash)
        finish_output(
            args.out,
            {
                "status": "c4_timing_complete",
                "accepted": True,
                "release_ready": False,
                "model_alias": args.model_alias,
                "batch_size": 1,
                "precision": "fp16",
                "warmup_images": WARMUP_IMAGES,
                "timed_images": TIMED_IMAGES,
                "decode_excluded": True,
                "timer_scope": "decoded_RGB_to_piece_placement_including_native_preprocess_router_TTA_argmax_host_transfer_and_FEN_construction",
                "selection_receipt_sha256": sha256(
                    Path(args.selection).with_suffix(".complete.json")
                ),
                "selection": {
                    key: selection_receipt[key]
                    for key in ("selection_seed", "scenes", "images")
                },
                "dev_inventory_sha256": inventory_hash,
                "model_inputs_sha256": inputs,
                "hardware": {"before": hardware_before, "after": hardware_after},
                "software": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "summary": summary,
                "final_test_images_read": False,
                "scored_partition": "dev",
            },
        )
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def run_all(args) -> None:
    """Run all 15 aliases serially in fresh processes, then publish one aggregate."""
    if (
        socket.gethostname().split(".")[0] != "render-host"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
    ):
        raise RuntimeError("timing runs require host render-host with CUDA_VISIBLE_DEVICES=0")
    selected, selection_receipt, _ = load_selection(args.selection, args.c1_root)
    selection_receipt_path = Path(args.selection).with_suffix(".complete.json")
    selection_receipt_sha256 = sha256(selection_receipt_path)
    validate_c2_matrix(args.matrix_root, args.baseline_receipt, args.c1_root)
    validate_lora_report_gate(args.lora_report_receipt, args.matrix_root)
    validate_private_receipt(args.submission_receipt, args.submission)
    out = Path(args.out)
    reserve_output(out)
    run_dir = Path(f"{out}.runs")
    run_dir.mkdir()
    log_dir = Path(f"{out}.logs")
    log_dir.mkdir()
    controller_log = Path(f"{out}.controller.jsonl")
    receipts = []
    with controller_log.open("x") as log:
        for alias in MODEL_ALIASES:
            prefix = run_dir / alias
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--c1-root",
                str(args.c1_root),
                "--selection",
                str(args.selection),
                "--matrix-root",
                str(args.matrix_root),
                "--baseline-receipt",
                str(args.baseline_receipt),
                "--lora-report-receipt",
                str(args.lora_report_receipt),
                "--cq-repo",
                str(args.cq_repo),
                "--checkpoint",
                str(args.checkpoint),
                "--submission",
                str(args.submission),
                "--submission-receipt",
                str(args.submission_receipt),
                "--model-alias",
                alias,
                "--out",
                str(prefix),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            (log_dir / f"{alias}.log").write_text(
                result.stdout
                + ("\n--- stderr ---\n" + result.stderr if result.stderr else "")
            )
            event = {"model_alias": alias, "returncode": result.returncode}
            log.write(json.dumps(event, separators=(",", ":")) + "\n")
            log.flush()
            if result.returncode != 0:
                raise RuntimeError(f"C4 timing child failed: {alias}")
            receipt_path = Path(f"{prefix}.complete.json")
            receipt = verify_receipt(receipt_path)
            if (
                receipt.get("status") != "c4_timing_complete"
                or receipt.get("model_alias") != alias
                or receipt.get("release_ready") is not False
                or receipt.get("final_test_images_read") is not False
                or receipt.get("selection_receipt_sha256") != selection_receipt_sha256
                or receipt.get("timed_images") != TIMED_IMAGES
            ):
                raise ValueError(f"invalid completed timing receipt: {alias}")
            timing_rows = read_jsonl(Path(f"{prefix}.times.jsonl"))
            if [row.get("order") for row in timing_rows] != list(
                range(TIMED_IMAGES)
            ) or [row.get("id") for row in timing_rows] != [
                row["id"] for row in selected
            ]:
                raise ValueError(f"timing rows differ from common selection: {alias}")
            recomputed = summarize_times([row.get("seconds") for row in timing_rows])
            if recomputed != receipt.get("summary"):
                raise ValueError(f"timing summary differs from raw durations: {alias}")
            receipts.append(
                {
                    "model_alias": alias,
                    "receipt": str(receipt_path.resolve()),
                    "receipt_sha256": sha256(receipt_path),
                    "summary": receipt["summary"],
                }
            )
    _write_json(Path(f"{out}.runs.json"), {"runs": receipts})
    first_receipt = verify_receipt(Path(receipts[0]["receipt"]))
    fixed_hardware = first_receipt["hardware"]["before"]
    for row in receipts[1:]:
        hardware = verify_receipt(Path(row["receipt"]))["hardware"]["before"]
        if any(
            hardware[field] != fixed_hardware[field]
            for field in ("name", "driver_version", "memory.total", "power.limit")
        ):
            raise ValueError("hardware identity changed across C4 model processes")
    lines = [
        "# C4 end-to-end latency",
        "",
        (
            f"Fixed C1 dev selection: {selection_receipt['scenes']} scenes / "
            f"{selection_receipt['images']} parity-paired images, "
            f"{selection_receipt['unique_position_groups']} unique position groups "
            f"({selection_receipt['repeated_position_groups']} repeated group-scenes)."
        ),
        (
            f"Hardware: {fixed_hardware['name']}; fixed power limit "
            f"{fixed_hardware['power.limit']} W; driver {fixed_hardware['driver_version']}."
        ),
        (
            "Batch size 1, native FP16 autocast, 30 excluded warm-up images. File decode is excluded; "
            "native resize/detector/router/warp/TTA/argmax/host transfer/FEN construction are included as applicable."
        ),
        "Private variants retain different native members, routing and TTA, so these timings do not compare equal FLOPs.",
        "",
        "| Public model alias | Median ms/image | p90 ms/image |",
        "|---|---:|---:|",
    ]
    for row in receipts:
        lines.append(
            f"| {row['model_alias']} | {row['summary']['median_ms']:.3f} | "
            f"{row['summary']['p90_ms']:.3f} |"
        )
    lines += [
        "",
        (
            "These measurements describe this fixed hardware/software/input pipeline only; "
            "no accuracy, superiority, equal-compute, or general latency conclusion follows automatically."
        ),
        "",
    ]
    with Path(f"{out}.md").open("x") as stream:
        stream.write("\n".join(lines))
    finish_output(
        out,
        {
            "status": "c4_timing_complete",
            "accepted": True,
            "release_ready": False,
            "models": len(receipts),
            "model_aliases": list(MODEL_ALIASES),
            "selection_receipt_sha256": selection_receipt_sha256,
            "final_test_images_read": False,
            "scored_partition": "dev",
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--c1-root", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)

    def shared_run_arguments(command):
        command.add_argument("--c1-root", required=True, type=Path)
        command.add_argument("--selection", required=True, type=Path)
        command.add_argument("--matrix-root", required=True, type=Path)
        command.add_argument("--baseline-receipt", required=True, type=Path)
        command.add_argument("--lora-report-receipt", required=True, type=Path)
        command.add_argument("--cq-repo", required=True, type=Path)
        command.add_argument("--checkpoint", required=True, type=Path)
        command.add_argument("--submission", required=True, type=Path)
        command.add_argument("--submission-receipt", required=True, type=Path)
        command.add_argument("--out", required=True, type=Path)

    run = sub.add_parser("run")
    shared_run_arguments(run)
    run.add_argument("--model-alias", required=True, choices=MODEL_ALIASES)
    all_runs = sub.add_parser("timing-all")
    shared_run_arguments(all_runs)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare_selection(args.c1_root, args.out)
    elif args.command == "run":
        run_timing(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
