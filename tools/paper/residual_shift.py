"""Preregistered residual-shift analysis for the fixed confirmation cohort.

The metric path consumes only the saved release-model predictions for the
canonical (``parity``/a1-dark) view.  The attention path is descriptive: it
captures all 64 query maps for a fixed, outcome-independent sample and does not
turn attention into a causal explanation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

try:
    from tools.paper.benchmark_io import (
        finish_output,
        read_jsonl,
        reserve_output,
        sha256,
    )
    from tools.paper.parity_probe import gpu_idle, swap_parity
    from tools.paper.transfer_cohort import verify_receipt
except ImportError:  # Direct execution from tools/paper/.
    from benchmark_io import finish_output, read_jsonl, reserve_output, sha256
    from parity_probe import gpu_idle, swap_parity
    from transfer_cohort import verify_receipt


CLASSES = ".PNBRQKpnbrqk"
FAMILIES = (
    "deplorable",
    "killgorack",
    "lyricsz",
    "omie",
    "phuong",
    "pjhanzlik",
)
ATTENTION_FAMILIES = ("lyricsz", "pjhanzlik", "deplorable")
ATTENTION_PER_FAMILY = 10
ATTENTION_SEED = 20260915
BASE_CHECKPOINT_SHA256 = (
    "6151bdd98fbe25f32080c097eba7ae75808b5615160e4867240931085fc762e5"
)
UPSTREAM_COMMIT = "2a9f3c00619133541c65de4315149037b5dece80"
ATTENTION_SOURCE_SHA256 = (
    "af091615c10157259ceddcdb0ce6e0e871c74b046b7ef5b87db0c84939ed9d15"
)
SOURCE_SIZE = (1920, 1080)
MODEL_SIZE = (644, 644)
CANONICAL_CONDITION = "parity"


@contextmanager
def attention_precision(torch):
    """R1: compare both attention paths in full FP32, restoring caller state."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        if torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("full FP32 attention precision was not enabled")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def fen_labels(fen: str) -> np.ndarray:
    """Return 64 class ids in FEN order, a8 through h1."""
    placement = fen.split()[0]
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"expected eight FEN ranks: {fen!r}")
    labels: list[int] = []
    for rank in ranks:
        expanded: list[int] = []
        for symbol in rank:
            if symbol in "12345678":
                expanded.extend([0] * int(symbol))
            elif symbol in CLASSES[1:]:
                expanded.append(CLASSES.index(symbol))
            else:
                raise ValueError(f"invalid FEN symbol {symbol!r}: {fen!r}")
        if len(expanded) != 8:
            raise ValueError(f"FEN rank does not contain eight squares: {fen!r}")
        labels.extend(expanded)
    return np.asarray(labels, dtype=np.int64)


def _rate(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "correct": int(numerator),
        "total": int(denominator),
        "accuracy_pct": 100.0 * numerator / denominator if denominator else 0.0,
    }


def summarize_arrays(truth: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute the locked confusion, class and occupancy summaries."""
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 2 or truth.shape[1] != 64:
        raise ValueError("truth/predicted must have the same (boards, 64) shape")
    if not len(truth):
        raise ValueError("cannot summarize an empty cohort")
    if np.any((truth < 0) | (truth >= len(CLASSES))) or np.any(
        (predicted < 0) | (predicted >= len(CLASSES))
    ):
        raise ValueError("class id outside the 13-class label space")

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(confusion, (truth.ravel(), predicted.ravel()), 1)
    occupied = truth != 0
    binary_correct = (truth != 0) == (predicted != 0)
    board_wrong = (truth != predicted).sum(axis=1)
    histogram = np.bincount(board_wrong, minlength=65)
    return {
        "n_boards": len(truth),
        "n_squares": int(truth.size),
        "classes": CLASSES,
        "confusion_gt_rows_pred_columns": confusion.tolist(),
        "occupied_gt_exact_class": _rate(
            int(((truth == predicted) & occupied).sum()), int(occupied.sum())
        ),
        "binary_occupancy": _rate(int(binary_correct.sum()), int(truth.size)),
        "all_square_exact_class": _rate(
            int((truth == predicted).sum()), int(truth.size)
        ),
        "exact_boards": _rate(int((board_wrong == 0).sum()), len(truth)),
        "wrong_squares_per_board_histogram_0_to_64": histogram.tolist(),
    }


def summarize_predictions(rows: list[dict]) -> dict:
    validate_default_rows(rows)
    truth = np.stack([fen_labels(row["gt"]) for row in rows])
    predicted = np.stack([fen_labels(row["pred"]) for row in rows])
    by_family = {}
    for family in FAMILIES:
        indices = [i for i, row in enumerate(rows) if row["family"] == family]
        by_family[family] = summarize_arrays(truth[indices], predicted[indices])
    return {
        "label_order": CLASSES,
        "square_order": "a8,b8,...,h8,a7,...,h1",
        "overall": summarize_arrays(truth, predicted),
        "by_geometry": by_family,
    }


def markdown_report(metrics: dict, prefix: Path) -> str:
    """Describe all families without automatically turning attention into causality."""
    lines = [
        "# Residual recognition shift on canonical a1-dark boards",
        "",
        "Fixed historical confirmation cohort: 288 boards, 48 per geometry. Only the",
        "unchanged release model's `parity` condition is analyzed; the historical",
        "`control` condition is a1-light and excluded. No selectors or final-test data.",
        "",
        "**Mechanism remains unresolved.** These metrics describe the errors; the",
        "fixed-sample attention maps are descriptive routing evidence, not a causal",
        "intervention or a proof that highlighted pixels caused the predictions.",
        "",
        "| Family | Exact boards | All-square % | GT-occupied class % | Binary occupancy % |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, item in metrics["by_geometry"].items():
        exact = item["exact_boards"]
        lines.append(
            f"| {family} | {exact['correct']}/{exact['total']} | "
            f"{item['all_square_exact_class']['accuracy_pct']:.3f} | "
            f"{item['occupied_gt_exact_class']['accuracy_pct']:.3f} | "
            f"{item['binary_occupancy']['accuracy_pct']:.3f} |"
        )
    lines += [
        "",
        "## Error types and wrong-square histograms",
        "",
        "Confusion matrices use GT rows and prediction columns in `.PNBRQKpnbrqk` order.",
        f"Full 13×13 matrices and all 65 histogram bins are in `{prefix.name}.metrics.json`.",
        "Below, `wrong:boards` lists every nonzero histogram bin, not selected examples.",
    ]
    for family, item in metrics["by_geometry"].items():
        matrix = item["confusion_gt_rows_pred_columns"]
        additions = sum(matrix[0][1:])
        omissions = sum(row[0] for row in matrix[1:])
        confusions = sum(
            matrix[i][j] for i in range(1, 13) for j in range(1, 13) if i != j
        )
        histogram = ", ".join(
            f"{i}:{n}"
            for i, n in enumerate(item["wrong_squares_per_board_histogram_0_to_64"])
            if n
        )
        lines += [
            "",
            f"### {family}",
            f"Empty→piece: {additions}; piece→empty: {omissions}; occupied wrong class: {confusions}.",
            f"Histogram: {histogram}.",
        ]
    lines += [
        "",
        "## Fixed attention sample",
        "",
        "Ten outcome-independent boards each from lyricsz, pjhanzlik and deplorable;",
        "all 64 final-decoder-layer, head-mean query maps per board are retained.",
        "Square model-input sheets preserve the 644×644 squash; 16:9 source sheets",
        "stretch those same maps back to 1920×1080. Heat colours are normalized",
        "to each query map's own maximum for display, not comparable absolute saliency.",
        f"Selection: `{prefix.name}.attention_selection.json`; raw maps/logits:",
        f"`{prefix.name}.attention_raw.npz`; all sheet hashes and forward-equivalence",
        f"checks: `{prefix.name}.contact_sheets.json`.",
        "",
        "No model selection, corrective rule, or causal conclusion follows automatically.",
        "`release_ready=false`.",
        "",
    ]
    return "\n".join(lines)


def validate_default_rows(rows: list[dict]) -> None:
    """Fail closed on any noncanonical, selector-derived, or sealed row."""
    if len(rows) != 288:
        raise ValueError(f"expected 288 canonical confirmation rows, found {len(rows)}")
    keys = [row.get("key") for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate canonical prediction key")
    counts = Counter(row.get("family") for row in rows)
    if counts != Counter({family: 48 for family in FAMILIES}):
        raise ValueError(f"unexpected geometry counts: {dict(counts)}")
    for row in rows:
        if row.get("model") != "default" or row.get("condition") != CANONICAL_CONDITION:
            raise ValueError(
                "only saved default-model canonical predictions are allowed"
            )
        if not str(row.get("key", "")).startswith("confirmation/"):
            raise ValueError("C3 accepts only the fixed confirmation partition")
        if "final_test" in json.dumps(row, sort_keys=True).lower():
            raise ValueError("final_test data is forbidden in C3")
        fen_labels(row["gt"])
        fen_labels(row["pred"])


def select_attention_rows(records: list[dict]) -> list[dict]:
    """Hash-rank ten canonical boards per locked geometry before inference."""
    eligible = [
        row
        for row in records
        if row.get("partition") == "confirmation"
        and row.get("condition") == CANONICAL_CONDITION
        and row.get("family") in ATTENTION_FAMILIES
    ]
    selected = []
    for family in ATTENTION_FAMILIES:
        group = [row for row in eligible if row["family"] == family]
        if len(group) != 48:
            raise ValueError(
                f"expected 48 canonical {family} records, found {len(group)}"
            )
        ranked = sorted(
            group,
            key=lambda row: (
                hashlib.sha256(
                    f"{ATTENTION_SEED}:{row['group_id']}:{row['image_sha256']}".encode()
                ).hexdigest(),
                row["key"],
            ),
        )
        selected.extend(ranked[:ATTENTION_PER_FAMILY])
    return selected


def validate_canonical_parity_sources(control: bytes, parity: bytes) -> str:
    """Prove the treatment, rather than the byte-identical control, has dark a1."""
    if swap_parity(control) != parity:
        raise ValueError("recorded parity renderer is not the declared parity swap")
    pattern = rb"\(r\+c\)%2===1\s*\?\s*(lm|dm)\s*:\s*(lm|dm)"
    control_match = re.findall(pattern, control)
    parity_match = re.findall(pattern, parity)
    if control_match != [(b"lm", b"dm")] or parity_match != [(b"dm", b"lm")]:
        raise ValueError(
            "renderer checkerboard branches do not match the locked contract"
        )
    # Renderer row 0 is rank 8, so a1 is (row=7, column=0), the odd branch.
    if (7 + 0) % 2 != 1 or parity_match[0][0] != b"dm":
        raise AssertionError("a1 must select the dark material in the parity renderer")
    return CANONICAL_CONDITION


def validate_attention_grid(grid: np.ndarray, batch_size: int) -> None:
    grid = np.asarray(grid)
    if grid.ndim != 4 or grid.shape[:2] != (batch_size, 64):
        raise ValueError(f"unexpected attention shape: {grid.shape}")
    if not np.isfinite(grid).all() or np.any(grid < -1e-7):
        raise ValueError("attention maps must be finite and nonnegative")
    if not np.allclose(grid.sum(axis=(-2, -1)), 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError("patch attention must be normalized per query")


def validate_attention_model_shape(model) -> None:
    """Check the minimal-reproduction model exposes the upstream viz contract."""
    required = ("encoder", "queries", "decoder", "norm", "head")
    if any(not hasattr(model, name) for name in required):
        raise TypeError("release model does not expose the query-attention contract")
    layers = getattr(model.decoder, "layers", None)
    if layers is None or not len(layers):
        raise TypeError("release model has no decoder layers to capture")
    for layer in layers:
        if not hasattr(layer, "multihead_attn") or not hasattr(layer, "self_attn"):
            raise TypeError(
                "decoder layer is incompatible with upstream attention capture"
            )
    if not hasattr(model.encoder, "num_prefix_tokens") or not hasattr(
        model.encoder, "patch_embed"
    ):
        raise TypeError(
            "encoder lacks patch/prefix metadata required by attention maps"
        )


def verify_path_hashes(expected: dict[str, str], *, label: str) -> None:
    """Rehash an explicit, receipt-owned file set; never discover extra inputs."""
    if not expected:
        raise ValueError(f"{label} hash inventory is empty")
    for name, digest in expected.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"{label} changed: {path}")


def execution_source_hashes(protocol: Path) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        protocol.resolve(),
        Path(__file__).with_name("benchmark_io.py").resolve(),
        Path(__file__).with_name("transfer_cohort.py").resolve(),
        Path(__file__).with_name("parity_probe.py").resolve(),
        Path(__file__).resolve().parents[2]
        / "docs/RESIDUAL_SHIFT_PROTOCOL_2026-09-10.md",
    )
    return {str(path): sha256(path) for path in paths}


def _load_locked_inputs(run: Path) -> tuple[list[dict], list[dict], dict]:
    prepare_path = run / "prepare.complete.json"
    render_path = run / "render_validation.complete.json"
    confirmation_path = run / "confirmation.complete.json"
    prepare = verify_receipt(prepare_path)
    render = verify_receipt(render_path)
    confirmation = verify_receipt(confirmation_path)
    if prepare.get("status") != "cohorts_fixed_before_inference":
        raise ValueError("unexpected prepare receipt status")
    if render.get("status") != "all_six_families_validated":
        raise ValueError("unexpected render-validation receipt status")
    if confirmation.get("status") != "fixed_confirmation_complete":
        raise ValueError("unexpected confirmation receipt status")
    if confirmation.get("base_checkpoint_sha256") != BASE_CHECKPOINT_SHA256:
        raise ValueError("confirmation used a different base checkpoint")
    verify_path_hashes(
        confirmation.get("source_sha256", {}), label="confirmation model/source input"
    )
    if confirmation.get("prepare_sha256") != sha256(run / "prepare.json"):
        raise ValueError("confirmation is not bound to prepare.json")
    if confirmation.get("records_sha256") != sha256(run / "prepare.records.jsonl"):
        raise ValueError("confirmation is not bound to prepare records")
    if render.get("source_manifest_sha256") != sha256(run / "prepare.records.jsonl"):
        raise ValueError("render validation is not bound to prepare records")

    plan = json.loads((run / "prepare.json").read_text())
    for family, family_plan in plan.get("families", {}).items():
        control_path = run / "src/rendering" / family_plan["control_renderer"]
        parity_path = run / "src/rendering" / family_plan["parity_renderer"]
        if (
            validate_canonical_parity_sources(
                control_path.read_bytes(), parity_path.read_bytes()
            )
            != CANONICAL_CONDITION
        ):
            raise AssertionError(f"unexpected canonical condition for {family}")

    records = read_jsonl(run / "render_validation.records.jsonl")
    canonical_records = [
        row for row in records if row.get("condition") == CANONICAL_CONDITION
    ]
    if len(canonical_records) != 288:
        raise ValueError("render validation lacks 288 canonical rows")
    if any(row.get("partition") != "confirmation" for row in canonical_records):
        raise ValueError("C3 record population is not confirmation-only")
    record_index = {row["key"]: row for row in canonical_records}
    if len(record_index) != 288:
        raise ValueError("duplicate canonical render key")
    for row in canonical_records:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError(f"image changed: {row['key']}")

    all_predictions = read_jsonl(run / "confirmation.predictions.jsonl")
    default = [
        row
        for row in all_predictions
        if row.get("model") == "default" and row.get("condition") == CANONICAL_CONDITION
    ]
    validate_default_rows(default)
    if {row["key"] for row in default} != set(record_index):
        raise ValueError("saved default predictions do not match canonical renders")
    for row in default:
        if row["gt"] != record_index[row["key"]]["fen"]:
            raise ValueError(f"ground truth differs from frozen render: {row['key']}")

    inputs = {
        str(path.resolve()): sha256(path)
        for path in (
            prepare_path,
            render_path,
            confirmation_path,
            run / "prepare.json",
            run / "prepare.records.jsonl",
            run / "render_validation.records.jsonl",
            run / "confirmation.predictions.jsonl",
        )
    }
    return canonical_records, sorted(default, key=lambda row: row["key"]), inputs


def _assert_upstream(upstream: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError("attention source is not the pinned upstream commit")
    dirty = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    if dirty.stdout.strip():
        raise ValueError("pinned upstream attention checkout is dirty")
    digest = sha256(upstream / "chessqueries/viz/attention.py")
    if digest != ATTENTION_SOURCE_SHA256:
        raise ValueError("pinned upstream attention source has an unexpected Git blob")
    return digest


def contact_geometry(source_scale: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return interpolation and tile sizes without collapsing either aspect ratio."""
    return (SOURCE_SIZE, (192, 108)) if source_scale else (MODEL_SIZE, (192, 192))


def _contact_sheet(image, grids, source_scale: bool):
    """Render all 64 fixed query maps as an 8x8 thumbnail sheet."""
    import cv2

    target, (tile_width, tile_height) = contact_geometry(source_scale)
    base = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    tiles = []
    for query, grid in enumerate(grids):
        heat = cv2.resize(grid, target, interpolation=cv2.INTER_LINEAR)
        heat = np.clip(heat / max(float(heat.max()), 1e-12), 0.0, 1.0)
        color = cv2.applyColorMap(np.uint8(np.rint(255 * heat)), cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(base, 0.55, color, 0.45, 0.0)
        overlay = cv2.resize(
            overlay, (tile_width, tile_height), interpolation=cv2.INTER_AREA
        )
        cv2.putText(
            overlay,
            f"{chr(97 + query % 8)}{8 - query // 8}",
            (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(overlay)
    return np.concatenate(
        [np.concatenate(tiles[row : row + 8], axis=1) for row in range(0, 64, 8)],
        axis=0,
    )


def _run_attention(
    selected: list[dict],
    predictions: dict[str, dict],
    checkpoint: Path,
    cq_repo: Path,
    upstream: Path,
    output_prefix: Path,
) -> tuple[dict, dict]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("C3 attention must use CUDA_VISIBLE_DEVICES=0")
    if sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise ValueError("base checkpoint changed")
    initial_attention_sha256 = _assert_upstream(upstream)
    gpu_idle()

    import cv2
    import torch
    from torchvision.io import read_image

    # Resolve the minimal reproduction first.  It is the exact implementation
    # that produced the locked predictions; upstream is added only for its viz
    # helper, never as a replacement model/data module.
    sys.path.insert(0, str(cq_repo))
    eval_transform = importlib.import_module("data").eval_transform
    load_model = importlib.import_module("eval").load_model

    sys.path.insert(0, str(upstream))
    extract_attention = importlib.import_module(
        "chessqueries.viz.attention"
    ).extract_attention

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    model = load_model(checkpoint, "cuda:0").eval().requires_grad_(False)
    validate_attention_model_shape(model)
    transform = eval_transform(644)
    maps_out, logits_out, keys_out = [], [], []
    contact_dir = Path(f"{output_prefix}.contact_sheets")
    contact_dir.mkdir(parents=True, exist_ok=False)
    contact_hashes = {}
    equivalence = []
    for start in range(0, len(selected), 2):
        gpu_idle(os.getpid())
        batch_rows = selected[start : start + 2]
        raw_images = []
        tensors = []
        for row in batch_rows:
            image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
            if image is None or (image.shape[1], image.shape[0]) != SOURCE_SIZE:
                raise ValueError(f"expected a 1920x1080 source image: {row['key']}")
            raw_images.append(image)
            tensors.append(transform(read_image(row["image"]).float()))
        batch = torch.stack(tensors).to("cuda:0")
        attention_layers = list(model.decoder.layers)
        attention_modules = [
            module
            for layer in attention_layers
            for module in (layer.multihead_attn, layer.self_attn)
        ]
        if any("forward" in module.__dict__ for module in attention_modules):
            raise ValueError("cross-attention forward was already overridden")
        previous_precision = torch.get_float32_matmul_precision()
        if batch.dtype != torch.float32 or any(
            p.dtype != torch.float32
            for p in model.parameters()
            if p.is_floating_point()
        ):
            raise ValueError("attention comparison requires FP32 model and inputs")
        with attention_precision(torch), torch.no_grad():
            reference_before = model(batch)
            logits, maps = extract_attention(model, batch)
            reference_after = model(batch)
        if torch.get_float32_matmul_precision() != previous_precision:
            raise ValueError("attention precision context was not restored")
        if any("forward" in module.__dict__ for module in attention_modules):
            raise ValueError("attention extraction did not restore module forwards")
        capture_delta = float((logits - reference_before).abs().max().cpu())
        restoration_delta = float(
            (reference_after - reference_before).abs().max().cpu()
        )
        capture_ok = torch.allclose(logits, reference_before, atol=1e-4, rtol=1e-4)
        restoration_ok = torch.allclose(
            reference_after, reference_before, atol=1e-4, rtol=1e-4
        )
        check = {
            "keys": [row["key"] for row in batch_rows],
            "capture_max_abs_logit_delta": capture_delta,
            "restored_max_abs_logit_delta": restoration_delta,
            "capture_equivalent": bool(capture_ok),
            "restoration_equivalent": bool(restoration_ok),
            "matmul_precision_during_capture": "highest",
            "sdp_backend": "auto",
            "caller_precision_restored": previous_precision,
        }
        with Path(f"{output_prefix}.attention_checks.jsonl").open("a") as stream:
            stream.write(json.dumps(check, allow_nan=False) + "\n")
        if not capture_ok or not restoration_ok:
            raise ValueError(f"attention forward equivalence failed: {check}")
        equivalence.append(check)
        grid = maps.cross_grid(layer=-1, head=None).float().cpu().numpy()
        validate_attention_grid(grid, len(batch_rows))
        labels = logits.argmax(-1).cpu().numpy()
        for offset, (row, raw_image) in enumerate(zip(batch_rows, raw_images)):
            expected = fen_labels(predictions[row["key"]]["pred"])
            if not np.array_equal(labels[offset], expected):
                raise ValueError(
                    f"fresh attention forward differs from saved default: {row['key']}"
                )
            maps_out.append(grid[offset])
            logits_out.append(logits[offset].float().cpu().numpy())
            keys_out.append(row["key"])
            stem = row["key"].replace("/", "__")
            for suffix, source_scale in (
                ("model644", False),
                ("source1920x1080", True),
            ):
                path = contact_dir / f"{stem}.{suffix}.png"
                sheet = _contact_sheet(raw_image, grid[offset], source_scale)
                if not cv2.imwrite(str(path), sheet):
                    raise OSError(f"failed to write {path}")
                contact_hashes[str(path.relative_to(output_prefix.parent))] = sha256(
                    path
                )
        del batch, logits, maps, reference_before, reference_after

    gpu_idle(os.getpid())
    if _assert_upstream(upstream) != initial_attention_sha256:
        raise ValueError("attention source changed during capture")
    if sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise ValueError("base checkpoint changed during capture")

    np.savez_compressed(
        f"{output_prefix}.attention_raw.npz",
        keys=np.asarray(keys_out),
        query_order=np.asarray([f"{chr(97 + i % 8)}{8 - i // 8}" for i in range(64)]),
        cross_grid=np.stack(maps_out),
        logits=np.stack(logits_out),
    )
    contact_manifest = {
        "description": "All 64 fixed queries; no query or board chosen by outcome.",
        "model_input_scale": list(MODEL_SIZE),
        "source_frame_scale": list(SOURCE_SIZE),
        "model_input_sheet_tile": [192, 192],
        "source_frame_sheet_tile": [192, 108],
        "forward_equivalence": equivalence,
        "files_sha256": dict(sorted(contact_hashes.items())),
    }
    Path(f"{output_prefix}.contact_sheets.json").write_text(
        json.dumps(contact_manifest, indent=2, allow_nan=False) + "\n"
    )
    return contact_manifest, {
        "upstream_commit": UPSTREAM_COMMIT,
        "attention_source_sha256": initial_attention_sha256,
        "checkpoint_sha256": sha256(checkpoint),
        "attention_precision": "fp32_highest_auto_sdp",
        "historical_prediction_precision": "fp32_high_tf32",
    }


def run(args) -> None:
    run_root = args.run.resolve()
    if "final_test" in str(run_root).lower():
        raise ValueError("final_test paths are forbidden in C3")
    protocol = args.protocol.resolve()
    source_hashes = execution_source_hashes(protocol)
    records, default, inputs = _load_locked_inputs(run_root)
    selected = select_attention_rows(records)
    selected_keys = {row["key"] for row in selected}
    if len(selected_keys) != 30:
        raise ValueError("attention selection must contain 30 unique boards")

    prefix = args.out.resolve()
    reserve_output(prefix)
    Path(f"{prefix}.default_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in default)
    )
    metrics = summarize_predictions(default)
    Path(f"{prefix}.metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    selected_payload = {
        "seed": ATTENTION_SEED,
        "algorithm": "ascending SHA256(seed:group_id:image_sha256), key tiebreak",
        "families": list(ATTENTION_FAMILIES),
        "n_per_family": ATTENTION_PER_FAMILY,
        "rows": selected,
    }
    Path(f"{prefix}.attention_selection.json").write_text(
        json.dumps(selected_payload, indent=2, allow_nan=False) + "\n"
    )

    plan = json.loads((run_root / "prepare.json").read_text())
    contact_manifest, model_inputs = _run_attention(
        selected,
        {row["key"]: row for row in default},
        Path(plan["checkpoint"]),
        Path(plan["cq_repo"]),
        args.upstream.resolve(),
        prefix,
    )
    current_records, current_default, current_inputs = _load_locked_inputs(run_root)
    if (
        inputs != current_inputs
        or records != current_records
        or default != current_default
    ):
        raise ValueError("C3 inputs changed during execution")
    verify_path_hashes(source_hashes, label="C3 execution source")
    Path(f"{prefix}.md").write_text(markdown_report(metrics, prefix))
    finish_output(
        prefix,
        {
            "status": "c3_residual_complete",
            "result_disposition": "descriptive_complete_even_if_no_pattern",
            "accepted": None,
            "cohort": "fixed_confirmation_parity_a1_dark",
            "boards": 288,
            "attention_boards": 30,
            "attention_queries_per_board": 64,
            "input_sha256": inputs,
            "model_inputs": model_inputs,
            "source_sha256": source_hashes,
            "selection_sha256": sha256(Path(f"{prefix}.attention_selection.json")),
            "contact_sheet_manifest_sha256": sha256(
                Path(f"{prefix}.contact_sheets.json")
            ),
            "contact_sheet_count": len(contact_manifest["files_sha256"]),
            "final_test_images_read": False,
            "release_ready": False,
            "interpretation_boundary": (
                "attention is descriptive localization evidence, not causal proof"
            ),
        },
    )
    print(json.dumps({"complete": f"{prefix}.complete.json"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
