#!/usr/bin/env python3
"""Preregistered C2 encoder-LoRA learning-curve runner.

This runner deliberately exposes one run at a time.  It never reads benchmark
``final_test`` images, never selects a checkpoint on development performance,
and writes only LoRA (and, when explicitly enabled, head) tensors.  The caller
is responsible for scheduling the ten registered runs and for choosing a fresh
output prefix for every invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.paper.benchmark_io import finish_output, read_jsonl, reserve_output, sha256
from tools.paper.fen_pools import sym_key
from tools.paper.transfer_cohort import verify_receipt

CQ_REVISION = "2a9f3c00619133541c65de4315149037b5dece80"
BASE_CHECKPOINT_SHA256 = (
    "6151bdd98fbe25f32080c097eba7ae75808b5615160e4867240931085fc762e5"
)
CHESSRED_ANNOTATIONS_SHA256 = (
    "16e99d7e8535c0fc56507caa2aa5f7594d7ca076ebab3f5a432cfd4aa10668cc"
)
LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
BUDGETS = (500, 2_000, 8_000)
ARMS = ("dark", "both")
RUN_SEEDS = {500: (303,), 2_000: (303, 304, 305), 8_000: (303,)}
PARITIES = ("a1_dark", "a1_light")
BOOTSTRAP_SEED = 20260913
VIT_L_LORA_TENSORS = 192
VIT_L_LORA_PARAMETERS = 3_145_728
QUERY_HEAD_PARAMETERS = 13_325
RECIPE = {
    "lr": 1e-4,
    "weight_decay": 0.0,
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "batch_size": 6,
    "steps": 1_500,
    "resolution": 644,
    "optimizer": "AdamW",
    "scheduler": None,
    "warmup_steps": 0,
    "input_augmentation": "none",
    "train_precision": "bf16",
    "evaluation_precision": "fp32_tf32",
    "eval_batch_size": 4,
    "bootstrap_seed": BOOTSTRAP_SEED,
}


@dataclass(frozen=True)
class RunSpec:
    budget: int
    arm: str
    seed: int
    train_head: bool

    def __post_init__(self) -> None:
        if self.budget not in BUDGETS:
            raise ValueError(f"budget must be one of {BUDGETS}")
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.seed not in RUN_SEEDS[self.budget]:
            raise ValueError(
                f"seed {self.seed} is not registered for budget {self.budget}"
            )
        if type(self.train_head) is not bool:
            raise TypeError("train_head must be an explicit boolean")


def registered_runs(train_head: bool) -> list[RunSpec]:
    """Return the ten preregistered arm/budget/seed cells."""
    if type(train_head) is not bool:
        raise TypeError("train_head must be an explicit boolean")
    return [
        RunSpec(budget, arm, seed, train_head)
        for budget in BUDGETS
        for seed in RUN_SEEDS[budget]
        for arm in ARMS
    ]


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def validate_chessqueries_source(repo: Path) -> dict[str, str]:
    """Pin the upstream implementation and snapshot every imported source file."""
    repo = Path(repo).resolve()
    if _git_head(repo) != CQ_REVISION:
        raise ValueError(f"ChessQueries checkout must be pinned at {CQ_REVISION}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True
    )
    if status.strip():
        raise ValueError("ChessQueries source checkout must be clean")
    paths = sorted((repo / "minimal_reproduction").rglob("*.py")) + [
        repo / "chessqueries/models/lora.py"
    ]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(
            "pinned ChessQueries checkout is missing required sources"
        )
    return {str(path): sha256(path) for path in paths}


def harness_source_inputs() -> dict[str, str]:
    """Bind the runner and every repository helper it executes."""
    paths = [
        Path(__file__),
        ROOT / "tools/paper/benchmark_io.py",
        ROOT / "tools/paper/benchmark_v2.py",
        ROOT / "tools/paper/fen_pools.py",
        ROOT / "tools/paper/transfer_cohort.py",
    ]
    return {str(path.resolve()): sha256(path) for path in paths}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream(repo: Path, data_root: Path):
    """Load the authors' minimal model/data/eval path and their LoRA injector."""
    os.environ["CHESSQUERIES_DATA_ROOT"] = str(Path(data_root).resolve())
    minimal = Path(repo).resolve() / "minimal_reproduction"
    # The authors' eval.py imports these modules by their standalone names.
    if str(minimal) not in sys.path:
        sys.path.insert(0, str(minimal))
    data = _load_module("data", minimal / "data.py")
    _load_module("model", minimal / "model.py")
    evaluation = _load_module("cq_minimal_eval", minimal / "eval.py")
    lora = _load_module(
        "cq_pinned_lora", Path(repo).resolve() / "chessqueries/models/lora.py"
    )
    return data, evaluation, lora


def configure_lora(
    model, inject_lora, *, train_head: bool, enforce_vit_l_contract: bool = False
):
    """Freeze the base, inject encoder adapters, and optionally train the head.

    Frozen downstream modules remain in the autograd graph.  In particular this
    function disables either upstream encoder no-grad flag rather than using a
    frozen-forward shortcut that would sever gradients to the adapters.
    """
    if type(train_head) is not bool:
        raise TypeError("train_head must be an explicit boolean")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if hasattr(model, "encoder_frozen"):
        model.encoder_frozen = False
    if hasattr(model, "freeze_encoder"):
        model.freeze_encoder = False
    params = list(
        inject_lora(
            model.encoder,
            targets=LORA_TARGETS,
            r=RECIPE["rank"],
            alpha=RECIPE["alpha"],
            dropout=RECIPE["dropout"],
        )
    )
    if train_head:
        for parameter in model.head.parameters():
            parameter.requires_grad = True
            params.append(parameter)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    expected = {
        name
        for name, _ in model.named_parameters()
        if name.endswith((".A", ".B")) or (train_head and name.startswith("head."))
    }
    if not trainable or trainable != expected:
        raise ValueError(
            f"unexpected trainable parameter boundary: {sorted(trainable ^ expected)}"
        )
    if enforce_vit_l_contract:
        lora_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.endswith((".A", ".B"))
        ]
        if (
            len(lora_parameters) != VIT_L_LORA_TENSORS
            or sum(parameter.numel() for parameter in lora_parameters)
            != VIT_L_LORA_PARAMETERS
        ):
            raise ValueError("ViT-L LoRA tensor/parameter cardinality mismatch")
        head_parameters = sum(
            parameter.numel() for parameter in model.head.parameters()
        )
        if head_parameters != QUERY_HEAD_PARAMETERS:
            raise ValueError("query-head parameter cardinality mismatch")
    return params


def seed_run(seed: int) -> None:
    """Seed adapter initialization and the later shuffled loader from one run seed."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_model_and_rebind(model, device):
    """Move the complete configured model, then return its *current* trainable params.

    LoRA is injected while the base is on CPU because the pinned upstream injector
    constructs A/B on CPU.  Rebinding after ``Module.to`` is deliberate: PyTorch is
    allowed to replace Parameter objects for some device transitions (notably meta).
    """
    import torch

    before_parameters = {
        name: (parameter.dtype, parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }
    before_buffers = {name: buffer.dtype for name, buffer in model.named_buffers()}
    trainable_names = [
        name for name, (_, requires_grad) in before_parameters.items() if requires_grad
    ]
    model.to(device)
    current_parameters = dict(model.named_parameters())
    current_buffers = dict(model.named_buffers())
    if set(current_parameters) != set(before_parameters):
        raise ValueError("model parameter inventory changed during device move")
    if set(current_buffers) != set(before_buffers):
        raise ValueError("model buffer inventory changed during device move")
    target = torch.device(device)
    for name, parameter in current_parameters.items():
        dtype, requires_grad = before_parameters[name]
        if (
            parameter.device != target
            or parameter.dtype != dtype
            or parameter.requires_grad != requires_grad
        ):
            raise ValueError(f"parameter device/dtype/trainability mismatch: {name}")
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise ValueError(f"expected fp32 model parameter before autocast: {name}")
    for name, buffer in current_buffers.items():
        if buffer.device != target or buffer.dtype != before_buffers[name]:
            raise ValueError(f"buffer device/dtype mismatch: {name}")
    rebound = [current_parameters[name] for name in trainable_names]
    if any(
        parameter is not current_parameters[name]
        for name, parameter in zip(trainable_names, rebound)
    ):
        raise AssertionError(
            "rebound optimizer parameters are not current model objects"
        )
    return rebound


def prepare_baseline_for_evaluation(model, device):
    """Freeze and move an inference-only baseline; no optimizer list is permitted."""
    model.requires_grad_(False)
    trainable = move_model_and_rebind(model, device)
    if trainable:
        raise AssertionError("baseline move unexpectedly produced trainable parameters")
    model.eval()
    return model


def validate_receipt_bindings(
    benchmark: Mapping,
    adaptation: Mapping,
    design: Mapping,
    *,
    benchmark_receipt_sha256: str,
    design_receipt_sha256: str,
) -> None:
    """Validate the exact C1 design -> benchmark -> adaptation receipt chain."""
    if (
        design.get("status") != "design_locked_before_rendering"
        or design.get("benchmark_schema") != 2
    ):
        raise ValueError("C1 design receipt is not a locked schema-v2 design")
    if (
        benchmark.get("status") != "schema_v2_frozen"
        or benchmark.get("benchmark_schema") != 2
        or benchmark.get("final_test_sealed") is not True
        or benchmark.get("images") != 20_000
    ):
        raise ValueError(
            "C1 benchmark receipt is not the complete sealed schema-v2 freeze"
        )
    if (
        adaptation.get("status") != "schema_v2_frozen"
        or adaptation.get("benchmark_schema") != 2
        or adaptation.get("final_test_sealed") is not False
        or adaptation.get("images") != 16_000
    ):
        raise ValueError("adaptation receipt is not the complete schema-v2 freeze")
    if benchmark.get("design_receipt_sha256") != design_receipt_sha256:
        raise ValueError("benchmark receipt is not bound to the current design receipt")
    if adaptation.get("design_receipt_sha256") != design_receipt_sha256:
        raise ValueError(
            "adaptation receipt is not bound to the current design receipt"
        )
    if adaptation.get("benchmark_receipt_sha256") != benchmark_receipt_sha256:
        raise ValueError(
            "adaptation receipt is not bound to the current benchmark receipt"
        )


def _receipt_inputs(run: Path) -> dict[str, str]:
    """Validate both C1 receipts and their binding, returning hashed inputs."""
    run = Path(run).resolve()
    benchmark_receipt_path = run / "benchmark.complete.json"
    adaptation_receipt_path = run / "adaptation.complete.json"
    design_receipt_path = run / "design.complete.json"
    benchmark = verify_receipt(benchmark_receipt_path)
    adaptation = verify_receipt(adaptation_receipt_path)
    design = verify_receipt(design_receipt_path)
    validate_receipt_bindings(
        benchmark,
        adaptation,
        design,
        benchmark_receipt_sha256=sha256(benchmark_receipt_path),
        design_receipt_sha256=sha256(design_receipt_path),
    )
    return {
        str(path): sha256(path)
        for path in (
            benchmark_receipt_path,
            adaptation_receipt_path,
            design_receipt_path,
            run / "benchmark.jsonl",
            run / "adaptation.jsonl",
            run / "design.plan.json",
        )
    }


def _validate_rows(
    rows: Sequence[Mapping], *, adaptation: bool
) -> dict[str, list[Mapping]]:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    allowed = {"adaptation_train"} if adaptation else {"dev"}
    for row in rows:
        if row.get("partition") not in allowed:
            raise ValueError(
                f"forbidden partition in {'adaptation' if adaptation else 'dev'} rows"
            )
        if row.get("benchmark_schema") != 2:
            raise ValueError("row is not benchmark schema v2")
        if not all(
            isinstance(row.get(field), str) and row[field]
            for field in ("position_group", "factor_cell", "parity")
        ):
            raise ValueError("row lacks position_group, factor_cell, or parity")
        if not isinstance(row.get("source_sha256"), dict) or not row["source_sha256"]:
            raise ValueError("row lacks renderer source hashes")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", str(v))
            for v in row["source_sha256"].values()
        ):
            raise ValueError("row has an invalid renderer source hash")
        groups[str(row["position_group"])].append(row)
    for group, views in groups.items():
        cells: dict[str, list[Mapping]] = defaultdict(list)
        for row in views:
            cells[str(row.get("factor_cell"))].append(row)
        expected_cells = 1 if adaptation else 20
        expected_views = 2 if adaptation else 40
        if len(views) != expected_views or len(cells) != expected_cells:
            raise ValueError(
                f"position group {group} must have {expected_cells} factor cell(s) "
                f"and {expected_views} views"
            )
        for cell, pair in cells.items():
            if len(pair) != 2 or {row.get("parity") for row in pair} != set(PARITIES):
                raise ValueError(
                    f"position group {group}, factor cell {cell} lacks one parity twin"
                )
    return groups


def assert_cross_population_disjoint(adaptation_groups, dev_groups) -> None:
    def keys(groups, label):
        result = set()
        for group, views in groups.items():
            observed = {sym_key(row["fen"]) for row in views}
            if len(observed) != 1:
                raise ValueError(f"{label} group {group} contains different positions")
            result.update(observed)
        return result

    if keys(adaptation_groups, "adaptation") & keys(dev_groups, "C1"):
        raise ValueError("adaptation and C1 positions overlap under D4")


def load_populations(run: Path):
    """Load adaptation_train and the development firewall, never final-test images."""
    # Keep module import data-only/lightweight; benchmark_v2's renderer imports are
    # needed only when an actual completed population is opened.
    from tools.paper.benchmark_v2 import load_dev

    inputs = _receipt_inputs(run)
    adaptation = read_jsonl(Path(run) / "adaptation.jsonl")
    if len(adaptation) != 16_000:
        raise ValueError(
            f"adaptation freeze must contain 16,000 images, found {len(adaptation)}"
        )
    adaptation_groups = _validate_rows(adaptation, adaptation=True)
    if len(adaptation_groups) != 8_000:
        raise ValueError("adaptation freeze must contain 8,000 position groups")
    dev = load_dev(Path(run) / "benchmark.jsonl")
    if len(dev) != 16_000:
        raise ValueError(
            f"C1 development split must contain 16,000 views, found {len(dev)}"
        )
    dev_groups = _validate_rows(dev, adaptation=False)
    if len(dev_groups) != 400:
        raise ValueError("C1 development split must contain 400 position groups")
    c1_groups = defaultdict(list)
    for row in read_jsonl(Path(run) / "benchmark.jsonl"):
        if row.get("partition") not in {"dev", "final_test"}:
            raise ValueError("unknown C1 partition in benchmark metadata")
        c1_groups[str(row["position_group"])].append(row)
    assert_cross_population_disjoint(adaptation_groups, c1_groups)
    return adaptation, dev, inputs


def select_training_rows(
    rows: Sequence[Mapping], nested_budgets: Mapping[str, Sequence[str]], spec: RunSpec
) -> list[Mapping]:
    """Select a registered matched-position arm from the locked nested prefix."""
    groups = _validate_rows(rows, adaptation=True)
    if set(nested_budgets) != {str(n) for n in BUDGETS}:
        raise ValueError("design plan lacks the exact registered nested budgets")
    prefixes = {n: list(nested_budgets[str(n)]) for n in BUDGETS}
    if any(len(prefixes[n]) != n or len(set(prefixes[n])) != n for n in BUDGETS):
        raise ValueError("nested budget count or uniqueness mismatch")
    if not set(prefixes[500]) <= set(prefixes[2_000]) <= set(prefixes[8_000]):
        raise ValueError("adaptation budgets are not nested")
    if set(prefixes[8_000]) != set(groups):
        raise ValueError("8,000-group prefix differs from adaptation freeze")
    chosen = set(prefixes[spec.budget])
    parities = {"a1_dark"} if spec.arm == "dark" else set(PARITIES)
    selected = [
        row
        for row in rows
        if row["position_group"] in chosen and row["parity"] in parities
    ]
    expected = spec.budget if spec.arm == "dark" else 2 * spec.budget
    if (
        len(selected) != expected
        or {row["position_group"] for row in selected} != chosen
    ):
        raise ValueError("arm selection violates the matched-position budget")
    return selected


def _resolve(run: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(run) / path


def _samples(rows, run: Path, Sample, fen_to_labels):
    return [
        Sample(_resolve(run, row["image"]), fen_to_labels(row["fen"]), str(row["id"]))
        for row in rows
    ]


def _hash_row_inputs(rows: Iterable[Mapping], run: Path) -> dict[str, str]:
    inputs = {}
    for row in rows:
        for field, hash_field in (("image", "sha256"), ("label", "label_sha256")):
            path = _resolve(run, row[field])
            digest = sha256(path)
            if digest != row.get(hash_field):
                raise ValueError(f"{field} hash mismatch for {row.get('id')}")
            inputs[str(path.resolve())] = digest
    return inputs


def verify_hashes(inputs: Mapping[str, str]) -> None:
    for name, digest in inputs.items():
        if sha256(name) != digest:
            raise ValueError(f"run input changed: {name}")


def load_release_predictions(path: Path, current_inputs: Mapping[str, str]):
    """Load only predictions bound to a completed release-baseline receipt."""
    path = Path(path).resolve()
    suffix = ".dev_predictions.jsonl"
    if not str(path).endswith(suffix):
        raise ValueError(f"baseline predictions must end in {suffix}")
    prefix = Path(str(path)[: -len(suffix)])
    receipt_path = Path(f"{prefix}.complete.json")
    receipt = verify_receipt(receipt_path)
    if (
        receipt.get("status") != "c2_complete"
        or receipt.get("mode") != "baseline"
        or receipt.get("base_checkpoint_sha256") != BASE_CHECKPOINT_SHA256
        or receipt.get("accepted") is not True
        or receipt.get("final_test_images_read") is not False
        or receipt.get("scored_partitions") != ["dev", "chessred_test"]
    ):
        raise ValueError("baseline predictions lack a valid C2 release receipt")
    if receipt.get("model_inputs_sha256") != dict(current_inputs):
        raise ValueError(
            "baseline release is not bound to the current C1/source inventory"
        )
    return read_jsonl(path), receipt_path


def chessred_inventory(data_root: Path) -> dict[str, str]:
    """Validate the complete public ChessReD source inventory before evaluation."""
    root = Path(data_root).resolve() / "chessred"
    annotation = root / "annotations.json"
    if sha256(annotation) != CHESSRED_ANNOTATIONS_SHA256:
        raise ValueError("ChessReD annotations SHA-256 mismatch")
    payload = json.loads(annotation.read_text())
    images = payload.get("images", [])
    if len(images) != 10_800:
        raise ValueError(
            f"ChessReD must contain 10,800 source image records, found {len(images)}"
        )
    paths = [root / row["path"] for row in images]
    if len(set(paths)) != 10_800 or any(not path.is_file() for path in paths):
        raise FileNotFoundError("ChessReD source image copy is incomplete")
    # One aggregate digest keeps before/after checks bounded while binding every file.
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return {
        str(annotation): sha256(annotation),
        "chessred_files_inventory": digest.hexdigest(),
    }


def _recheck_chessred(data_root: Path, expected: Mapping[str, str]) -> None:
    if chessred_inventory(data_root) != dict(expected):
        raise ValueError("ChessReD files changed during the run")


def assert_gpu0_idle() -> None:
    """Fail closed when another compute process occupies GPU 0; never kills it."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    others = {
        int(line.strip()) for line in result.stdout.splitlines() if line.strip()
    } - {os.getpid()}
    if others:
        raise RuntimeError(
            f"GPU 0 is in use by other compute process(es): {sorted(others)}"
        )


def _write_progress(path: Path, payload: Mapping) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train_fixed_steps(
    model,
    dataset,
    params,
    *,
    seed: int,
    progress_path: Path,
    workers: int = 4,
) -> dict:
    """Run the fixed final-step recipe: no scheduler, selection, or early stop."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    seed_run(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=RECIPE["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("training population does not contain one full batch")
    optimizer = torch.optim.AdamW(
        params, lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"]
    )
    model.train()
    iterator = iter(loader)
    started = time.monotonic()
    step, final_loss = 0, None
    _write_progress(
        progress_path,
        {
            "status": "training",
            "step": 0,
            "steps": RECIPE["steps"],
            "loss": None,
            "elapsed_seconds": 0.0,
        },
    )
    try:
        for step in range(1, RECIPE["steps"] + 1):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, labels = next(iterator)
            images = images.to("cuda:0", non_blocking=True)
            labels = labels.to("cuda:0", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"non-finite logits at step {step}")
                loss = nn.functional.cross_entropy(
                    logits.flatten(0, 1), labels.flatten()
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            final_loss = float(loss.detach())
            loss.backward()
            missing = [
                index
                for index, parameter in enumerate(params)
                if parameter.grad is None
            ]
            nonfinite = [
                index
                for index, parameter in enumerate(params)
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if missing or nonfinite:
                raise FloatingPointError(
                    f"invalid trainable gradients at step {step}: "
                    f"missing={missing}, nonfinite={nonfinite}"
                )
            optimizer.step()
            if step % 50 == 0 or step == RECIPE["steps"]:
                progress = {
                    "status": "training",
                    "step": step,
                    "steps": RECIPE["steps"],
                    "loss": final_loss,
                    "elapsed_seconds": time.monotonic() - started,
                }
                _write_progress(progress_path, progress)
                print(json.dumps(progress), flush=True)
            if step % 100 == 0:
                assert_gpu0_idle()
    except BaseException as error:
        _write_progress(
            progress_path,
            {
                "status": "execution_failed",
                "step": step,
                "steps": RECIPE["steps"],
                "last_finite_loss": final_loss,
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    model.eval()
    result = {
        "steps_run": RECIPE["steps"],
        "final_step": RECIPE["steps"],
        "final_train_loss": final_loss,
        "train_seconds": time.monotonic() - started,
        "selection_enabled": False,
        "stopped_early": False,
    }
    _write_progress(progress_path, {"status": "training_complete", **result})
    return result


def _fen(labels: Sequence[int]) -> str:
    pieces = ".PNBRQKpnbrqk"
    ranks = []
    for start in range(0, 64, 8):
        rank, empty = "", 0
        for label in labels[start : start + 8]:
            if label == 0:
                empty += 1
            else:
                rank += (str(empty) if empty else "") + pieces[label]
                empty = 0
        ranks.append(rank + (str(empty) if empty else ""))
    return "/".join(ranks)


def _metrics(rows: Sequence[Mapping]) -> dict:
    n = len(rows)
    exact = sum(row["exact"] for row in rows)
    correct = sum(row["squares_correct"] for row in rows)
    return {
        "n": n,
        "exact_boards": exact,
        "exact_pct": 100.0 * exact / n,
        "per_square_pct": 100.0 * correct / (64 * n),
    }


def summarize_predictions(rows: Sequence[Mapping]) -> dict:
    by_cell, by_parity, by_joint = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for row in rows:
        by_cell[row["factor_cell"]].append(row)
        by_parity[row["parity"]].append(row)
        by_joint[(row["factor_cell"], row["parity"])].append(row)
    return {
        "overall": _metrics(rows),
        "by_factor_cell": {
            key: _metrics(value) for key, value in sorted(by_cell.items())
        },
        "by_parity": {key: _metrics(value) for key, value in sorted(by_parity.items())},
        "by_factor_cell_parity": {
            f"{cell}/{parity}": _metrics(value)
            for (cell, parity), value in sorted(by_joint.items())
        },
    }


def evaluate_rows(model, rows, run: Path, data, *, batch_size: int, workers: int):
    """Evaluate C1 dev through the authors' fp32/TF32 path and retain all predictions."""
    import torch
    from torch.utils.data import DataLoader

    torch.set_float32_matmul_precision("high")
    model.eval()
    samples = _samples(rows, run, data.Sample, data.fen_to_labels)
    loader = DataLoader(
        data.BoardDataset(samples, data.eval_transform(RECIPE["resolution"])),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )
    predictions, labels = [], []
    with torch.no_grad():
        for images, expected in loader:
            logits = model(images.to("cuda:0", non_blocking=True))
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    "non-finite logits during C1 development evaluation"
                )
            predictions.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(expected.tolist())
    output = []
    if len(predictions) != len(rows):
        raise ValueError("C1 development prediction count mismatch")
    for row, pred, expected in zip(rows, predictions, labels):
        correct = sum(left == right for left, right in zip(pred, expected))
        output.append(
            {
                "id": row["id"],
                "position_group": row["position_group"],
                "factor_cell": row["factor_cell"],
                "parity": row["parity"],
                "exact": correct == 64,
                "squares_correct": correct,
                "gt": _fen(expected),
                "pred": _fen(pred),
            }
        )
    return output


def evaluate_chessred(
    model, data, *, batch_size: int, workers: int, regression_floor_pct: float
):
    """Public regression guard only; this set is never used for selection."""
    import torch
    from torch.utils.data import DataLoader

    samples = data.load_chessred("test")
    if len(samples) != 2_129:
        raise ValueError(f"ChessReD test must have 2,129 boards, found {len(samples)}")
    loader = DataLoader(
        data.BoardDataset(samples, data.eval_transform(RECIPE["resolution"])),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )
    predictions, labels, rows = [], [], []
    model.eval()
    with torch.no_grad():
        for images, expected in loader:
            logits = model(images.to("cuda:0", non_blocking=True))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite logits during ChessReD evaluation")
            predictions.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(expected.tolist())
    if len(predictions) != len(samples):
        raise ValueError("ChessReD prediction count mismatch")
    for sample, pred, expected in zip(samples, predictions, labels):
        correct = sum(left == right for left, right in zip(pred, expected))
        rows.append(
            {
                "id": str(sample.sample_id),
                "exact": correct == 64,
                "squares_correct": correct,
                "gt": _fen(expected),
                "pred": _fen(pred),
            }
        )
    metrics = _metrics(rows)
    metrics["regression_floor_pct"] = regression_floor_pct
    metrics["regression_pass"] = metrics["exact_pct"] >= regression_floor_pct
    return rows, metrics


def _prediction_index(rows, label: str):
    index = {}
    for row in rows:
        try:
            key = (row["id"], row["factor_cell"], row["parity"])
            required = (
                row["position_group"],
                row["gt"],
                row["exact"],
                row["squares_correct"],
            )
        except KeyError as error:
            raise ValueError(
                f"{label} prediction lacks required field {error}"
            ) from error
        if key in index:
            raise ValueError(f"duplicate {label} prediction key: {key}")
        if (
            not isinstance(required[0], str)
            or not required[0]
            or not isinstance(required[1], str)
            or type(required[2]) is not bool
            or type(required[3]) is not int
            or not 0 <= required[3] <= 64
        ):
            raise ValueError(f"invalid {label} prediction row: {key}")
        index[key] = row
    if not index:
        raise ValueError(f"{label} predictions are empty")
    return index


def _paired_bootstrap(candidate, baseline, *, seed: int, draws: int) -> dict:
    if draws <= 0:
        raise ValueError("draws must be positive")
    grouped = defaultdict(list)
    gained = lost = 0
    for row, old in zip(candidate, baseline):
        gained += int(row["exact"] and not old["exact"])
        lost += int(old["exact"] and not row["exact"])
        grouped[row["position_group"]].append(
            (
                float(row["exact"]) - float(old["exact"]),
                (row["squares_correct"] - old["squares_correct"]) / 64.0,
            )
        )
    deltas = np.array(
        [
            [np.mean([x[0] for x in values]), np.mean([x[1] for x in values])]
            for values in grouped.values()
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, 2), dtype=float)
    for index in range(draws):
        boot[index] = 100.0 * deltas[rng.integers(0, len(deltas), len(deltas))].mean(
            axis=0
        )
    point = 100.0 * deltas.mean(axis=0)
    return {
        "position_groups": len(grouped),
        "resampling": "position_group_unstratified",
        "exact_delta_pp": float(point[0]),
        "square_delta_pp": float(point[1]),
        "exact_ci95_pp": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
        "square_ci95_pp": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
        "gained": gained,
        "lost": lost,
    }


def whole_matrix_bootstrap(
    candidate, baseline, *, seed: int = BOOTSTRAP_SEED, draws: int = 2_000
) -> dict:
    """Bootstrap whole position clusters across the matrix and within each cell/parity."""
    if seed != BOOTSTRAP_SEED:
        raise ValueError(f"bootstrap seed is fixed at {BOOTSTRAP_SEED}")
    candidates = _prediction_index(candidate, "candidate")
    baselines = _prediction_index(baseline, "baseline")
    if set(candidates) != set(baselines):
        raise ValueError("candidate and baseline development inventories differ")
    ordered = sorted(candidates)
    for key in ordered:
        row, old = candidates[key], baselines[key]
        if row["gt"] != old["gt"]:
            raise ValueError(f"candidate and baseline ground truth differs: {key}")
        if row["position_group"] != old["position_group"]:
            raise ValueError(f"candidate and baseline position group differs: {key}")
    result = _paired_bootstrap(
        [candidates[key] for key in ordered],
        [baselines[key] for key in ordered],
        seed=seed,
        draws=draws,
    )
    result["resampling"] = "position_group_across_whole_matrix_unstratified"
    joints = sorted({(key[1], key[2]) for key in ordered})
    result["by_factor_cell_parity"] = {}
    for cell, parity in joints:
        keys = [key for key in ordered if key[1:] == (cell, parity)]
        joint = _paired_bootstrap(
            [candidates[key] for key in keys],
            [baselines[key] for key in keys],
            seed=seed,
            draws=draws,
        )
        joint["resampling"] = "position_group_within_factor_cell_parity_unstratified"
        result["by_factor_cell_parity"][f"{cell}/{parity}"] = joint
    return result


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _save_adapter(path: Path, model, *, train_head: bool) -> list[str]:
    from safetensors.torch import save_file

    tensors = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if name.endswith((".A", ".B")) or (train_head and name.startswith("head."))
    }
    if not tensors or any(
        not (name.endswith((".A", ".B")) or (train_head and name.startswith("head.")))
        for name in tensors
    ):
        raise ValueError(
            "adapter serialization boundary is empty or includes base weights"
        )
    save_file(tensors, str(path))
    return sorted(tensors)


def run(args) -> None:
    """Execute one baseline or one registered adapter run."""
    if socket.gethostname().split(".")[0] != "render-host":
        raise RuntimeError("live C2 runs are restricted to host render-host")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("set CUDA_VISIBLE_DEVICES=0 before starting a live C2 run")
    run_root, cq_repo, data_root = map(Path, (args.run, args.cq_repo, args.data_root))
    source_inputs = validate_chessqueries_source(cq_repo)
    adaptation, dev, input_hashes = load_populations(run_root)
    chessred_inputs = chessred_inventory(data_root)
    input_hashes.update(source_inputs)
    input_hashes.update(harness_source_inputs())
    input_hashes[str(Path(args.checkpoint).resolve())] = sha256(args.checkpoint)
    if input_hashes[str(Path(args.checkpoint).resolve())] != BASE_CHECKPOINT_SHA256:
        raise ValueError("frozen base checkpoint SHA-256 mismatch")
    input_hashes.update(_hash_row_inputs(dev, run_root))

    selected = []
    spec = None
    if args.mode == "train":
        spec = RunSpec(args.budget, args.arm, args.seed, args.train_head)
        baseline_path = Path(args.baseline_predictions).resolve()
        baseline, baseline_receipt = load_release_predictions(
            baseline_path, input_hashes
        )
        input_hashes[str(baseline_path)] = sha256(baseline_path)
        input_hashes[str(baseline_receipt)] = sha256(baseline_receipt)
        plan = json.loads((run_root / "design.plan.json").read_text())
        selected = select_training_rows(adaptation, plan["nested_budgets"], spec)
        input_hashes.update(_hash_row_inputs(selected, run_root))
    reserve_output(args.out)
    assert_gpu0_idle()
    data, evaluation, lora = load_upstream(cq_repo, data_root)
    import torch

    torch.set_float32_matmul_precision("high")
    # The pinned LoRA injector creates its tensors on CPU.  Configure the complete
    # model there, then move once and rebuild the optimizer parameter list.
    model = evaluation.load_model(Path(args.checkpoint), "cpu")
    fit = None
    adapter_names = []
    if spec is not None:
        seed_run(spec.seed)  # LoRA A initialization occurs inside configure_lora.
        params = configure_lora(
            model,
            lora.inject_lora,
            train_head=spec.train_head,
            enforce_vit_l_contract=True,
        )
        params = move_model_and_rebind(model, "cuda:0")
        current_trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if len(params) != len(current_trainable) or any(
            left is not right for left, right in zip(params, current_trainable)
        ):
            raise AssertionError(
                "optimizer list does not reference current model parameters"
            )
        dataset = data.BoardDataset(
            _samples(selected, run_root, data.Sample, data.fen_to_labels),
            data.eval_transform(RECIPE["resolution"]),
        )
        try:
            fit = train_fixed_steps(
                model,
                dataset,
                params,
                seed=spec.seed,
                progress_path=Path(f"{args.out}.training_progress.json"),
                workers=args.workers,
            )
        except BaseException:
            try:
                _save_adapter(
                    Path(f"{args.out}.partial.adapter.safetensors"),
                    model,
                    train_head=spec.train_head,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as snapshot_error:
                print(
                    json.dumps(
                        {
                            "partial_adapter_snapshot_failed": type(
                                snapshot_error
                            ).__name__,
                            "error": str(snapshot_error),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            raise
        fit.update(
            {
                "matched_position_groups": spec.budget,
                "distinct_training_images": len(selected),
                "trainable_parameters": sum(parameter.numel() for parameter in params),
                "lora_tensors": VIT_L_LORA_TENSORS,
                "lora_parameters": VIT_L_LORA_PARAMETERS,
                "head_parameters": QUERY_HEAD_PARAMETERS if spec.train_head else 0,
                "examples_seen": RECIPE["steps"] * RECIPE["batch_size"],
                "equivalent_dataset_passes": (
                    RECIPE["steps"] * RECIPE["batch_size"] / len(selected)
                ),
            }
        )
        # Preserve the fixed final-step checkpoint before either long evaluation.
        adapter_names = _save_adapter(
            Path(f"{args.out}.adapter.safetensors"), model, train_head=spec.train_head
        )
        assert_gpu0_idle()
    else:
        prepare_baseline_for_evaluation(model, "cuda:0")

    predictions = evaluate_rows(
        model,
        dev,
        run_root,
        data,
        batch_size=RECIPE["eval_batch_size"],
        workers=args.workers,
    )
    assert_gpu0_idle()
    public_floor = 99.12
    public_predictions, public_metrics = evaluate_chessred(
        model,
        data,
        batch_size=RECIPE["eval_batch_size"],
        workers=args.workers,
        regression_floor_pct=public_floor,
    )
    assert_gpu0_idle()
    _write_jsonl(Path(f"{args.out}.dev_predictions.jsonl"), predictions)
    _write_jsonl(
        Path(f"{args.out}.chessred_test_predictions.jsonl"), public_predictions
    )
    metrics = {
        "dev": summarize_predictions(predictions),
        "chessred_test": public_metrics,
    }
    if spec is not None:
        metrics["paired_vs_release"] = whole_matrix_bootstrap(
            predictions, baseline, seed=BOOTSTRAP_SEED
        )
    Path(f"{args.out}.metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    verify_hashes(input_hashes)
    if validate_chessqueries_source(cq_repo) != source_inputs:
        raise ValueError("ChessQueries source inventory changed during the run")
    _recheck_chessred(data_root, chessred_inputs)
    finish_output(
        args.out,
        {
            "status": "c2_complete",
            "mode": args.mode,
            "release_ready": False,
            "c1_root": str(run_root.resolve()),
            "chessqueries_revision": CQ_REVISION,
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "recipe": RECIPE,
            "run_spec": spec.__dict__ if spec else None,
            "fit": fit,
            "adapter_tensor_names": adapter_names,
            "model_inputs_sha256": input_hashes,
            "public_data_inputs_sha256": chessred_inputs,
            "public_regression_pass": public_metrics["regression_pass"],
            "accepted": public_metrics["regression_pass"],
            "final_test_images_read": False,
            "scored_partitions": ["dev", "chessred_test"],
        },
    )
    print(json.dumps({"complete": f"{args.out}.complete.json"}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("baseline", "train"))
    result.add_argument("--run", type=Path, required=True, help="completed C1 root")
    result.add_argument("--cq-repo", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--budget", type=int, choices=BUDGETS)
    result.add_argument("--arm", choices=ARMS)
    result.add_argument("--seed", type=int)
    head = result.add_mutually_exclusive_group()
    head.add_argument("--train-head", dest="train_head", action="store_true")
    head.add_argument("--freeze-head", dest="train_head", action="store_false")
    result.set_defaults(train_head=None)
    result.add_argument("--baseline-predictions", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.mode == "train":
        missing = [
            name
            for name in ("budget", "arm", "seed", "train_head", "baseline_predictions")
            if getattr(args, name) is None
        ]
        if missing:
            parser().error("train requires explicit " + ", ".join(missing))
    elif any(
        getattr(args, name) is not None
        for name in ("budget", "arm", "seed", "train_head", "baseline_predictions")
    ):
        parser().error("baseline does not accept training-only arguments")
    run(args)


if __name__ == "__main__":
    main()
