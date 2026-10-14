"""Fixed development/confirmation study; docs/TRANSFER_STUDY_2026-09-09.md.

The encoder is computed once for the two inference hypotheses. Adaptation freezes
encoder/decoder and changes only explicitly audited query/norm/head parameters.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .adaptation_metrics import (
        choose_adapter,
        choose_rule,
        confirm_pass,
        paired_summary,
        score_predictions,
    )
    from .benchmark_io import finish_output, read_jsonl, reserve_output, sha256
    from .fen_pools import cells, labels_to_fen
    from .parity_probe import CHECKPOINT_SHA, gpu_idle
    from .parity_rules import predict_rules
    from .transfer_cohort import validate_confirmation, verify_receipt
except ImportError:
    from adaptation_metrics import (
        choose_adapter,
        choose_rule,
        confirm_pass,
        paired_summary,
        score_predictions,
    )
    from benchmark_io import finish_output, read_jsonl, reserve_output, sha256
    from fen_pools import cells, labels_to_fen
    from parity_probe import CHECKPOINT_SHA, gpu_idle
    from parity_rules import predict_rules
    from transfer_cohort import validate_confirmation, verify_receipt

SEEDS = (303, 304, 305)
PREFIXES = ("queries.", "norm.", "head.")
CLASSES = ".PNBRQKpnbrqk"


def setup(run):
    verify_receipt(run / "prepare.complete.json")
    plan = json.loads((run / "prepare.json").read_text())
    if sha256(plan["checkpoint"]) != CHECKPOINT_SHA:
        raise ValueError("base checkpoint changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("use CUDA_VISIBLE_DEVICES=0 on the idle render-host GPU only")
    gpu_idle()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    sys.path.insert(0, plan["cq_repo"])
    from eval import load_model

    from data import eval_transform

    model = load_model(Path(plan["checkpoint"]), "cuda")
    model.eval().requires_grad_(False)
    initial = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.startswith(PREFIXES)
    }
    return plan, model, initial, eval_transform(644)


def provenance(run, plan):
    paths = [
        Path(__file__),
        *(
            Path(__file__).with_name(name)
            for name in (
                "parity_rules.py",
                "adaptation_metrics.py",
                "transfer_cohort.py",
                "benchmark_io.py",
                "parity_probe.py",
                "fen_pools.py",
            )
        ),
    ]
    paths += sorted(Path(plan["cq_repo"]).glob("*.py"))
    return dict(
        base_checkpoint_sha256=CHECKPOINT_SHA,
        prepare_sha256=sha256(run / "prepare.json"),
        records_sha256=sha256(run / "prepare.records.jsonl"),
        source_sha256={str(path.resolve()): sha256(path) for path in paths},
    )


def check_sources(identity):
    for path, digest in identity["source_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"source changed during study: {path}")


def write_partial(path, evidence):
    """Atomic, no-clobber partial evidence; never a completion receipt."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(evidence, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def labels_for(records):
    return np.array(
        [[CLASSES.index(c) for c in cells(row["fen"])] for row in records],
        dtype=np.int64,
    )


def load_images(records, transform):
    # Keep torchvision an execution-only dependency.  The CPU contract tests
    # exercise the adapter helpers without installing the image stack.
    from torchvision.io import read_image

    for row in records:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError(f"image changed: {row['key']}")
    return torch.stack(
        [transform(read_image(row["image"]).float()) for row in records]
    ).to("cuda")


def hidden(model, memory):
    return model.decoder(model.queries(memory.shape[0]), memory)


def decode(model, memory):
    return model.head(model.norm(hidden(model, memory)))


def restore(model, initial, adapted=None):
    state = dict(model.named_parameters())
    if adapted is not None and not set(adapted).issubset(initial):
        raise ValueError("adapter contains parameters outside the allowed boundary")
    with torch.no_grad():
        for name, value in initial.items():
            state[name].copy_(value)
        for name, value in (adapted or {}).items():
            if tuple(state[name].shape) != tuple(value.shape):
                raise ValueError(f"adapter shape mismatch: {name}")
            state[name].copy_(value.to(state[name].device))


def dual_decode(model, memory):
    """Inference only; never mutate embedding indices between forward/backward."""
    with torch.no_grad():
        color = model.queries.color_idx
        saved = color.clone()
        try:
            first = decode(model, memory)
            color.copy_(1 - saved)
            second = decode(model, memory)
        finally:
            color.copy_(saved)
    return torch.stack((first, second), dim=1)


def configure_training(model, kind):
    if kind not in ("head", "query"):
        raise ValueError("unknown adapter kind")
    model.eval().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    names = []
    query, other = [], []
    for name, parameter in model.named_parameters():
        allowed = name.startswith(("norm.", "head.")) or (
            kind == "query" and name.startswith("queries.")
        )
        if allowed:
            parameter.requires_grad_(True)
            names.append(name)
            (query if name.startswith("queries.") else other).append(parameter)
    expected = 15373 if kind == "head" else 99341
    count = sum(p.numel() for p in query + other)
    if count != expected:
        raise ValueError(
            f"unexpected trainable parameter count: {count}, expected {expected}"
        )
    groups = [{"params": other, "lr": 1e-4}]
    if query:
        groups.append({"params": query, "lr": 1e-3})
    return names, query + other, groups, count


def predict_cached(model, features, hidden_base, indices, kind):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), 4):
            idx = indices[start : start + 4]
            logits = (
                model.head(model.norm(hidden_base[idx]))
                if kind == "head"
                else decode(model, features[idx])
            )
            chunks.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(chunks)


def train_one(
    model,
    initial,
    features,
    hidden_base,
    labels,
    records,
    kind,
    seed,
    run,
    deadline,
    baseline_validation,
    progress,
):
    from safetensors.torch import save_file

    restore(model, initial)
    torch.manual_seed(seed)
    names, parameters, groups, count = configure_training(model, kind)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    train = np.array(
        [i for i, row in enumerate(records) if row["partition"] == "train"]
    )
    valid = np.array(
        [i for i, row in enumerate(records) if row["partition"] == "validation"]
    )
    truth = labels_for(records)
    valid_records = [records[i] for i in valid]
    rng = np.random.default_rng(seed)
    order, cursor = rng.permutation(train), 0
    best_rank = None
    best_state = None
    history = []
    best_step = 0
    best_score = None
    progress["current"] = dict(kind=kind, seed=seed, history=history, step=0)
    for step in range(101):
        if time.monotonic() > deadline:
            raise RuntimeError(
                "20-minute training cap reached; no silent budget extension"
            )
        loss_value = None
        if step:
            if cursor + 8 > len(order):
                order, cursor = rng.permutation(train), 0
            idx = order[cursor : cursor + 8]
            cursor += 8
            optimizer.zero_grad(set_to_none=True)
            logits = (
                model.head(model.norm(hidden_base[idx]))
                if kind == "head"
                else decode(model, features[idx])
            )
            loss = F.cross_entropy(logits.reshape(-1, 13), labels[idx].reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite adaptation loss")
            loss.backward()
            if any(
                p.grad is not None
                for name, p in model.named_parameters()
                if name.startswith(("encoder.", "decoder."))
            ):
                raise ValueError("frozen encoder/decoder received gradients")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            progress["optimizer_updates_completed"] += 1
            progress["current"]["step"] = step
            loss_value = float(loss.detach())
        if step % 25 == 0:
            gpu_idle(os.getpid())
            predicted = predict_cached(model, features, hidden_base, valid, kind)
            if step == 0 and not np.array_equal(predicted, baseline_validation):
                raise ValueError(
                    "step-0 validation differs from the locked release baseline"
                )
            score = score_predictions(predicted, truth[valid], valid_records)
            rank = (
                score["overall"]["exact_pct"],
                score["overall"]["per_square_pct"],
                -step,
            )
            history.append(dict(step=step, loss=loss_value, validation=score))
            if best_rank is None or rank > best_rank:
                best_rank, best_step, best_score = rank, step, score
                state = model.state_dict()
                best_state = {
                    name: state[name].detach().cpu().contiguous().clone()
                    for name in names
                }
            progress["current"].update(best_step=best_step, best_validation=best_score)
            print(
                json.dumps(
                    dict(
                        stage="training",
                        kind=kind,
                        seed=seed,
                        step=step,
                        loss=loss_value,
                        validation=score["overall"],
                    )
                ),
                flush=True,
            )
    name = f"{kind}_seed{seed}"
    path = run / "adapters" / f"{name}.safetensors"
    if path.exists():
        raise FileExistsError(path)
    save_file(best_state, str(path))
    return dict(
        name=name,
        kind=kind,
        seed=seed,
        parameter_count=count,
        trainable_names=names,
        best_step=best_step,
        validation=best_score,
        history=history,
        path=str(path.resolve()),
        sha256=sha256(path),
    )


def develop(run):
    prefix = run / "development"
    reserve_output(prefix)
    plan, model, initial, transform = setup(run)
    identity = provenance(run, plan)
    records = [
        r
        for r in read_jsonl(run / "prepare.records.jsonl")
        if r["partition"] != "confirmation"
    ]
    if len(records) != 384:
        raise ValueError("expected 384 development images")
    truth = labels_for(records)
    features, hidden_base, dual = [], [], []
    max_forward_error = 0.0
    # Normal tensors under no_grad: inference tensors cannot be saved for backward.
    with torch.no_grad():
        for start in range(0, len(records), 4):
            batch = records[start : start + 4]
            images = load_images(batch, transform)
            memory = model.encoder.forward_features(images)
            if memory.dtype != torch.float32 or torch.is_inference(memory):
                raise ValueError("encoder cache must be normal FP32 tensors")
            features.append(memory.detach())
            hidden_base.append(hidden(model, memory).detach())
            dual.append(dual_decode(model, memory).cpu().numpy())
            if start % 64 == 0:
                gpu_idle(os.getpid())
                print(
                    json.dumps(dict(stage="feature_cache", images=start + len(batch))),
                    flush=True,
                )
    features = torch.cat(features)
    hidden_base = torch.cat(hidden_base)
    if hidden_base.dtype != torch.float32 or torch.is_inference(hidden_base):
        raise ValueError("decoder cache must be normal FP32 tensors")
    dual = np.concatenate(dual)
    predicted, diagnostics = predict_rules(dual)
    valid = np.array(
        [i for i, row in enumerate(records) if row["partition"] == "validation"]
    )
    valid_records = [records[i] for i in valid]
    with torch.no_grad():
        for start in range(0, len(valid), 4):
            idx = valid[start : start + 4]
            original = model(load_images([records[i] for i in idx], transform))
            cached = decode(model, features[idx])
            max_forward_error = max(
                max_forward_error, float((original - cached).abs().max())
            )
            if max_forward_error > 1e-5 or not torch.equal(
                original.argmax(-1), cached.argmax(-1)
            ):
                raise ValueError(
                    "shared-encoder path differs from original validation forward"
                )
    rule_scores = {
        name: score_predictions(values[valid], truth[valid], valid_records)
        for name, values in predicted.items()
    }
    selected_rule = choose_rule(rule_scores)
    print(
        json.dumps(
            dict(
                stage="rule_selection",
                selected=selected_rule,
                scores={k: v["overall"] for k, v in rule_scores.items()},
            )
        ),
        flush=True,
    )
    labels = torch.from_numpy(truth).to("cuda")
    (run / "adapters").mkdir()
    candidates = []
    deadline = time.monotonic() + 20 * 60
    started = time.monotonic()
    progress = dict(
        completed_candidates=candidates, current=None, optimizer_updates_completed=0
    )
    try:
        for kind in ("head", "query"):
            for seed in SEEDS:
                candidates.append(
                    train_one(
                        model,
                        initial,
                        features,
                        hidden_base,
                        labels,
                        records,
                        kind,
                        seed,
                        run,
                        deadline,
                        predicted["default"][valid],
                        progress,
                    )
                )
    except (Exception, KeyboardInterrupt) as error:
        write_partial(
            Path(f"{prefix}.partial.json"),
            dict(
                status="partial_not_eligible_for_confirmation",
                error=str(error),
                elapsed_seconds=time.monotonic() - started,
                identity=identity,
                **progress,
            ),
        )
        raise
    adapter_scores = {
        "default": rule_scores["default"],
        **{r["name"]: r["validation"] for r in candidates},
    }
    selected_adapter = choose_adapter(
        adapter_scores, {r["name"]: r["parameter_count"] for r in candidates}
    )
    selection = dict(
        selected_rule=selected_rule,
        selected_adapter=selected_adapter,
        rule_scores=rule_scores,
        adapters=candidates,
        identity=identity,
        records_sha256=sha256(run / "prepare.records.jsonl"),
        train_positions=128,
        validation_positions=64,
        shared_forward_max_abs_error=max_forward_error,
        optimizer_updates=600,
        training_wall_seconds=time.monotonic() - started,
        baseline_image_encodes=384,
        forward_equivalence_image_encodes=len(valid),
    )
    check_sources(identity)
    for row in records:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("development image changed")
    if sha256(plan["checkpoint"]) != CHECKPOINT_SHA:
        raise ValueError("base checkpoint changed")
    np.savez_compressed(
        f"{prefix}.logits.npz",
        logits=dual,
        labels=truth,
        confidence_choice=diagnostics["confidence_choice"],
        structure_choice=diagnostics["structure_choice"],
    )
    with Path(f"{prefix}.selection.json").open("x") as stream:
        json.dump(selection, stream, indent=2, allow_nan=False)
    finish_output(
        prefix,
        dict(
            status="selection_locked_before_confirmation",
            selected_rule=selected_rule,
            selected_adapter=selected_adapter,
            **identity,
        ),
    )
    print(
        json.dumps(
            dict(
                stage="locked",
                selected_adapter=selected_adapter,
                selected_rule=selected_rule,
            )
        ),
        flush=True,
    )


def confirm(run):
    from safetensors.torch import load_file

    verify_receipt(run / "development.complete.json")
    verify_receipt(run / "render_validation.complete.json")
    revalidated = validate_confirmation(run, write=False)
    selection = json.loads((run / "development.selection.json").read_text())
    check_sources(selection["identity"])
    if selection["records_sha256"] != sha256(run / "prepare.records.jsonl"):
        raise ValueError("cohort changed after selection")
    prefix = run / "confirmation"
    reserve_output(prefix)
    plan, model, initial, transform = setup(run)
    records = read_jsonl(run / "render_validation.records.jsonl")
    if revalidated != records:
        raise ValueError("confirmation render records changed after validation")
    if len(records) != 576 or len({r["group_id"] for r in records}) != 288:
        raise ValueError("confirmation population mismatch")
    truth = labels_for(records)
    adapters = {}
    for candidate in selection["adapters"]:
        if sha256(candidate["path"]) != candidate["sha256"]:
            raise ValueError("selected checkpoint changed")
        adapters[candidate["name"]] = load_file(candidate["path"])
    all_dual = []
    predictions = {name: [] for name in adapters}
    with torch.no_grad():
        for start in range(0, len(records), 4):
            batch = records[start : start + 4]
            restore(model, initial)
            memory = model.encoder.forward_features(load_images(batch, transform))
            base_hidden = hidden(model, memory)
            all_dual.append(dual_decode(model, memory).cpu().numpy())
            for candidate in selection["adapters"]:
                restore(model, initial, adapters[candidate["name"]])
                logits = (
                    model.head(model.norm(base_hidden))
                    if candidate["kind"] == "head"
                    else decode(model, memory)
                )
                predictions[candidate["name"]].append(logits.argmax(-1).cpu().numpy())
            if start % 64 == 0:
                gpu_idle(os.getpid())
                print(
                    json.dumps(
                        dict(stage="confirmation_inference", images=start + len(batch))
                    ),
                    flush=True,
                )
    restore(model, initial)
    all_dual = np.concatenate(all_dual)
    rules, diagnostics = predict_rules(all_dual)
    predictions = {name: np.concatenate(values) for name, values in predictions.items()}
    predictions.update(rules)
    scores = {
        name: score_predictions(values, truth, records)
        for name, values in predictions.items()
    }
    effects = {
        name: paired_summary(values, predictions["default"], truth, records)
        for name, values in predictions.items()
        if name != "default"
    }
    decisions = {}
    for kind in ("rule", "adapter"):
        chosen = selection[f"selected_{kind}"]
        decisions[kind] = dict(
            selected=chosen, passed=False, reasons=["no_development_candidate"]
        )
        if chosen is not None:
            decisions[kind] = dict(
                selected=chosen,
                **confirm_pass(
                    kind, scores[chosen], scores["default"], effects[chosen]
                ),
            )
    wrong_default = np.sum(predictions["default"] != truth, axis=1)
    wrong_flipped = np.sum(predictions["flipped"] != truth, axis=1)
    oracle_whole = np.where(
        (wrong_flipped < wrong_default)[:, None],
        predictions["flipped"],
        predictions["default"],
    )
    oracle_square = np.where(
        predictions["default"] == truth, predictions["default"], predictions["flipped"]
    )
    cross_scores = {}
    for family in dict.fromkeys(r["family"] for r in records):
        cross_scores[family] = {}
        for condition in ("control", "parity"):
            idx = np.array(
                [
                    i
                    for i, r in enumerate(records)
                    if r["family"] == family and r["condition"] == condition
                ]
            )
            subset = [records[i] for i in idx]
            cross_scores[family][condition] = {
                name: score_predictions(values[idx], truth[idx], subset)["overall"]
                for name, values in predictions.items()
            }
    summary = dict(
        scores=scores,
        effects=effects,
        decisions=decisions,
        by_family_condition=cross_scores,
        oracle_bounds=dict(
            scope="hard selection between two argmax hypotheses only; not deployable",
            whole_hypothesis=score_predictions(oracle_whole, truth, records),
            per_square=score_predictions(oracle_square, truth, records),
        ),
        records=576,
        positions=288,
        selection_sha256=sha256(run / "development.selection.json"),
        confirmation_image_encodes=576,
        hypotheses_per_image=2,
        no_ground_truth_at_inference=True,
    )
    with Path(f"{prefix}.predictions.jsonl").open("x") as stream:
        for index, row in enumerate(records):
            for name, values in predictions.items():
                stream.write(
                    json.dumps(
                        dict(
                            key=row["key"],
                            group_id=row["group_id"],
                            family=row["family"],
                            condition=row["condition"],
                            model=name,
                            gt=row["fen"],
                            pred=labels_to_fen(values[index]),
                        )
                    )
                    + "\n"
                )
    np.savez_compressed(
        f"{prefix}.logits.npz",
        logits=all_dual,
        labels=truth,
        confidence_choice=diagnostics["confidence_choice"],
        structure_choice=diagnostics["structure_choice"],
    )
    with Path(f"{prefix}.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    check_sources(selection["identity"])
    for row in records:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("confirmation image changed")
    if sha256(plan["checkpoint"]) != CHECKPOINT_SHA:
        raise ValueError("base checkpoint changed")
    finish_output(
        prefix,
        dict(
            status="fixed_confirmation_complete",
            decisions=decisions,
            selection_sha256=summary["selection_sha256"],
            **provenance(run, plan),
        ),
    )
    print(
        json.dumps(
            dict(
                stage="confirmation_complete",
                decisions=decisions,
                scores={name: value["overall"] for name, value in scores.items()},
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("develop", "confirm"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    (develop if args.mode == "develop" else confirm)(args.run)


if __name__ == "__main__":
    main()
