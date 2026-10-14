"""One-shot fresh-position follow-up; docs/GUARDED_FOLLOWUP_2026-09-09.md."""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np

try:
    from .adaptation_metrics import confirm_pass, paired_summary, score_predictions
    from .benchmark_io import (
        finish_output,
        preflight,
        read_jsonl,
        reserve_output,
        sha256,
        split_rows,
    )
    from .fen_pools import cells, labels_to_fen, sym_key
    from .parity_rules import (
        PIECE_SYMBOLS,
        _structure_violations,
        predict_guarded,
        predict_rules,
    )
    from .transfer_cohort import FAMILIES, record, validate_confirmation, verify_receipt
except ImportError:
    from adaptation_metrics import confirm_pass, paired_summary, score_predictions
    from benchmark_io import (
        finish_output,
        preflight,
        read_jsonl,
        reserve_output,
        sha256,
        split_rows,
    )
    from fen_pools import cells, labels_to_fen, sym_key
    from parity_rules import (
        PIECE_SYMBOLS,
        _structure_violations,
        predict_guarded,
        predict_rules,
    )
    from transfer_cohort import FAMILIES, record, validate_confirmation, verify_receipt


def sample_fresh(available, excluded, family_index):
    """Fixed sampling without model outputs; excluded contains canonical sym_key values."""
    if len({row["id"] for row in available}) != len(available):
        raise ValueError("duplicate source IDs")
    rows = sorted(
        [row for row in available if sym_key(row["fen"]) not in excluded],
        key=lambda row: row["id"],
    )
    rng = random.Random(20260911 + family_index)
    chosen = []
    for quadrant in range(4):
        group = [
            r for r in rows if quadrant * 90 <= r["azimuth_deg"] < (quadrant + 1) * 90
        ]
        if len(group) < 6:
            raise ValueError("insufficient fresh positions in azimuth stratum")
        chosen.extend(rng.sample(group, 6))
    if len({sym_key(r["fen"]) for r in chosen}) != 24:
        raise ValueError("duplicate or D4-equivalent fresh position")
    return chosen


def truth_for(records):
    truth = np.array(
        [[PIECE_SYMBOLS.index(c) for c in cells(r["fen"])] for r in records]
    )
    if np.any(_structure_violations(truth[:, None, :]) != 0):
        raise ValueError("truth support condition failed; abort entire matrix")
    return truth


def validate_population(records, excluded):
    if len(records) != 288 or len({r["group_id"] for r in records}) != 144:
        raise ValueError("expected 144 paired positions")
    groups = {}
    for row in records:
        key = sym_key(row["fen"])
        if key in excluded:
            raise ValueError("position overlaps prior study")
        groups.setdefault(key, []).append(row)
    if len(groups) != 144:
        raise ValueError("D4 duplicate in cohort")
    for rows in groups.values():
        if len(rows) != 2 or {r["condition"] for r in rows} != {"control", "parity"}:
            raise ValueError("each position must have exactly two parity views")
        if len({(r["fen"], r["family"], r["group_id"]) for r in rows}) != 1:
            raise ValueError("paired identity mismatch")
    if {f: sum(r["family"] == f for r in records) for f in FAMILIES} != dict.fromkeys(
        FAMILIES, 48
    ):
        raise ValueError("family population mismatch")
    truth_for(records)


def code_identity():
    return {
        str(path.resolve()): sha256(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def check_code(identity):
    for path, digest in identity.items():
        if sha256(path) != digest:
            raise ValueError(f"study code changed: {path}")


def prepare(run, previous, freeze):
    verify_receipt(previous / "prepare.complete.json")
    development = verify_receipt(previous / "development.complete.json")
    confirmation = verify_receipt(previous / "confirmation.complete.json")
    expected_decisions = {
        "rule": {"selected": "structure", "passed": False, "reasons": ["exact_loss"]},
        "adapter": {
            "selected": None,
            "passed": False,
            "reasons": ["no_development_candidate"],
        },
    }
    if (
        confirmation["status"] != "fixed_confirmation_complete"
        or confirmation["decisions"] != expected_decisions
        or confirmation["prepare_sha256"] != sha256(previous / "prepare.json")
        or confirmation["records_sha256"] != sha256(previous / "prepare.records.jsonl")
        or confirmation["selection_sha256"]
        != sha256(previous / "development.selection.json")
    ):
        raise ValueError("follow-up requires the completed, unchanged failed trial")
    old = json.loads((previous / "prepare.json").read_text())
    upstream = {
        str(p): development["source_sha256"][str(p)]
        for p in sorted(Path(old["cq_repo"]).glob("*.py"))
    }
    if not upstream:
        raise ValueError("missing upstream source snapshot")
    check_code(upstream)
    prior_records = read_jsonl(previous / "prepare.records.jsonl")
    excluded = {sym_key(r["fen"]) for r in prior_records}
    if len(excluded) != 480:
        raise ValueError("expected all 480 prior positions")
    source_root = Path(old["source_root"])
    inventory = preflight(source_root, freeze, [f"bench_geom_{f}" for f in FAMILIES])
    if inventory["freeze_sha256"] != old["freeze_sha256"]:
        raise ValueError("source freeze changed")
    for name, digest in old["source_sha256"].items():
        if sha256(previous / "src/rendering" / name) != digest:
            raise ValueError(f"previous renderer source changed: {name}")
    prefix = run / "prepare"
    reserve_output(prefix)
    shutil.copytree(previous / "src/rendering", run / "src/rendering")
    (run / "manifests").mkdir()
    records, families = [], {}
    for index, family in enumerate(sorted(FAMILIES)):
        split = f"bench_geom_{family}"
        metadata = {r["id"]: r for r in split_rows(source_root / split)}
        chosen = sample_fresh(
            [r for r in inventory["rows"] if r["split"] == split], excluded, index
        )
        manifest = []
        for frozen in chosen:
            row = dict(
                metadata[frozen["id"]],
                probe_family=family,
                probe_original_sha256=frozen["sha256"],
                probe_source_split=split,
            )
            manifest.append(row)
            for condition in ("control", "parity"):
                records.append(
                    record(
                        row,
                        run / f"renders/{family}/{condition}/images/{row['id']}.png",
                        "confirmation",
                        family,
                        condition,
                    )
                )
        with (run / "manifests" / f"{family}.jsonl").open("x") as stream:
            stream.writelines(json.dumps(r) + "\n" for r in manifest)
        families[family] = dict(old["families"][family], n=24)
    validate_population(records, excluded)
    plan = dict(
        old,
        previous_run=str(previous.resolve()),
        families=families,
        counts={"confirmation": 288},
        manifest_sha256={
            p.name: sha256(p) for p in sorted((run / "manifests").glob("*.jsonl"))
        },
        previous_records_sha256=sha256(previous / "prepare.records.jsonl"),
        previous_confirmation_sha256=sha256(previous / "confirmation.complete.json"),
        study_code_sha256=code_identity(),
        upstream_source_sha256=upstream,
        seed=20260911,
        bootstrap_draws=2000,
        excluded_d4_positions=len(excluded),
    )
    with Path(f"{prefix}.records.jsonl").open("x") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in records)
    with Path(f"{prefix}.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
    finish_output(
        prefix, {"status": "fresh_guard_cohort_locked", "positions": 144, "images": 288}
    )
    print(
        json.dumps({"prepared_positions": 144, "truth_support_pass": True}), flush=True
    )


def validate(run, write=True):
    plan = json.loads((run / "prepare.json").read_text())
    check_code(plan["study_code_sha256"])
    check_code(plan["upstream_source_sha256"])
    for name, digest in plan["manifest_sha256"].items():
        if sha256(run / "manifests" / name) != digest:
            raise ValueError(f"frozen manifest changed: {name}")
    prior_path = Path(plan["previous_run"]) / "prepare.records.jsonl"
    if sha256(prior_path) != plan["previous_records_sha256"]:
        raise ValueError("prior exclusion inventory changed")
    if (
        sha256(prior_path.parent / "confirmation.complete.json")
        != plan["previous_confirmation_sha256"]
    ):
        raise ValueError("prior study completion changed")
    records = validate_confirmation(run, write=False)
    validate_population(records, {sym_key(r["fen"]) for r in read_jsonl(prior_path)})
    if not write:
        return records
    prefix = run / "render_validation"
    reserve_output(prefix)
    with Path(f"{prefix}.records.jsonl").open("x") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in records)
    finish_output(
        prefix,
        {
            "status": "guard_inputs_frozen_before_inference",
            "positions": 144,
            "images": 288,
            "truth_support_pass": True,
            "study_code_sha256": plan["study_code_sha256"],
        },
    )
    print(json.dumps({"validated_control_pairs": 144}), flush=True)


def infer(run):
    # Lazy torch import: data qualification and CPU tests never allocate CUDA.
    import torch

    try:
        from .parity_probe import CHECKPOINT_SHA, gpu_idle
        from .transfer_study import dual_decode, load_images, setup
    except ImportError:
        from parity_probe import CHECKPOINT_SHA, gpu_idle
        from transfer_study import dual_decode, load_images, setup

    verify_receipt(run / "render_validation.complete.json")
    records = validate(run, write=False)
    if records != read_jsonl(run / "render_validation.records.jsonl"):
        raise ValueError("images changed after freeze")
    truth = truth_for(records)
    prefix = run / "guard"
    reserve_output(prefix)
    plan, model, _, transform = setup(run)
    upstream_identity = {
        str(p): sha256(p) for p in sorted(Path(plan["cq_repo"]).glob("*.py"))
    }
    dual = []
    with torch.no_grad():
        for start in range(0, len(records), 4):
            memory = model.encoder.forward_features(
                load_images(records[start : start + 4], transform)
            )
            dual.append(dual_decode(model, memory).cpu().numpy())
            if start % 64 == 0:
                gpu_idle(os.getpid())
                print(json.dumps({"inferred_images": start + 4}), flush=True)
    logits = np.concatenate(dual)
    rules, _ = predict_rules(logits)
    guarded, diagnostics = predict_guarded(logits)
    predictions = {name: rules[name] for name in ("default", "flipped", "structure")}
    predictions["guarded"] = guarded
    scores = {
        name: score_predictions(p, truth, records) for name, p in predictions.items()
    }
    effects = {
        name: paired_summary(p, predictions["default"], truth, records, seed=20260911)
        for name, p in predictions.items()
        if name != "default"
    }
    decision = confirm_pass(
        "rule", scores["guarded"], scores["default"], effects["guarded"]
    )
    exact = {name: np.all(p == truth, axis=1) for name, p in predictions.items()}
    if np.any(exact["default"] & ~exact["guarded"]):
        raise ValueError("conditional exact nonregression invariant failed")
    cross = {}
    for family in FAMILIES:
        cross[family] = {}
        for condition in ("control", "parity"):
            idx = [
                i
                for i, r in enumerate(records)
                if r["family"] == family and r["condition"] == condition
            ]
            cross[family][condition] = {
                name: score_predictions(p[idx], truth[idx], [records[i] for i in idx])[
                    "overall"
                ]
                for name, p in predictions.items()
            }
    summary = {
        "scores": scores,
        "effects": effects,
        "decision": decision,
        "by_family_condition": cross,
        "retained_valid_default": int(diagnostics["retained_valid_default"].sum()),
        "selected_flipped": int(diagnostics["actual_choice"].sum()),
        "records": 288,
        "positions": 144,
        "truth_support_pass": True,
        "selector_inputs": ["logits"],
        "truth_used_for_support_audit_and_scoring": True,
        "upstream_source_sha256": upstream_identity,
    }
    with Path(f"{prefix}.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with Path(f"{prefix}.predictions.jsonl").open("x") as stream:
        for i, row in enumerate(records):
            stream.writelines(
                json.dumps(
                    {
                        "key": row["key"],
                        "model": name,
                        "gt": row["fen"],
                        "pred": labels_to_fen(p[i]),
                    }
                )
                + "\n"
                for name, p in predictions.items()
            )
    np.savez_compressed(
        f"{prefix}.logits.npz", logits=logits, labels=truth, **diagnostics
    )
    if records != validate(run, write=False):
        raise ValueError("input changed during inference")
    check_code(upstream_identity)
    if sha256(plan["checkpoint"]) != CHECKPOINT_SHA:
        raise ValueError("checkpoint changed")
    finish_output(
        prefix,
        {
            "status": "one_shot_guard_complete",
            "decision": decision,
            "base_checkpoint_sha256": CHECKPOINT_SHA,
            "input_receipt_sha256": sha256(run / "render_validation.complete.json"),
            "study_code_sha256": plan["study_code_sha256"],
            "upstream_source_sha256": upstream_identity,
        },
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "scores": {n: s["overall"] for n, s in scores.items()},
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "validate", "infer"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.previous is None or args.freeze is None:
            parser.error("prepare requires previous and freeze")
        prepare(args.run, args.previous, args.freeze)
    elif args.mode == "validate":
        validate(args.run)
    else:
        infer(args.run)


if __name__ == "__main__":
    main()
