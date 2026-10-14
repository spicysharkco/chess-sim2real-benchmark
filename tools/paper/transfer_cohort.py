"""Prepare and validate the fixed cross-geometry study; no model inference."""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

try:
    from .benchmark_io import (
        finish_output,
        preflight,
        read_jsonl,
        reserve_output,
        sha256,
        split_rows,
    )
    from .fen_pools import sym_key
    from .parity_probe import swap_parity, validate_pairs
except ImportError:
    from benchmark_io import (
        finish_output,
        preflight,
        read_jsonl,
        reserve_output,
        sha256,
        split_rows,
    )
    from fen_pools import sym_key
    from parity_probe import swap_parity, validate_pairs

FAMILIES = {
    "deplorable": ("renderer_ph_geometry.html", "deplorable_mountaineer_chess"),
    "killgorack": ("renderer_staunton_a.html", "killgorack_chess_20260906"),
    "lyricsz": ("renderer_ph_geometry_d.html", "lyricsz_stylized_chess_cc0_20260906"),
    "omie": ("renderer_ph_geometry_h2.html", "omie_chess_cc0_20260906"),
    "phuong": ("renderer_staunton_c.html", "phuong_chess_20260907"),
    "pjhanzlik": ("renderer_ph_geometry_h.html", "pjhanzlik_chess_cc0_20260906"),
}
# A public export reads this marker in place of a digest it withholds.
WITHHELD_DIGEST = "[redacted: withheld digest]"
GENERATOR_SHA = WITHHELD_DIGEST


def partition_development(rows):
    rng = random.Random(20260910)
    assignment = {}
    for split in sorted({row["probe_source_split"] for row in rows}):
        for quadrant in range(4):
            group = sorted(
                [
                    row
                    for row in rows
                    if row["probe_source_split"] == split
                    and quadrant * 90
                    <= row["camera"]["azimuth_deg"]
                    < (quadrant + 1) * 90
                ],
                key=lambda row: row["id"],
            )
            if len(group) != 12:
                raise ValueError(
                    "development strata must have exactly twelve positions"
                )
            chosen = {row["id"] for row in rng.sample(group, 8)}
            assignment.update(
                {
                    row["id"]: "train" if row["id"] in chosen else "validation"
                    for row in group
                }
            )
    return assignment


def record(row, image, partition, family, condition, original_id=None):
    group = hashlib.sha256(row["fen"].encode()).hexdigest()
    return dict(
        key=f"{partition}/{family}/{condition}/{row['id']}",
        group_id=group,
        id=row["id"],
        source_id=original_id or row["id"],
        family=family,
        condition=condition,
        partition=partition,
        fen=row["fen"],
        image=str(image.resolve()),
        image_sha256=None,
    )


def verify_receipt(path):
    receipt = json.loads(path.read_text())
    for name, digest in receipt["artifacts_sha256"].items():
        if sha256(path.parent / name) != digest:
            raise ValueError(f"receipt mismatch: {name}")
    return receipt


def prepare(run, old_run, source_root, renderer_root, freeze):
    old, _ = validate_pairs(old_run)
    verify_receipt(old_run / "prepare.complete.json")
    partitions = partition_development(old)
    inventory = preflight(
        source_root, freeze, [f"bench_geom_{family}" for family in FAMILIES]
    )
    prefix = run / "prepare"
    reserve_output(prefix)
    rendering = run / "src/rendering"
    rendering.mkdir(parents=True, exist_ok=True)
    (rendering / "assets").mkdir(exist_ok=True)
    historical = run / "historical_generate_ph_followup.py"
    if GENERATOR_SHA != WITHHELD_DIGEST and sha256(historical) != GENERATOR_SHA:
        raise ValueError(
            "historical generator snapshot does not match recorded render source"
        )
    shutil.copyfile(historical, rendering / "generate_ph_followup.py")
    for filename in ("render_batch.mjs", "three.min.js", "GLTFLoader.js"):
        shutil.copyfile(renderer_root / filename, rendering / filename)
    records = []
    for row in old:
        for condition in ("control", "parity"):
            item = record(
                row,
                old_run / f"render_{condition}/images/{row['id']}.png",
                partitions[row["id"]],
                "polyhaven",
                condition,
                row["probe_source_id"],
            )
            item["image_sha256"] = sha256(item["image"])
            records.append(item)
    (run / "manifests").mkdir()
    source_metadata = {}
    for family_index, (family, (renderer_name, asset)) in enumerate(FAMILIES.items()):
        split = f"bench_geom_{family}"
        metadata = {r["id"]: r for r in split_rows(source_root / split)}
        available = sorted(
            [r for r in inventory["rows"] if r["split"] == split], key=lambda r: r["id"]
        )
        rng = random.Random(20260910 + family_index)
        chosen = []
        for quadrant in range(4):
            group = [
                r
                for r in available
                if quadrant * 90 <= r["azimuth_deg"] < (quadrant + 1) * 90
            ]
            chosen.extend(rng.sample(group, 12))
        original = (renderer_root / renderer_name).read_bytes()
        if any(
            sha256(renderer_root / renderer_name)
            != metadata[r["id"]]["source_sha256"][renderer_name]
            for r in chosen
        ):
            raise ValueError(f"renderer source mismatch for {family}")
        (rendering / renderer_name).write_bytes(original)
        treatment_name = f"renderer_transfer_{family}.html"
        (rendering / treatment_name).write_bytes(swap_parity(original))
        shutil.copytree(renderer_root / "assets" / asset, rendering / "assets" / asset)
        for dependency in (
            "render_batch.mjs",
            "three.min.js",
            "generate_ph_followup.py",
        ):
            if (
                sha256(rendering / dependency)
                != metadata[chosen[0]["id"]]["source_sha256"][dependency]
            ):
                raise ValueError(
                    f"historical dependency mismatch: {family}/{dependency}"
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
        source_metadata[family] = dict(
            control_renderer=renderer_name,
            parity_renderer=treatment_name,
            n=48,
            geometry=manifest[0]["piece_geometry"],
        )
    memberships = {}
    for row in records:
        key = sym_key(row["fen"])
        memberships.setdefault(key, set()).add(row["partition"])
    if any(len(value) != 1 for value in memberships.values()):
        raise ValueError("position or D4-equivalent position crosses a partition")
    counts = {
        part: sum(r["partition"] == part for r in records)
        for part in ("train", "validation", "confirmation")
    }
    if counts != dict(train=256, validation=128, confirmation=576):
        raise ValueError(f"incorrect cohort sizes: {counts}")
    with Path(f"{prefix}.records.jsonl").open("x") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in records)
    plan = dict(
        old_run=str(old_run.resolve()),
        source_root=str(source_root.resolve()),
        checkpoint=str((old_run / "upstream/checkpoint.safetensors").resolve()),
        cq_repo=str((old_run / "upstream/minimal_reproduction").resolve()),
        freeze_sha256=inventory["freeze_sha256"],
        counts=counts,
        families=source_metadata,
        source_sha256={
            str(p.relative_to(rendering)): sha256(p)
            for p in sorted(rendering.rglob("*"))
            if p.is_file()
        },
    )
    with Path(f"{prefix}.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
    finish_output(
        prefix,
        dict(
            status="cohorts_fixed_before_inference",
            counts=counts,
            positions=len(memberships),
            generator_sha256=GENERATOR_SHA,
        ),
    )
    print(json.dumps(dict(prepared=counts)), flush=True)


def validate_confirmation(run, write=True):
    verify_receipt(run / "prepare.complete.json")
    plan = json.loads((run / "prepare.json").read_text())
    for name, digest in plan["source_sha256"].items():
        if sha256(run / "src/rendering" / name) != digest:
            raise ValueError(f"render source changed: {name}")
    records = read_jsonl(run / "prepare.records.jsonl")
    index = {
        (r["family"], r["condition"], r["id"]): r
        for r in records
        if r["partition"] == "confirmation"
    }
    for family in FAMILIES:
        manifest_path = run / "manifests" / f"{family}.jsonl"
        manifest = read_jsonl(manifest_path)
        manifest_sha = sha256(manifest_path)
        rendered = {
            c: {r["id"]: r for r in split_rows(run / f"renders/{family}/{c}")}
            for c in ("control", "parity")
        }
        if any(set(rows) != {r["id"] for r in manifest} for rows in rendered.values()):
            raise ValueError(f"incomplete render population: {family}")
        for row in manifest:
            a, b = rendered["control"][row["id"]], rendered["parity"][row["id"]]
            ignored = {"image_sha256", "source_sha256", "renderer"}
            inputs = {k: v for k, v in row.items() if k not in ignored}
            for condition, actual in (("control", a), ("parity", b)):
                if {k: v for k, v in actual.items() if k not in ignored} != inputs:
                    raise ValueError(
                        f"render input row changed: {family}/{condition}/{row['id']}"
                    )
                renderer_name = plan["families"][family][f"{condition}_renderer"]
                expected_sources = {
                    name: plan["source_sha256"][name]
                    for name in (
                        renderer_name,
                        "render_batch.mjs",
                        "generate_ph_followup.py",
                        "three.min.js",
                    )
                }
                expected_sources[manifest_path.name] = manifest_sha
                if actual["source_sha256"] != expected_sources:
                    raise ValueError(
                        f"render dependency snapshot changed: {family}/{condition}"
                    )
                if (
                    actual["source_sha256"].get(manifest_path.name) != manifest_sha
                    or actual["source_sha256"].get(renderer_name)
                    != plan["source_sha256"][renderer_name]
                ):
                    raise ValueError(
                        f"render manifest/renderer provenance changed: {family}/{condition}"
                    )
            if a["renderer"] != b["renderer"] or any(
                a[k] != b[k] or a[k] != row[k]
                for k in (
                    "fen",
                    "camera",
                    "piece_geometry",
                    "corners_px",
                    "a8_image_corner",
                )
            ):
                raise ValueError(f"non-parity factor changed: {family}/{row['id']}")
            for condition in ("control", "parity"):
                record_ = index[(family, condition, row["id"])]
                digest = sha256(record_["image"])
                if digest != rendered[condition][row["id"]]["image_sha256"]:
                    raise ValueError("rendered image mismatch")
                record_["image_sha256"] = digest
            if (
                index[(family, "control", row["id"])]["image_sha256"]
                != row["probe_original_sha256"]
            ):
                raise ValueError(f"control not byte-identical: {family}/{row['id']}")
            if (
                index[(family, "control", row["id"])]["image_sha256"]
                == index[(family, "parity", row["id"])]["image_sha256"]
            ):
                raise ValueError("parity intervention did not change image")
    if not write:
        return [r for r in records if r["partition"] == "confirmation"]
    prefix = run / "render_validation"
    reserve_output(prefix)
    with Path(f"{prefix}.records.jsonl").open("x") as stream:
        stream.writelines(
            json.dumps(r) + "\n" for r in records if r["partition"] == "confirmation"
        )
    finish_output(
        prefix,
        dict(
            status="all_six_families_validated",
            positions=288,
            images=576,
            source_manifest_sha256=sha256(run / "prepare.records.jsonl"),
        ),
    )
    print(json.dumps(dict(validated_pairs=288, families=len(FAMILIES))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "validate"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--old-run", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--renderer-root", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if a.mode == "prepare":
        if any(
            x is None for x in (a.old_run, a.source_root, a.renderer_root, a.freeze)
        ):
            p.error("prepare requires old-run, source-root, renderer-root, freeze")
        prepare(a.run, a.old_run, a.source_root, a.renderer_root, a.freeze)
    else:
        rows = validate_confirmation(a.run, write=not a.check_only)
        if a.check_only:
            print(json.dumps(dict(revalidated_images=len(rows))))


if __name__ == "__main__":
    main()
