"""Prepare and freeze paired benchmark v2 and its disjoint adaptation pool.

CPU/data operations only. No detector or recognizer is invoked, including on
final_test. Source assets and exclusions must already exist on the render host.
"""

# This standalone CLI establishes its checkout import root before repo imports.
# ruff: noqa: E402

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rendering import generate_benchmark_v2 as design
from tools.paper.annotate_benchmark import ground_truth, load_helpers
from tools.paper.benchmark_io import (
    finish_output,
    png_dimensions,
    read_jsonl,
    reserve_output,
    sha256,
    split_rows,
)
from tools.paper.fen_pools import placement, read_pool, sym_key, symmetry_keys
from tools.paper.transfer_cohort import verify_receipt

RESOURCES = {
    "polyhaven": ("renderer_ph_colors_h.html", None, "polyhaven_chess_set"),
    "deplorable": (
        "renderer_ph_geometry.html",
        "deplorable_geometry.json",
        "deplorable_mountaineer_chess",
    ),
    "killgorack": (
        "renderer_staunton_a.html",
        "killgorack_geometry.json",
        "killgorack_chess_20260906",
    ),
    "lyricsz": (
        "renderer_ph_geometry_d.html",
        "lyricsz_geometry.json",
        "lyricsz_stylized_chess_cc0_20260906",
    ),
    "omie": (
        "renderer_ph_geometry_h2.html",
        "omie_geometry.json",
        "omie_chess_cc0_20260906",
    ),
    "phuong": (
        "renderer_staunton_c.html",
        "phuong_geometry.json",
        "phuong_chess_20260907",
    ),
    "pjhanzlik": (
        "renderer_ph_geometry_h.html",
        "pjhanzlik_geometry.json",
        "pjhanzlik_chess_cc0_20260906",
    ),
    "nailfighter": (
        "renderer_ph_geometry.html",
        "nailfighter_geometry.json",
        "nailfighter_chess_cc0_20260906",
    ),
}
PH_DESCRIPTOR = {
    "source": "Poly Haven chess_set 1k glTF",
    "url": "https://polyhaven.com/a/chess_set",
    "license": "CC0",
    "fixed_grid_fit_scale": 0.82,
    "geometry_jitter": False,
}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_rows(path, rows):
    with Path(path).open("x") as stream:
        stream.writelines(
            json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        )


def source_identity(repo):
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True
    )
    if status.strip():
        raise ValueError("render source checkout is not clean")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def check_sources(repo, plan):
    if source_identity(repo) != plan["repo_commit"]:
        raise ValueError("render source revision changed")
    for name, digest in plan["source_sha256"].items():
        if sha256(repo / name) != digest:
            raise ValueError(f"render source/asset changed: {name}")


def resources(repo, families):
    rendering = repo / "rendering"
    descriptors, files, specs = {}, set(), {}
    for family in families:
        renderer, descriptor, asset = RESOURCES[family]
        descriptors[family] = (
            json.loads((rendering / descriptor).read_text())
            if descriptor
            else PH_DESCRIPTOR
        )
        extra = [
            "benchmark_v2.js",
            "generate_geometry.py",
            "generate_colors_b.py",
            "holdout_palettes.json",
        ]
        if descriptor:
            extra.append(descriptor)
        if family in {"deplorable", "nailfighter"}:
            extra.extend(["deplorable_geometry.json", "nailfighter_geometry.json"])
        if family in {"polyhaven", "lyricsz", "pjhanzlik"}:
            extra.append("GLTFLoader.js")
        extra = sorted(set(extra))
        files.update(
            rendering / name
            for name in [
                renderer,
                "render_batch.mjs",
                "generate_benchmark_v2.py",
                "three.min.js",
                *extra,
            ]
        )
        asset_files = [
            p
            for p in sorted((rendering / "assets" / asset).rglob("*"))
            if p.is_file() and not p.name.startswith("._")
        ]
        if not asset_files:
            raise ValueError(f"empty/missing asset directory: {family}")
        files.update(asset_files)
        descriptor_value = descriptors[family]
        identity = descriptor_value.get("asset_identity", "")
        if (
            identity.endswith(".json")
            and sha256(rendering / "assets" / identity)
            != descriptor_value["asset_sha256"]
        ):
            raise ValueError(f"geometry asset digest mismatch: {family}")
        for name, digest in descriptor_value.get("mesh_sha256", {}).items():
            if sha256(rendering / "assets" / asset / name) != digest:
                raise ValueError(f"mesh digest mismatch: {family}/{name}")
        specs[family] = {"renderer": renderer, "extra_sources": extra}
    files.update((repo / "tools/paper").glob("*.py"))
    files.update([rendering / "annotate_orientation.py", repo / "src/board.py"])
    return (
        descriptors,
        specs,
        {str(p.relative_to(repo)): sha256(p) for p in sorted(files)},
    )


def collect_exclusions(config):
    """Required explicit named inventory; missing/empty/count-mismatched pools fail."""
    entries = json.loads(Path(config).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("nonempty explicit source inventory is required")
    if len({e["name"] for e in entries}) != len(entries):
        raise ValueError("duplicate source names")
    exact, canonical, evidence = set(), set(), []
    for entry in entries:
        if entry.get("sha256") and sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"source file digest mismatch: {entry['name']}")
        rows = read_pool(entry["path"])
        if not rows or len(rows) != entry["expected_rows"]:
            raise ValueError(f"source count mismatch: {entry['name']}: {len(rows)}")
        fens = {placement(fen) for fen in rows}
        keys = symmetry_keys(fens, workers=16 if len(fens) > 100000 else 1)
        fingerprint = hashlib.sha256("\n".join(sorted(fens)).encode()).hexdigest()
        if entry.get("placement_sha256") not in (None, fingerprint):
            raise ValueError(f"source fingerprint mismatch: {entry['name']}")
        exact.update(fens)
        canonical.update(keys)
        if entry.get("sha256") and sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"source file changed while reading: {entry['name']}")
        evidence.append(
            {
                "source": entry["name"],
                "rows": len(rows),
                "unique": len(fens),
                "unique_d4": len(keys),
                "placement_sha256": fingerprint,
            }
        )
        print(
            json.dumps({"source_checked": entry["name"], "rows": len(rows)}), flush=True
        )
    return exact, canonical, evidence


def validate_design(rows, groups, cohort, palettes, descriptors):
    families = tuple(descriptors)
    expected = design.design_rows(
        groups,
        descriptors,
        palettes,
        adaptation=cohort == "adaptation",
        geometries=families,
    )
    ignored = {"excluded_fen_count", "excluded_fen_sha256", "repo_commit"}
    if [{k: v for k, v in row.items() if k not in ignored} for row in rows] != expected:
        raise ValueError("manifest does not reproduce the preregistered design")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate image IDs")
    if cohort == "benchmark":
        if len(groups) != 500 or Counter(g["partition"] for g in groups) != {
            "dev": 400,
            "final_test": 100,
        }:
            raise ValueError("benchmark must contain 400/100 position groups")
    elif cohort == "adaptation":
        if len(groups) != 8000 or len(rows) != 16000:
            raise ValueError("adaptation must contain 8000 groups/16000 images")
    else:
        raise ValueError("unknown cohort")
    for group in groups:
        if group["position_group"] != design.canonical_position_group(group["fen"]):
            raise ValueError("position group identity mismatch")


def prepare(run, repo, pools, nailfighter=False):
    commit = source_identity(repo)
    families = (*design.GEOMETRIES, "nailfighter") if nailfighter else design.GEOMETRIES
    descriptors, specs, hashes = resources(repo, families)
    palettes = json.loads((repo / "rendering/holdout_palettes.json").read_text())
    exact, excluded, evidence = collect_exclusions(pools)
    prefix = run / "design"
    reserve_output(prefix)
    base = design.generate_groups(500, design.BENCHMARK_SEED, excluded)
    benchmark = design.partition_benchmark_groups(base)
    excluded.update(sym_key(g["fen"]) for g in benchmark)
    adaptation = design.generate_groups(
        8000, design.ADAPTATION_GENERATION_SEED, excluded
    )
    digest = hashlib.sha256("\n".join(sorted(exact)).encode()).hexdigest()
    adaptation_digest = hashlib.sha256(
        "\n".join(sorted(exact | {g["fen"] for g in benchmark})).encode()
    ).hexdigest()
    plan = {
        "benchmark_schema": 2,
        "repo_commit": commit,
        "source_sha256": hashes,
        "families": specs,
        "geometry_descriptors": descriptors,
        "palettes": palettes,
        "pool_inventory_sha256": sha256(pools),
        "source_pools": evidence,
        "excluded_fen_count": len(exact),
        "excluded_fen_sha256": digest,
        "adaptation_excluded_fen_count": len(exact) + 500,
        "adaptation_excluded_fen_sha256": adaptation_digest,
        "shards": [],
        "release_ready": False,
    }
    (run / "manifests").mkdir()
    for cohort, groups in (("benchmark", benchmark), ("adaptation", adaptation)):
        rows = design.design_rows(
            groups,
            descriptors,
            palettes,
            adaptation=cohort == "adaptation",
            geometries=families,
        )
        for row in rows:
            row.update(
                repo_commit=commit,
                excluded_fen_count=plan["adaptation_excluded_fen_count"]
                if cohort == "adaptation"
                else len(exact),
                excluded_fen_sha256=adaptation_digest
                if cohort == "adaptation"
                else digest,
            )
        validate_design(rows, groups, cohort, palettes, descriptors)
        write_rows(run / f"design.{cohort}.groups.jsonl", groups)
        write_rows(run / f"design.{cohort}.rows.jsonl", rows)
        for partition in sorted({r["partition"] for r in rows}):
            for family in families:
                subset = [
                    r
                    for r in rows
                    if r["partition"] == partition
                    and r["factors"]["geometry"] == family
                ]
                for start in range(0, len(subset), 500):
                    selected = subset[start : start + 500]
                    name = f"{cohort}_{partition}_{family}_{start // 500:03d}"
                    path = run / "manifests" / f"{name}.jsonl"
                    write_rows(path, selected)
                    plan["shards"].append(
                        {
                            "name": name,
                            "cohort": cohort,
                            "partition": partition,
                            "family": family,
                            "count": len(selected),
                            "manifest_sha256": sha256(path),
                        }
                    )
    plan["nested_budgets"] = {
        str(n): [
            g["position_group"] for g in design.adaptation_budget_groups(adaptation, n)
        ]
        for n in design.ADAPTATION_BUDGETS
    }
    write_json(run / "design.plan.json", plan)
    write_rows(run / "design.candidates.jsonl", benchmark + adaptation)
    check_sources(repo, plan)
    finish_output(
        prefix,
        {
            "status": "design_locked_before_rendering",
            "benchmark_schema": 2,
            "repo_commit": commit,
            "benchmark_groups": 500,
            "adaptation_groups": 8000,
        },
    )
    print(
        json.dumps({"design_complete": str(prefix), "shards": len(plan["shards"])}),
        flush=True,
    )


def load_plan(run, repo):
    verify_receipt(run / "design.complete.json")
    plan = json.loads((run / "design.plan.json").read_text())
    check_sources(repo, plan)
    for shard in plan["shards"]:
        if (
            sha256(run / "manifests" / f"{shard['name']}.jsonl")
            != shard["manifest_sha256"]
        ):
            raise ValueError(f"manifest changed: {shard['name']}")
    return plan


def render_commands(run, repo, cohort, playwright, browser):
    plan = load_plan(run, repo)
    commands = []
    for shard in plan["shards"]:
        if shard["cohort"] != cohort:
            continue
        spec = plan["families"][shard["family"]]
        commands.append(
            [
                "node",
                str(repo / "rendering/render_batch.mjs"),
                f"manifest={run / 'manifests' / (shard['name'] + '.jsonl')}",
                f"out={run / 'renders' / shard['name']}",
                f"renderer={spec['renderer']}",
                f"engine=benchmark_v2_{shard['family']}",
                "generator=generate_benchmark_v2.py",
                "backend=swiftshader",
                f"sources={','.join(spec['extra_sources'])}",
                f"playwright={playwright}",
                f"browser={browser}",
            ]
        )
    return commands


def freeze(run, repo, cohort, audit):
    plan = load_plan(run, repo)
    audit_data = json.loads(Path(audit).read_text())
    check = audit_data.get("check", {})
    if (
        check.get("clean") is not True
        or check.get("leaked") != 0
        or any(check.get(k) != 8500 for k in ("n", "unique", "unique_sym"))
        or any(
            value in {"exact", "sym"}
            for hit in check.get("hits", [])
            for value in hit.get("where", {}).values()
        )
    ):
        raise ValueError("complete 8500-position exclusion audit is required")
    candidate_file = run / "design.candidates.jsonl"
    candidate_fens = {g["fen"] for g in read_jsonl(candidate_file)}
    expected_hash = hashlib.sha256(
        "\n".join(sorted(candidate_fens)).encode()
    ).hexdigest()
    if (
        audit_data["check"].get("candidate_content_sha256") != sha256(candidate_file)
        or audit_data["check"].get("candidate_canonical_set_sha256") != expected_hash
    ):
        raise ValueError("audit candidates differ from generated population")
    expected_pools = {r["source"]: r for r in plan["source_pools"]}
    if set(audit_data.get("pools", {})) != set(expected_pools):
        raise ValueError("audit omitted or added a source pool")
    for name, expected_pool in expected_pools.items():
        actual_pool = audit_data["pools"][name]
        if any(
            actual_pool[k] != expected_pool[e]
            for k, e in (
                ("n", "rows"),
                ("unique", "unique"),
                ("unique_sym", "unique_d4"),
                ("placement_sha256", "placement_sha256"),
            )
        ):
            raise ValueError(f"audit source differs from generation exclusion: {name}")
    if cohort == "adaptation":
        verify_receipt(run / "benchmark.complete.json")
    prefix = run / cohort
    reserve_output(prefix)
    groups = read_jsonl(run / f"design.{cohort}.groups.jsonl")
    designed = read_jsonl(run / f"design.{cohort}.rows.jsonl")
    validate_design(
        designed, groups, cohort, plan["palettes"], plan["geometry_descriptors"]
    )
    board, ao = load_helpers(repo)
    rows, summaries = [], []
    for shard in plan["shards"]:
        if shard["cohort"] != cohort:
            continue
        folder = run / "renders" / shard["name"]
        manifest = read_jsonl(run / "manifests" / f"{shard['name']}.jsonl")
        actual = split_rows(folder)
        if {r["id"] for r in actual} != {r["id"] for r in manifest}:
            raise ValueError(f"render population mismatch: {shard['name']}")
        by_id = {r["id"]: r for r in manifest}
        spec = plan["families"][shard["family"]]
        sources = {
            name: plan["source_sha256"][f"rendering/{name}"]
            for name in [
                spec["renderer"],
                "render_batch.mjs",
                "generate_benchmark_v2.py",
                "three.min.js",
                *spec["extra_sources"],
            ]
        }
        sources[f"{shard['name']}.jsonl"] = shard["manifest_sha256"]
        for row in actual:
            if {
                k: v
                for k, v in row.items()
                if k not in {"renderer", "source_sha256", "image_sha256"}
            } != by_id[row["id"]]:
                raise ValueError("rendered input row changed")
            if row["source_sha256"] != sources:
                raise ValueError("rendered source lineage differs from locked checkout")
            image = folder / "images" / f"{row['id']}.png"
            if (
                png_dimensions(image) != [1920, 1080]
                or sha256(image) != row["image_sha256"]
            ):
                raise ValueError("image hash/dimension mismatch")
            gt = ground_truth(row, board, ao)
            canonical = ao.project(ao.CANONICAL_WORLD, row["camera"], row["renderer"])
            if not np.allclose(
                canonical, row["renderer"]["canonical_corners_px"], atol=1e-6, rtol=0
            ):
                raise ValueError("independent camera projection mismatch")
            rows.append(
                dict(
                    row,
                    **gt,
                    image=str(image.relative_to(run)),
                    sha256=row["image_sha256"],
                    md5=hashlib.md5(image.read_bytes()).hexdigest(),
                    split=shard["name"],
                )
            )
        runtime = json.loads((folder / "runtime.json").read_text())
        summaries.append(
            {
                "shard": shard["name"],
                "count": len(actual),
                "runtime": {
                    k: runtime[k]
                    for k in ("node", "browser", "requested_backend", "webgl")
                },
            }
        )
    if len(rows) != len(designed) or {r["id"] for r in rows} != {
        r["id"] for r in designed
    }:
        raise ValueError("frozen population incomplete")
    rows.sort(key=lambda r: r["id"])
    pairs = {}
    for row in rows:
        pairs.setdefault((row["position_group"], row["factor_cell"]), []).append(row)
    for pair in pairs.values():
        if len(pair) != 2 or {r["parity"] for r in pair} != set(design.PARITIES):
            raise ValueError("missing parity twin")
        first, second = pair
        if (
            first["camera"] != second["camera"]
            or first["renderer"] != second["renderer"]
            or first["sha256"] == second["sha256"]
        ):
            raise ValueError("parity pair must share exact camera and differ in image")
    for row in rows:
        folder = run / "renders" / row["split"]
        (folder / "corners").mkdir(exist_ok=True)
        corners = folder / "corners" / f"{row['id']}.txt"
        with corners.open("x") as stream:
            stream.write("".join(f"{x:.3f},{y:.3f}\n" for x, y in row["corners_px"]))
        label = folder / "labels" / f"{row['id']}.txt"
        row.update(
            corners=str(corners.relative_to(run)),
            corners_sha256=sha256(corners),
            label=str(label.relative_to(run)),
            label_sha256=sha256(label),
        )
    final_hashes = [
        {"id": r["id"], "image": r["image"], "sha256": r["sha256"]}
        for r in rows
        if r["partition"] == "final_test"
    ]
    summary = {
        "benchmark_schema": 2,
        "cohort": cohort,
        "release_ready": False,
        "repo_commit": plan["repo_commit"],
        "position_groups": len(groups),
        "images": len(rows),
        "partitions": dict(Counter(r["partition"] for r in rows)),
        "cells": dict(
            Counter(f"{r['factor_cell']}/{r['parity']}/{r['partition']}" for r in rows)
        ),
        "unique_image_hashes": len({r["sha256"] for r in rows}),
        "detector_invoked": False,
        "recognizer_invoked": False,
        "audit_sha256": sha256(audit),
        "render_shards": summaries,
    }
    write_rows(run / f"{cohort}.jsonl", rows)
    write_json(run / f"{cohort}.summary.json", summary)
    check_sources(repo, plan)
    finish_output(
        prefix,
        {
            "status": "schema_v2_frozen",
            "benchmark_schema": 2,
            "release_ready": False,
            "repo_commit": plan["repo_commit"],
            "images": len(rows),
            "final_test_sealed": cohort == "benchmark",
            "final_test_image_hashes": final_hashes,
            "design_receipt_sha256": sha256(run / "design.complete.json"),
            "exclusion_audit_sha256": sha256(audit),
            "benchmark_receipt_sha256": sha256(run / "benchmark.complete.json")
            if cohort == "adaptation"
            else None,
        },
    )
    print(
        json.dumps(
            {"receipt": str(run / f"{cohort}.complete.json"), "images": len(rows)}
        ),
        flush=True,
    )


def load_dev(freeze_path):
    """Development-only entry point; this task provides no final-test override."""
    freeze_path = Path(freeze_path)
    receipt = verify_receipt(freeze_path.with_suffix(".complete.json"))
    if receipt.get("benchmark_schema") != 2 or not receipt.get("final_test_sealed"):
        raise ValueError("expected sealed benchmark-v2 receipt")
    rows = read_jsonl(freeze_path)
    if any(
        r.get("partition") not in {"dev", "final_test"}
        or r.get("benchmark_schema") != 2
        for r in rows
    ):
        raise ValueError("unknown partition or schema in benchmark freeze")
    final_hashes = [
        {"id": r["id"], "image": r["image"], "sha256": r["sha256"]}
        for r in rows
        if r["partition"] == "final_test"
    ]
    if final_hashes != receipt.get("final_test_image_hashes"):
        raise ValueError("sealed final-test inventory does not match freeze")
    memberships = {}
    for row in rows:
        memberships.setdefault(sym_key(row["fen"]), set()).add(row["partition"])
    if any(len(parts) != 1 for parts in memberships.values()):
        raise ValueError("D4-equivalent position crosses benchmark partitions")
    return [r for r in rows if r["partition"] == "dev"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "commands", "freeze"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--repo", default=ROOT, type=Path)
    parser.add_argument("--pools", type=Path)
    parser.add_argument("--nailfighter", action="store_true")
    parser.add_argument(
        "--cohort", choices=("benchmark", "adaptation"), default="benchmark"
    )
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--playwright")
    parser.add_argument("--browser")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.pools is None:
            parser.error("prepare requires --pools")
        prepare(args.run, args.repo, args.pools, args.nailfighter)
    elif args.mode == "commands":
        if not args.playwright or not args.browser:
            parser.error("commands requires --playwright and --browser")
        print(
            json.dumps(
                render_commands(
                    args.run, args.repo, args.cohort, args.playwright, args.browser
                )
            )
        )
    else:
        if args.audit is None:
            parser.error("freeze requires --audit")
        freeze(args.run, args.repo, args.cohort, args.audit)


if __name__ == "__main__":
    main()
