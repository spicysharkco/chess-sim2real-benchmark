"""Re-render a fixed development-only smoke subset using an exported package."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

try:
    from .benchmark_io import png_dimensions, sha256, split_rows
    from .fen_pools import sym_key
except ImportError:
    from benchmark_io import png_dimensions, sha256, split_rows
    from fen_pools import sym_key

FAMILIES = ("polyhaven", "phuong")
# The export puts this marker where a record quoted the digest of a file it changed.
WITHHELD_DIGEST = "[redacted: withheld digest]"
LAUNCHER = "rendering/render_batch.mjs"
ASSET_README = re.compile(r"rendering/assets/[^/]+/README\.md")


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def member(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("unsafe package path")
    path = root / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("missing or escaping package file: " + name)
    return path


def verify_package(root, generated_output=None, expected_digest=None):
    inventory_path = member(root, "PACKAGE_INTEGRITY.json")
    if expected_digest is not None and sha256(inventory_path) != expected_digest:
        raise ValueError("package inventory bytes changed")
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema") != "public-package-files/v1" or not inventory.get(
        "files_sha256"
    ):
        raise ValueError("package integrity inventory is missing")
    for name, expected in inventory["files_sha256"].items():
        if sha256(member(root, name)) != expected:
            raise ValueError("package hash mismatch: " + name)
    expected_names = set(inventory["files_sha256"]) | {"PACKAGE_INTEGRITY.json"}
    # Only the documented installation roots and this invocation's exact output
    # directory may add files. Do not traverse dependency caches or ignore arbitrary
    # unlisted source files (including bytecode beside packaged Python sources).
    for directory, subdirs, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        allowed = [root / ".runtime", root / ".venv"]
        if generated_output is not None:
            allowed.append(generated_output)
        for name in list(subdirs):
            path = base / name
            if path in allowed:
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError("escaping runtime directory: " + str(path))
                subdirs.remove(name)
            elif path.is_symlink():
                raise ValueError("unlisted package directory symlink: " + str(path))
        for name in filenames:
            relative = (base / name).relative_to(root).as_posix()
            if relative not in expected_names:
                raise ValueError("unlisted package file: " + relative)
    return inventory


def verify_manifest_pin(manifest, shard):
    """A shard manifest matches its plan pin, or the pin reads the withheld marker.

    The pin reads the marker for a manifest the export rewrote. The package
    integrity file binds the shipped bytes of that manifest.
    """
    pin = shard["manifest_sha256"]
    if pin != WITHHELD_DIGEST and sha256(manifest) != pin:
        raise ValueError("original shard manifest hash mismatch")


def verify_render_sources(root, plan, integrity):
    """Keep scene and asset bytes frozen; permit only declared launcher and README updates."""
    updates = integrity.get("source_updates", {})
    if any(name != LAUNCHER and not ASSET_README.fullmatch(name) for name in updates):
        raise ValueError("unapproved rendering source update")
    used = set()
    for name, expected in plan["source_sha256"].items():
        if not name.startswith("rendering/"):
            continue
        actual = sha256(member(root, name))
        if name in updates:
            # The frozen digest of an updated file is withheld: the plan shows the
            # marker and the declaration names the shipped bytes.
            if expected != WITHHELD_DIGEST or updates[name] != {
                "current_sha256": actual
            }:
                raise ValueError("rendering source update hash mismatch: " + name)
            used.add(name)
        elif actual != expected:
            raise ValueError("frozen renderer or asset changed: " + name)
    if used != set(updates):
        raise ValueError("unused rendering source update")


def verify_freeze(root):
    """Return the row inventory and its freeze receipt once they belong together."""
    frozen_path = member(root, "benchmark_v2/benchmark.jsonl")
    parent_path = member(root, "benchmark_v2/benchmark.complete.json")
    parent = json.loads(parent_path.read_text())
    if parent.get("images") != 20000 or parent.get("final_test_sealed") is not True:
        raise ValueError("unexpected v2 freeze scope")
    # The export withholds the freeze digest of a row inventory it changed. The
    # package inventory, checked before this, then binds the shipped rows.
    if parent["artifacts_sha256"]["benchmark.jsonl"] not in (
        WITHHELD_DIGEST,
        sha256(frozen_path),
    ):
        raise ValueError("original freeze does not bind the packaged row inventory")
    return frozen_path, parent_path


def select_rows(rows):
    """Two reference dark views per family, ordered IDs, distinct positions globally."""
    selected, used = [], set()
    for family in FAMILIES:
        candidates = sorted(
            (
                r
                for r in rows
                if r.get("partition") == "dev"
                and r.get("factor_axis") == "geometry"
                and r.get("factor_level") == family
                and r.get("parity") == "a1_dark"
            ),
            key=lambda row: (row["split"], row["id"]),
        )
        count = 0
        for row in candidates:
            key = sym_key(row["fen"])
            if key in used:
                continue
            used.add(key)
            selected.append(row)
            count += 1
            if count == 2:
                break
        if count != 2:
            raise ValueError("missing distinct development reference views: " + family)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--playwright", required=True)
    parser.add_argument("--browser", required=True)
    args = parser.parse_args()
    root = args.package.resolve()
    out = args.out.resolve()
    inventory_digest = sha256(member(root, "PACKAGE_INTEGRITY.json"))
    integrity = verify_package(root, out, inventory_digest)
    frozen_path, parent_path = verify_freeze(root)
    plan = json.loads(member(root, "benchmark_v2/design.plan.json").read_text())
    # Historical source checks use exact rendering bytes, never a forged Git revision.
    verify_render_sources(root, plan, integrity)
    rows = select_rows(read_rows(frozen_path))
    if out == root or root.is_relative_to(out):
        raise ValueError("output cannot contain the source package")
    out.mkdir(parents=True, exist_ok=False)
    (out / "INCOMPLETE").write_text("No successful smoke receipt yet.\n")
    (out / "manifests").mkdir()
    (out / "renders").mkdir()
    commands, results = [], []
    selected_shards = sorted({row["split"] for row in rows})
    for split in selected_shards:
        selected = [r for r in rows if r["split"] == split]
        shard = next(s for s in plan["shards"] if s["name"] == split)
        original = member(root, "benchmark_v2/manifests/" + split + ".jsonl")
        verify_manifest_pin(original, shard)
        original_rows = {r["id"]: r for r in read_rows(original)}
        inputs = []
        for row in selected:
            source = original_rows[row["id"]]
            if any(row.get(key) != value for key, value in source.items()):
                raise ValueError("manifest row differs from frozen scene")
            inputs.append(source)
        manifest = out / "manifests" / (split + ".jsonl")
        manifest.write_text(
            "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in inputs)
        )
        spec = plan["families"][shard["family"]]
        command = [
            "node",
            str(root / "rendering/render_batch.mjs"),
            f"manifest={manifest}",
            f"out={out / 'renders' / split}",
            f"renderer={spec['renderer']}",
            f"engine=benchmark_v2_{shard['family']}",
            "generator=generate_benchmark_v2.py",
            "backend=swiftshader",
            f"sources={','.join(spec['extra_sources'])}",
            f"playwright={Path(args.playwright).resolve()}",
            f"browser={Path(args.browser).resolve()}",
        ]
        commands.append(command)
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
        }
        with (out / (split + ".log")).open("x") as log:
            subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        folder = out / "renders" / split
        rendered = split_rows(folder)
        if {r["id"] for r in rendered} != {r["id"] for r in selected}:
            raise ValueError("rendered subset inventory differs")
        original_by_id = {r["id"]: r for r in selected}
        enriched = []
        for actual in rendered:
            expected = original_by_id[actual["id"]]
            image = folder / "images" / (actual["id"] + ".png")
            if (
                sha256(image) != expected["sha256"]
                or actual["image_sha256"] != expected["sha256"]
            ):
                raise ValueError(
                    "PNG differs from frozen image: " + split + "/" + actual["id"]
                )
            if (
                png_dimensions(image) != [1920, 1080]
                or actual["fen"] != expected["fen"]
            ):
                raise ValueError("render dimensions or labels differ")
            if (
                actual["renderer"]["canonical_corners_px"]
                != expected["renderer"]["canonical_corners_px"]
            ):
                raise ValueError("rendered projection differs")
            # Add verified frozen GT for the scorer; do not recompute labels from predictions.
            actual.update(
                corners_px=expected["corners_px"],
                a8_image_corner=expected["a8_image_corner"],
            )
            enriched.append(actual)
            results.append(
                {
                    "split": split,
                    "id": actual["id"],
                    "position_group": expected["position_group"],
                    "image_sha256": sha256(image),
                    "frozen_sha256": expected["sha256"],
                    "match": True,
                }
            )
        (folder / "metadata.jsonl").write_text(
            "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in enriched)
        )
    subset = out / "selected.freeze.jsonl"
    subset.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    )
    if verify_package(root, out, inventory_digest) != integrity:
        raise ValueError("package inventory changed during rendering")
    record = {
        "schema": "clean-package-render-smoke/v1",
        "status": "render_hashes_match",
        "scope": "four-row development smoke, not a full benchmark score",
        "benchmark_complete": False,
        "source_freeze_sha256": sha256(frozen_path),
        "source_receipt_sha256": sha256(parent_path),
        "package_inventory_sha256": inventory_digest,
        "source_updates": integrity.get("source_updates", {}),
        "selection": "two reference a1-dark geometry views per family, ascending frozen IDs, globally exact/D4-distinct",
        "families": list(FAMILIES),
        "selected_rows": results,
        "selected_freeze_sha256": sha256(subset),
        "commands": commands,
        "final_test_images_read": False,
        "gpu_used": False,
        "published": False,
        "script_sha256": sha256(Path(__file__)),
    }
    (out / "render.complete.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )
    (out / "INCOMPLETE").unlink()
    print(
        json.dumps({"images": len(results), "all_hashes_match": True, "out": str(out)})
    )


if __name__ == "__main__":
    main()
