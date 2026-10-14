#!/usr/bin/env python3
"""Score a recogniser on every benchmark v1 split and write one row per split.

Two model kinds, because the two systems the paper compares do not share an interface:

  --system chessqueries   the release checkpoint through the authors' own path (raw image squashed to
                          644, argmax over their 64 queries) -- the same code `cq_baseline.py` uses.
  --system submission     an extracted submission tree (`predict.py` + `model/`), optionally with a
                          (`member_eval.py`). Pass --variant to subset the members.

Output: `<out>.jsonl` one row per (split, system) with exact/per-square/ms-per-image and the
wrong-square histogram, plus `<out>.md` as a table. Per-board predictions go to
`<out>.<split>.tsv` so a miss can be traced back to an image.

  python tools/paper/score_benchmark.py --root ~/bench_v1/renders --freeze ~/bench_v1/FREEZE_2026-09-09.jsonl --system chessqueries \
      --repo-cq $DATA_ROOT/ext/chessqueries/repo/minimal_reproduction \
      --ckpt $DATA_ROOT/ext/chessqueries/chessqueries-vitL14-644-joint.safetensors \
      --out tools/reports/paper/bench_chessqueries_2026-09-09
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

try:
    from .benchmark_io import (
        finish_output,
        preflight,
        read_jsonl,
        reserve_output,
        sha256,
        verify_unchanged,
    )
    from .fen_pools import cells, labels_to_fen, placement
except ImportError:
    from benchmark_io import (
        finish_output,
        preflight,
        read_jsonl,
        reserve_output,
        sha256,
        verify_unchanged,
    )
    from fen_pools import cells, labels_to_fen, placement


def model_identity(a):
    """Pin inputs, not just a checkpoint basename or a stale checkout label."""
    if a.system == "chessqueries":
        if not a.ckpt or not a.repo_cq:
            raise ValueError("chessqueries requires --ckpt and --repo-cq")
        files = [Path(a.ckpt), *sorted(Path(a.repo_cq).rglob("*.py"))]
        if len(files) < 2:
            raise ValueError("missing ChessQueries source files")
    else:
        if not a.sub or not (Path(a.sub) / "predict.py").is_file():
            raise ValueError("submission requires --sub with predict.py")
        files = [
            p
            for p in sorted(Path(a.sub).rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json", ".pt", ".safetensors"}
        ]
    return {str(p.resolve()): sha256(p) for p in files}


def make_chessqueries(a):
    import torch

    sys.path.insert(0, a.repo_cq)
    from eval import load_model
    from torchvision.io import read_image

    from data import eval_transform

    torch.set_float32_matmul_precision("high")
    model = load_model(Path(a.ckpt), a.device)
    tf = eval_transform(644)

    def predict(path):
        x = tf(read_image(str(path)).float())[None].to(a.device)
        with torch.no_grad():
            return labels_to_fen(model(x).argmax(-1)[0].cpu().tolist())

    return predict


def make_submission(a):
    sub = Path(a.sub)
    if a.variant and a.variant != "full":
        sys.path.insert(0, str(Path(__file__).parent))
        from member_eval import make_variant

        sub, _ = make_variant(str(sub), a.variant)
    os.environ.setdefault("BUDGET_S", "1000000000")
    sys.path.insert(0, str(sub))
    spec = importlib.util.spec_from_file_location("predict", sub / "predict.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    return lambda path: P.predict(str(path))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument(
        "--freeze",
        required=True,
        type=Path,
        help="authoritative frozen JSONL or JSONL.gz",
    )
    ap.add_argument(
        "--parent-freeze",
        type=Path,
        help="parent JSONL inventory when --freeze is a derived selection, not the full benchmark",
    )
    ap.add_argument("--system", required=True, choices=["chessqueries", "submission"])
    ap.add_argument("--repo-cq", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--sub", default=None)
    ap.add_argument("--variant", default="full")
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="ChessQueries inference device (default: cuda)",
    )
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="CPU inventory check; no model or output writes",
    )
    ap.add_argument("--out", required=True)
    return ap


def scope_selection(inventory, freeze, parent_freeze):
    """Bind a proper row-preserving subset without calling it full coverage."""
    parent_digest = sha256(parent_freeze)
    parents = read_jsonl(parent_freeze)
    indexed = {(r["split"], r["id"]): r for r in parents}
    selected = read_jsonl(freeze)
    if len(indexed) != len(parents) or len(selected) >= len(parents):
        raise ValueError(
            "parent inventory must have unique IDs and contain a proper subset"
        )
    if any(indexed.get((r["split"], r["id"])) != r for r in selected):
        raise ValueError("selected freeze row differs from parent inventory")
    if sha256(parent_freeze) != parent_digest:
        raise ValueError("parent inventory changed during validation")
    inventory.update(
        scope="selection",
        selection_complete=inventory["benchmark_complete"],
        benchmark_complete=False,
        parent_freeze=str(parent_freeze.resolve()),
        parent_freeze_sha256=parent_digest,
        parent_frozen_n=len(parents),
    )


def main(argv=None):
    a = build_parser().parse_args(argv)

    if a.system != "chessqueries" and a.device == "cpu":
        raise ValueError("CPU mode is only supported for chessqueries")
    inventory = preflight(a.root, a.freeze, a.splits, a.limit)
    if a.parent_freeze:
        scope_selection(inventory, a.freeze, a.parent_freeze)
    receipt = {key: value for key, value in inventory.items() if key != "rows"}
    receipt["device"] = (
        a.device if a.system == "chessqueries" else "system_managed_unverified"
    )
    if a.validate_only:
        print(json.dumps(receipt, sort_keys=True))
        return
    receipt.update(
        model_inputs_sha256=model_identity(a),
        command=sys.argv,
        tool_sha256={
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("benchmark_io.py"),
                Path(__file__).with_name("fen_pools.py"),
                Path(__file__).with_name("member_eval.py"),
            )
        },
    )
    out = Path(a.out)
    reserve_output(out)
    predict = make_chessqueries(a) if a.system == "chessqueries" else make_submission(a)
    name = a.system if a.system == "chessqueries" else f"{Path(a.sub).name}:{a.variant}"
    results = []
    with open(f"{out}.jsonl", "x") as jf:
        for split in inventory["splits"]:
            records = [r for r in inventory["rows"] if r["split"] == split]
            ex = sq = empty = occupied = occupied_correct = 0
            hist = {}
            t0 = time.time()
            with open(f"{out}.{split}.tsv", "x") as tf:
                tf.write("id\tgt\tpred\tn_wrong\n")
                for record in records:
                    i, gt = record["id"], record["fen"]
                    pred = placement(predict(a.root / split / "images" / f"{i}.png"))
                    g, p = (
                        cells(gt),
                        cells(pred),
                    )  # validates both complete 64-square arrays
                    nw = sum(x != y for x, y in zip(g, p))
                    ex += nw == 0
                    sq += 64 - nw
                    empty += g.count(".")
                    occupied += 64 - g.count(".")
                    occupied_correct += sum(x != "." and x == y for x, y in zip(g, p))
                    hist[nw] = hist.get(nw, 0) + 1
                    tf.write(f"{i}\t{gt}\t{pred}\t{nw}\n")
            n = len(records)
            row = {
                "system": name,
                "split": split,
                "n": n,
                "exact_boards": ex,
                "correct_squares": sq,
                "total_squares": 64 * n,
                "exact": round(100 * ex / n, 2),
                "per_square": round(100 * sq / (64 * n), 3),
                "all_empty_per_square": round(100 * empty / (64 * n), 3),
                "occupied_square_accuracy": round(100 * occupied_correct / occupied, 3)
                if occupied
                else None,
                "ms_per_img": round(1000 * (time.time() - t0) / n, 1),
                "timing": (
                    "wall; includes first forward; device managed by external system"
                    if a.system != "chessqueries"
                    else "wall; includes first forward; not isolated GPU latency"
                    if a.device == "cuda"
                    else "wall; includes first forward; CPU"
                ),
                "wrong_hist": dict(sorted(hist.items())),
            }
            results.append(row)
            jf.write(json.dumps(row) + "\n")
            jf.flush()
            print(json.dumps(row), flush=True)
    with open(f"{out}.md", "x") as f:
        f.write(f"# {name} on frozen benchmark\n\n")
        if a.parent_freeze:
            f.write(
                f"Selection coverage: **{inventory['selection_complete']}**; "
                f"{inventory['selected_n']}/{inventory['frozen_n']} selected rows; "
                f"parent inventory: {inventory['parent_frozen_n']} rows. "
                "Full benchmark coverage: **False**.\n\n"
            )
        else:
            f.write(
                f"Full benchmark coverage: **{inventory['benchmark_complete']}**; "
                f"{inventory['selected_n']}/{inventory['frozen_n']} frames.\n\n"
            )
        f.write(
            "| split | n | exact % | per-square % | all-empty % | wall ms/img |\n"
            "|---|---:|---:|---:|---:|---:|\n"
        )
        f.writelines(
            f"| {r['split']} | {r['n']} | {r['exact']} | {r['per_square']} | "
            f"{r['all_empty_per_square']} | {r['ms_per_img']} |\n"
            for r in results
        )
    verify_unchanged(a.root, inventory, receipt["model_inputs_sha256"])
    if a.parent_freeze and sha256(a.parent_freeze) != inventory["parent_freeze_sha256"]:
        raise ValueError("parent inventory changed during inference")
    finish_output(
        out,
        dict(receipt, system=name, results=results, status="complete_for_selection"),
    )
    print("completed", f"{out}.complete.json")


if __name__ == "__main__":
    main()
