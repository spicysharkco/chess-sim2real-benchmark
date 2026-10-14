#!/usr/bin/env python3
"""Rebuild the ChessCog `render/<split>/NNNN.{png,json}` layout that the ChessQueries loaders expect
from our converted copy (loaders/chesscog.py output: images/, labels/, manifest.jsonl with
`source_split`, `fen`, `id`). Fallback for when OSF is unreachable; the OSF original, when it lands,
is compared against this by tools/paper/cq_baseline.py rows (same 342 ids, same FENs expected).

  python tools/paper/chesscog_render_from_manifest.py --src $DATA_ROOT/ext/chesscog \
      --out $DATA_ROOT/ext/chesscog_from_manifest --splits test val
"""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"])
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    counts = {s: 0 for s in a.splits}
    for line in open(src / "manifest.jsonl"):
        r = json.loads(line)
        split = r.get("source_split")
        if split not in counts or r.get("status", "ok") != "ok":
            continue
        rid = r["id"] if isinstance(r["id"], str) else f"{int(r['id']):06d}"
        img = src / "images" / f"{rid}.png"
        if not img.is_file():
            raise SystemExit(f"missing image {img}")
        fen = r.get("fen") or (src / "labels" / f"{rid}.txt").read_text().split()[0]
        d = out / "render" / split
        d.mkdir(parents=True, exist_ok=True)
        link = d / f"{rid}.png"
        if not link.exists():
            os.symlink(img.resolve(), link)
        (d / f"{rid}.json").write_text(json.dumps({"fen": fen, "source_id": r.get("source_id", r.get("source")),
                                                   "from": "loaders/chesscog.py manifest"}, indent=1))
        counts[split] += 1
    print(counts)


if __name__ == "__main__":
    main()
