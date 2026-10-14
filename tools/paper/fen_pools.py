#!/usr/bin/env python3
"""FEN pools and the leakage check for the benchmark freeze.

Collects the piece-placement field of every labelled position we know about (public sets, our official
train set, every render set, every holdout), one pool per source, and reports (a) pairwise overlaps
between pools and (b) which of a candidate FEN list (the benchmark test split about to be frozen) already
occurs in any pool, under three matchers: exact, 4-rotation/mirror (the recogniser cannot tell a
transposed board from a real one), and material-only (same multiset of pieces, a weaker "near-duplicate").

  python tools/paper/fen_pools.py --pool chessred=~/ext/chessred/annotations.json \
      --pool chesscog=~/ext/chesscog/manifest.jsonl --pool slcc=~/ext/slcc_train-host/dataset/annotations.json \
      [--check candidate_fens.txt] --out tools/reports/paper/fen_pools_<date>

A --pool value may be: a ChessReD annotations.json (FEN rebuilt from `annotations.pieces`), a ChessQueries
SLCC annotations.json (`samples[].gt_fen`), a CVChess labels json (list with a `fen` field), a
manifest.jsonl with a `fen` field, a directory holding labels/*.txt or *.txt / *.fen files (first token of
each file), a CSV or Zstandard-compressed CSV with a `FEN` column, or a plain text file with one FEN per
line. Only the placement field is kept; CSV move columns are never applied.
"""

import argparse
import csv
import datetime
import gzip
import hashlib
import json
import multiprocessing
import subprocess
from collections import Counter
from numbers import Integral
from pathlib import Path

PIECES = "PNBRQKpnbrqk"


def placement(fen):
    """Validate all 64 squares and canonicalize the placement field only."""
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("empty FEN")
    ranks = fen.split()[0].split("/")
    if len(ranks) != 8:
        raise ValueError("FEN must contain eight ranks")
    rows = []
    for rank in ranks:
        row = ""
        for ch in rank:
            if ch in "12345678":
                row += "." * int(ch)
            elif ch in PIECES:
                row += ch
            else:
                raise ValueError(f"invalid FEN symbol: {ch!r}")
        if len(row) != 8:
            raise ValueError("each FEN rank must expand to eight squares")
        rows.append(row)
    return compress(rows)


def cells(fen):
    return list("".join(expand(fen)))


def labels_to_fen(labels):
    """ChessQueries class IDs in canonical a8..h1 order, without truncation."""
    if len(labels) != 64 or any(
        isinstance(c, bool) or not isinstance(c, Integral) or not 0 <= c < 13
        for c in labels
    ):
        raise ValueError("model output must contain exactly 64 class IDs in [0, 12]")
    symbols = "." + PIECES
    return compress(
        ["".join(symbols[c] for c in labels[i : i + 8]) for i in range(0, 64, 8)]
    )


def expand(fen):
    rows = []
    for r in placement(fen).split("/"):
        row = ""
        for ch in r:
            row += "." * int(ch) if ch.isdigit() else ch
        rows.append(row)
    return rows


def compress(rows):
    out = []
    for row in rows:
        s, run = "", 0
        for ch in row:
            if ch == ".":
                run += 1
            else:
                if run:
                    s += str(run)
                    run = 0
                s += ch
        if run:
            s += str(run)
        out.append(s)
    return "/".join(out)


def sym_key(fen):
    """Canonical key over the 8 symmetries of the square (4 rotations x mirror)."""
    g = expand(fen)
    variants = []
    for _ in range(4):
        variants.append(compress(g))
        variants.append(compress([row[::-1] for row in g]))
        g = ["".join(g[7 - c][r] for c in range(8)) for r in range(8)]  # rotate 90
    return min(variants)


def material_key(fen):
    canonical = placement(fen)
    counts = Counter(canonical)
    return "".join(f"{p}{counts[p]}" for p in PIECES)


def symmetry_keys(fens, workers=1):
    """Build a D4-key set, optionally using a bounded process pool."""

    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    if workers == 1:
        return {sym_key(fen) for fen in fens}
    with multiprocessing.Pool(processes=workers) as pool:
        return set(pool.imap_unordered(sym_key, fens, chunksize=512))


def _csv_fens(stream, source):
    def nonempty_lines():
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{source}: empty CSV row at line {line_number}")
            yield line

    reader = csv.DictReader(nonempty_lines(), strict=True)
    if reader.fieldnames is None or "FEN" not in reader.fieldnames:
        raise ValueError(f"{source}: CSV requires a FEN column")
    fens = []
    for row_number, row in enumerate(reader, start=2):
        fen = row.get("FEN")
        if not isinstance(fen, str) or not fen.strip():
            raise ValueError(f"{source}: empty FEN at CSV row {row_number}")
        # Lichess puzzle rows store the pre-move position in FEN.  Moves are
        # deliberately ignored: the training corpus consumed this exact state.
        fens.append(placement(fen))
    return fens


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return _csv_fens(stream, path)


def _read_zstd_csv(path):
    try:
        process = subprocess.Popen(
            ["zstd", "-dc", "--", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        raise ValueError(f"{path}: cannot start zstd: {exc}") from exc
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("zstd pipes were not created")
    parse_error = None
    try:
        fens = _csv_fens(process.stdout, path)
    except (ValueError, csv.Error, UnicodeError, OSError) as exc:
        parse_error = exc
        # Drain the pipe so zstd reaches its natural exit status.  This lets a
        # truncated frame report decompression failure rather than whichever
        # partial CSV error happened to surface first.
        for _ in process.stdout:
            pass
        fens = []
    process.stdout.close()
    stderr = process.stderr.read()
    returncode = process.wait()
    process.stderr.close()
    if returncode:
        detail = stderr.strip() or f"exit status {returncode}"
        raise ValueError(f"{path}: zstd decompression failed: {detail}")
    if parse_error is not None:
        raise parse_error
    return fens


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pool(spec):
    p = Path(spec).expanduser()
    fens = []
    if p.is_dir():
        files = (
            sorted((p / "labels").glob("*.txt"))
            if (p / "labels").is_dir()
            else sorted(list(p.glob("*.txt")) + list(p.glob("*.fen")))
        )
        if not files and list(
            p.glob("*.json")
        ):  # ChessCog render/<split>/NNNN.json with "fen"
            for f in sorted(p.glob("*.json")):
                with f.open() as stream:
                    fen = json.load(stream).get("fen")
                if fen:
                    fens.append(placement(fen))
            return fens
        if not files:
            # render sets keep a per-image FEN in metadata jsonl?  try manifest.jsonl, then meta_raw.tsv
            for cand in ("manifest.jsonl", "meta_raw.tsv"):
                if (p / cand).is_file():
                    return read_pool(p / cand)
            raise SystemExit(
                f"{p}: no labels/*.txt, *.txt, *.fen, manifest.jsonl or meta_raw.tsv"
            )
        for f in files:
            t = f.read_text().strip()
            if t:
                fens.append(placement(t))
        return fens
    if p.suffix == ".jsonl" or p.name.endswith(".jsonl.gz"):
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt") as stream:
            for line in stream:
                if line.strip():
                    r = json.loads(line)
                    if r.get("fen"):
                        fens.append(placement(r["fen"]))
        return fens
    if p.suffix == ".csv":
        return _read_csv(p)
    if p.name.endswith(".csv.zst"):
        return _read_zstd_csv(p)
    if p.suffix == ".tsv":
        with p.open() as stream:
            for line in stream:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and "/" in parts[1]:
                    fens.append(placement(parts[1]))
        return fens
    if p.suffix == ".json":
        with p.open() as stream:
            d = json.load(stream)
        if isinstance(d, dict) and "samples" in d:  # SLCC
            return [placement(s["gt_fen"]) for s in d["samples"]]
        if (
            isinstance(d, dict) and "annotations" in d and "pieces" in d["annotations"]
        ):  # ChessReD
            cat = {c["id"]: c["name"] for c in d["categories"]}
            sym = {
                "white-pawn": "P",
                "white-knight": "N",
                "white-bishop": "B",
                "white-rook": "R",
                "white-queen": "Q",
                "white-king": "K",
                "black-pawn": "p",
                "black-knight": "n",
                "black-bishop": "b",
                "black-rook": "r",
                "black-queen": "q",
                "black-king": "k",
            }
            boards = {}
            for pc in d["annotations"]["pieces"]:
                name = cat[pc["category_id"]]
                if name not in sym:
                    continue
                sq = pc["chessboard_position"]
                col, row = "abcdefgh".index(sq[0]), 8 - int(sq[1])
                boards.setdefault(pc["image_id"], [["."] * 8 for _ in range(8)])[row][
                    col
                ] = sym[name]
            for im in d["images"]:
                g = boards.get(im["id"], [["."] * 8 for _ in range(8)])
                fens.append(compress(["".join(r) for r in g]))
            return fens
        if isinstance(d, list):  # CVChess labels
            for r in d:
                f = r.get("fen") or r.get("gt_fen") or r.get("label")
                if f:
                    fens.append(placement(f))
            return fens
        raise SystemExit(f"{p}: unknown json layout")
    # plain text
    with p.open() as stream:
        for line in stream:
            line = line.strip()
            if line and not line.startswith("#"):
                fens.append(placement(line))
    return fens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="append", default=[], help="name=path")
    ap.add_argument("--check", default=None, help="candidate FENs, one per line")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.pool:
        ap.error("at least one explicitly named source pool is required")
    out = Path(a.out)
    if any(Path(f"{out}.{suffix}").exists() for suffix in ("json", "md")):
        raise FileExistsError(f"audit output already exists: {out}")
    pools = {}
    for spec in a.pool:
        name, path = spec.split("=", 1)
        if not name or name in pools:
            raise ValueError(f"empty/duplicate pool name: {name!r}")
        fens = read_pool(path)
        if not fens:
            raise ValueError(f"empty source pool: {name}")
        pools[name] = {
            "n": len(fens),
            "exact": set(fens),
            "sym": symmetry_keys(fens, workers=a.workers),
            "material": Counter(material_key(f) for f in fens),
            "path": str(path),
        }
        print(
            f"{name:24s} {len(fens):7d} positions, {len(pools[name]['exact']):7d} unique, {len(pools[name]['sym']):7d} up to symmetry"
        )
    names = list(pools)
    overlap = {}
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            ex = len(pools[x]["exact"] & pools[y]["exact"])
            sy = len(pools[x]["sym"] & pools[y]["sym"])
            overlap[f"{x}|{y}"] = {"exact": ex, "sym": sy}
            if ex or sy:
                print(f"  overlap {x} ~ {y}: exact {ex}, up to symmetry {sy}")
    report = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="minutes"),
        "pools": {
            k: {
                "n": v["n"],
                "unique": len(v["exact"]),
                "unique_sym": len(v["sym"]),
                "path": v["path"],
                "placement_sha256": hashlib.sha256(
                    "\n".join(sorted(v["exact"])).encode()
                ).hexdigest(),
            }
            for k, v in pools.items()
        },
        "overlap": overlap,
    }
    if a.check:
        cand = read_pool(a.check)
        if not cand:
            raise ValueError("empty candidate population")
        hits = []
        for f in cand:
            k, m = sym_key(f), material_key(f)
            where = {
                n: (
                    "exact"
                    if f in p["exact"]
                    else "sym"
                    if k in p["sym"]
                    else "material"
                    if m in p["material"]
                    else None
                )
                for n, p in pools.items()
            }
            where = {n: w for n, w in where.items() if w}
            if where:
                hits.append({"fen": f, "where": where})
        candidate_path = Path(a.check).expanduser().resolve()
        if not candidate_path.is_file():
            raise ValueError("--check provenance requires a regular candidate file")
        candidate_sym = symmetry_keys(cand, workers=a.workers)
        canonical_set_sha = hashlib.sha256(
            "\n".join(sorted(set(cand))).encode()
        ).hexdigest()
        report["check"] = {
            "n": len(cand),
            "unique": len(set(cand)),
            "unique_sym": len(candidate_sym),
            "candidate_path": str(candidate_path),
            "candidate_content_sha256": _sha256_file(candidate_path),
            "candidate_canonical_set_sha256": canonical_set_sha,
            "leaked": len(
                [
                    h
                    for h in hits
                    if any(w in ("exact", "sym") for w in h["where"].values())
                ]
            ),
            "material_only": len(
                [h for h in hits if all(w == "material" for w in h["where"].values())]
            ),
            "hits": hits,
        }
        report["check"]["clean"] = report["check"]["leaked"] == 0 and report["check"][
            "unique_sym"
        ] == len(cand)
        report["check"]["coverage"] = (
            "supplied named pools only; not proof of complete training lineage"
        )
        print(
            f"check: {len(cand)} candidates, {report['check']['leaked']} leak (exact/symmetry), {report['check']['material_only']} material-only near-duplicates"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.json", "x") as stream:
        json.dump(report, stream, indent=1)
    with open(f"{out}.md", "x") as f:
        f.write(
            f"# FEN pools {report['date']}\n\n| pool | positions | unique | unique up to symmetry |\n|---|---:|---:|---:|\n"
        )
        for k, v in report["pools"].items():
            f.write(f"| {k} | {v['n']} | {v['unique']} | {v['unique_sym']} |\n")
        f.write("\n| pair | exact | up to symmetry |\n|---|---:|---:|\n")
        for k, v in overlap.items():
            if v["exact"] or v["sym"]:
                f.write(f"| {k} | {v['exact']} | {v['sym']} |\n")
        if a.check:
            c = report["check"]
            f.write(
                f"\ncheck `{a.check}`: {c['n']} candidates ({c['unique']} unique), **{c['leaked']} leaked** (exact or symmetry), {c['material_only']} material-only.\n"
            )
    print("wrote", f"{out}.json", f"{out}.md")
    if a.check and not report["check"]["clean"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
