#!/usr/bin/env python3
"""Exclusion check for the real-photo shoot sheet.

Streams pools of piece placements and reports any candidate position that matches one, exactly
or under the eight dihedral board symmetries. Pools may be JSONL with a `fen` or `gt_fen`
field, a plain FEN-per-line file, a TSV/CSV with a FEN column, or a Zstandard-compressed CSV
(the Lichess puzzle export). Only the placement field is ever compared.

Two kinds of pool:
  --pool NAME=PATH        forbidden: any exact or dihedral match is a hit
  --must-match NAME=PATH  required: every candidate must occur exactly in this pool (used to
                          prove the sheet was drawn from the v2 dev groups and nothing else)

  python tools/paper/real_photo_exclusion.py --candidates fens.txt \
      --must-match v2_dev=tools/reports/paper/real_photo_<date>/v2_dev_groups.jsonl \
      --pool v2_adaptation=<v2 dir>/design.adaptation.groups.jsonl \
      --pool lichess=~/bench_sources/lichess/lichess_db_puzzle.csv.zst \
      --out tools/reports/paper/real_photo_<date>/exclusion_<host>.json

Exit codes: 0 clean; 1 malformed or unreadable input; 3 a forbidden pool matched or a required
pool is missing a candidate; 4 a pool read zero rows. A pool that cannot be read completely is a
failed check, never a pass.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
from pathlib import Path

PIECES = frozenset("pnbrqkPNBRQK")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved(path):
    return Path(path).expanduser().resolve(strict=False)


def paths_alias(left, right):
    left_path = Path(left).expanduser()
    right_path = Path(right).expanduser()
    if resolved(left_path) == resolved(right_path):
        return True
    return (
        left_path.exists()
        and right_path.exists()
        and os.path.samefile(left_path, right_path)
    )


def refuse_output_alias(output, inputs):
    for source in inputs:
        if paths_alias(output, source):
            raise SystemExit(
                f"output {output} aliases an input; refusing to overwrite evidence"
            )


def grid(placement):
    rows = []
    for rank in placement.split("/"):
        row = []
        for char in rank:
            if char in "12345678":
                row.extend("." * int(char))
            elif char in PIECES:
                row.append(char)
            else:
                return None
        rows.append(row)
    if len(rows) != 8 or any(len(r) != 8 for r in rows):
        return None
    return rows


def to_placement(rows):
    out = []
    for row in rows:
        text = ""
        empty = 0
        for char in row:
            if char == ".":
                empty += 1
                continue
            if empty:
                text += str(empty)
                empty = 0
            text += char
        if empty:
            text += str(empty)
        out.append(text)
    return "/".join(out)


def dihedral(placement):
    """The eight rotations and reflections of a placement, as placement strings."""
    rows = grid(placement)
    if rows is None:
        return set()
    forms = set()
    current = rows
    for _ in range(4):
        # rotate 90 degrees
        current = [list(r) for r in zip(*current[::-1])]
        forms.add(to_placement(current))
        forms.add(to_placement([r[::-1] for r in current]))
    return forms


def placement_of(text):
    if not isinstance(text, str) or not text.strip():
        return None
    placement = text.split()[0]
    rows = grid(placement)
    return placement if rows is not None and to_placement(rows) == placement else None


def pool_placement(path, line_number, text):
    placement = placement_of(text)
    if placement is None:
        raise SystemExit(f"{path}:{line_number}: invalid or missing piece placement")
    return placement


def csv_fen_field(path, reader):
    supported = ("FEN", "fen", "gt_fen", "gt")
    fieldnames = reader.fieldnames or []
    matches = [field for field in fieldnames if field in supported]
    if len(matches) != 1:
        raise SystemExit(
            f"{path}: expected exactly one supported FEN column ({', '.join(supported)})"
        )
    return matches[0]


def iter_pool(path):
    path = Path(path).expanduser()
    name = path.name.lower()
    if name.endswith(".zst"):
        try:
            import zstandard
        except ImportError as exc:
            raise SystemExit("zstandard is required to read .zst pools") from exc

        with path.open("rb") as raw:
            stream = zstandard.ZstdDecompressor().stream_reader(raw)
            reader = csv.DictReader(
                io.TextIOWrapper(stream, encoding="utf-8"), strict=True
            )
            field = csv_fen_field(path, reader)
            for line_number, row in enumerate(reader, 2):
                yield pool_placement(path, line_number, row.get(field))
        return
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="strict") as handle:
        if name.endswith((".jsonl", ".ndjson")) or ".jsonl" in name:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"{path}:{line_number}: malformed JSON: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict):
                    raise SystemExit(
                        f"{path}:{line_number}: JSON row must be an object"
                    )
                for key in ("fen", "gt_fen", "board_fen", "placement"):
                    if key in row:
                        yield pool_placement(path, line_number, row[key])
                        break
                else:
                    raise SystemExit(
                        f"{path}:{line_number}: missing piece-placement field"
                    )
            return
        if name.endswith((".csv", ".tsv")):
            delimiter = "\t" if name.endswith(".tsv") else ","
            reader = csv.DictReader(handle, delimiter=delimiter, strict=True)
            field = csv_fen_field(path, reader)
            for line_number, row in enumerate(reader, 2):
                yield pool_placement(path, line_number, row.get(field))
            return
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                yield pool_placement(path, line_number, line)


def parse_pool(entry):
    name, sep, path = entry.partition("=")
    if not sep or not name or not path:
        raise SystemExit(f"pool must be NAME=PATH: {entry}")
    return name, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument(
        "--pool",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="forbidden pool; any exact or dihedral match is a hit",
    )
    parser.add_argument(
        "--must-match",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="required pool; every candidate must occur in it exactly",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.pool and not args.must_match:
        raise SystemExit("no pools given; refusing to report clean")

    parsed_pools = [("forbidden", *parse_pool(entry)) for entry in args.pool]
    parsed_pools += [("required", *parse_pool(entry)) for entry in args.must_match]
    names = [name for _, name, _ in parsed_pools]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate pool name; refusing to overwrite pool evidence")

    out = Path(args.out).expanduser()
    input_paths = [args.candidates, *(path for _, _, path in parsed_pools)]
    refuse_output_alias(out, input_paths)
    temporary_out = out.with_name(out.name + ".tmp")
    refuse_output_alias(temporary_out, input_paths)
    if out.exists() or out.is_symlink():
        raise SystemExit(f"output already exists; choose a fresh report path: {out}")
    if temporary_out.exists() or temporary_out.is_symlink():
        raise SystemExit(
            f"temporary output already exists; remove it explicitly: {temporary_out}"
        )

    candidate_path = Path(args.candidates).expanduser()
    candidate_bytes = candidate_path.read_bytes()
    try:
        candidate_text = candidate_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{candidate_path}: candidates are not valid UTF-8") from exc
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    candidates = []
    for line_number, line in enumerate(candidate_text.splitlines(), 1):
        if line.strip():
            candidates.append(pool_placement(candidate_path, line_number, line))
    if not candidates:
        raise SystemExit("no candidates; refusing to report clean")
    if len(candidates) != len(set(candidates)):
        raise SystemExit("duplicate candidates; refusing to report clean")

    exact = {fen: fen for fen in candidates}
    symmetric = {}
    for fen in candidates:
        forms = dihedral(fen)
        if not forms:
            raise SystemExit(
                f"candidate is not a readable placement, refusing to report clean: {fen}"
            )
        for form in forms:
            symmetric.setdefault(form, set()).add(fen)

    pools = {}
    hits = []
    errors = []
    for _, name, path in (pool for pool in parsed_pools if pool[0] == "forbidden"):
        source_sha256 = sha256_file(path)
        seen = 0
        for placement in iter_pool(path):
            if not placement:
                continue
            seen += 1
            if placement in exact:
                hits.append(
                    {"pool": name, "candidate": exact[placement], "match": "exact"}
                )
            elif placement in symmetric:
                for fen in sorted(symmetric[placement]):
                    hits.append({"pool": name, "candidate": fen, "match": "dihedral"})
        if sha256_file(path) != source_sha256:
            raise SystemExit(f"pool {name} changed while it was being checked: {path}")
        pools[name] = {
            "path": str(path),
            "sha256": source_sha256,
            "rows": seen,
            "kind": "forbidden",
        }
        if seen == 0:
            errors.append(f"pool {name} read zero rows: {path}")
        print(f"{name}: {seen} placements", file=sys.stderr)

    for _, name, path in (pool for pool in parsed_pools if pool[0] == "required"):
        source_sha256 = sha256_file(path)
        seen = 0
        found = set()
        for placement in iter_pool(path):
            if not placement:
                continue
            seen += 1
            if placement in exact:
                found.add(placement)
        pools[name] = {
            "path": str(path),
            "sha256": source_sha256,
            "rows": seen,
            "kind": "required",
            "candidates_found": len(found),
        }
        if sha256_file(path) != source_sha256:
            raise SystemExit(f"pool {name} changed while it was being checked: {path}")
        if seen == 0:
            errors.append(f"pool {name} read zero rows: {path}")
        for fen in candidates:
            if fen not in found:
                hits.append(
                    {"pool": name, "candidate": fen, "match": "missing_from_required"}
                )
        print(
            f"{name}: {seen} placements, {len(found)}/{len(candidates)} candidates present",
            file=sys.stderr,
        )

    report = {
        "schema": "real-photo-exclusion/v2",
        "validation_scope": "supplied_pools_only",
        "candidates": len(candidates),
        "candidates_sha256": candidate_sha256,
        "pools": pools,
        "hits": hits,
        "errors": errors,
        "clean": not hits and not errors,
    }
    if sha256_file(candidate_path) != candidate_sha256:
        raise SystemExit(
            f"candidates changed while pools were being checked: {candidate_path}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with temporary_out.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary_out, out)
    except FileExistsError as exc:
        temporary_out.unlink()
        raise SystemExit(
            f"output appeared during the run; refusing to replace it: {out}"
        ) from exc
    temporary_out.unlink()
    print(
        json.dumps(
            {
                "clean": report["clean"],
                "hits": len(hits),
                "errors": errors,
                "out": str(out),
            }
        )
    )
    if errors:
        return 4
    return 0 if report["clean"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
