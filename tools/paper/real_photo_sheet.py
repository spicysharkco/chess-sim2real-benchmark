#!/usr/bin/env python3
"""Generate the printable position sheet for the own real-photo test set.

Two position sources:

* ``--from-groups PATH`` (the frozen 2026-09-11 sheet uses this): positions are the benchmark
  v2 development groups themselves. Every row of PATH must carry ``partition: dev``; final_test
  rows are refused. Each photographed position then has a rendered twin in every factor cell of
  the benchmark, so render-to-photo comparisons are paired by position. Selection is a seeded
  shuffle per density class at the benchmark's one-in-ten sparse cadence; no per-position model
  outcome is consulted.
* fresh sampling (no ``--from-groups``): legal random self-play with the render benchmark's
  dense/sparse cadence (one sparse position in ten, ``rendering/generate_geometry.py``), with
  ``--exclude`` files of placements to skip under the eight board symmetries.

Both sources are filtered to the declared sparse/dense piece-count bands and so every position
can be set up with one standard Staunton set: at
most one queen, two rooks, two bishops, two knights and eight pawns per side, exactly one king
each. Promotion-heavy positions are dropped rather than resampled in place, so the stream stays
deterministic for a given seed.

  uv run --with chess python tools/paper/real_photo_sheet.py \\
      --from-groups tools/reports/paper/real_photo_<date>/v2_dev_groups.jsonl \\
      --out tools/reports/paper/real_photo_<date>

Outputs, all deterministic for a seed:
  shoot_sheet.jsonl   one row per position: id, fen, piece_count, density, near_side, lighting,
                      paired, conventions, source_group_id
  SHEET.md            printable sheet, ten positions per page, with a board diagram each
  fens.txt            the placements, one per line, the input to the exclusion gate
  sheet.manifest.json seed, counts, sha256 of the jsonl and of the group file

The exclusion gate (tools/paper/real_photo_exclusion.py) is a separate step and must report
``clean: true`` before any photograph is taken.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import chess

STANDARD_SET = {
    chess.KING: 1,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 2,
    chess.KNIGHT: 2,
    chess.PAWN: 8,
}

# rendering/generate_geometry.py: every tenth position is sparse; piece-count bands per class
SPARSE_EVERY = 10
SPARSE_BAND = (6, 15)
DENSE_BAND = (20, 34)


def play_position(rng, sparse):
    """Legal random self-play, capture-biased, matching rendering/generate_geometry.py."""
    board = chess.Board()
    stop_at = rng.randint(65, 125) if sparse else rng.randint(12, 46)
    for ply in range(stop_at):
        moves = list(board.legal_moves)
        if not moves:
            break
        captures = [move for move in moves if board.is_capture(move)]
        probability = 0.58 if sparse else 0.28
        pool = (
            captures if captures and ply > 12 and rng.random() < probability else moves
        )
        board.push(rng.choice(pool))
    return board


def fits_standard_set(board):
    for colour in (chess.WHITE, chess.BLACK):
        for piece_type, allowed in STANDARD_SET.items():
            if len(board.pieces(piece_type, colour)) > allowed:
                return False
        if len(board.pieces(chess.KING, colour)) != 1:
            return False
    return True


def density_of(piece_count):
    return "sparse" if piece_count <= SPARSE_BAND[1] else "dense"


def sample(count, seed, excluded=frozenset()):
    """Fresh positions. `excluded` holds placements already in some pool, plus their symmetries."""
    rng = random.Random(seed)
    seen = set()
    rows = []
    while len(rows) < count:
        sparse = len(rows) % SPARSE_EVERY == SPARSE_EVERY - 1
        board = play_position(rng, sparse)
        fen = board.board_fen()
        pieces = len(board.piece_map())
        low, high = SPARSE_BAND if sparse else DENSE_BAND
        wanted = low <= pieces <= high
        if not wanted or fen in seen or fen in excluded or not fits_standard_set(board):
            continue
        seen.add(fen)
        rows.append({"fen": fen, "piece_count": pieces, "density": density_of(pieces)})
    return rows


def load_groups(path, raw_bytes=None):
    """Benchmark v2 development groups. Refuses any row that is not partition `dev`."""
    rows = []
    ids = set()
    if raw_bytes is None:
        raw_bytes = Path(path).expanduser().read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{path}: group file is not valid UTF-8") from exc
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: malformed JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{number}: JSON row must be an object")
        if row.get("partition") != "dev":
            raise SystemExit(
                f"{path}:{number}: partition is {row.get('partition')!r}, only dev groups may be "
                "photographed"
            )
        for key in ("id", "fen", "piece_count"):
            if key not in row:
                raise SystemExit(f"{path}:{number}: missing {key}")
        if not isinstance(row["id"], str) or not row["id"]:
            raise SystemExit(f"{path}:{number}: id must be a non-empty string")
        if row["id"] in ids:
            raise SystemExit(f"{path}:{number}: duplicate id {row['id']!r}")
        ids.add(row["id"])
        if not isinstance(row["fen"], str):
            raise SystemExit(f"{path}:{number}: fen must be a string")
        try:
            board = chess.Board(row["fen"] + " w - - 0 1")
        except ValueError as exc:
            raise SystemExit(
                f"{path}:{number}: invalid placement: {row['fen']!r}"
            ) from exc
        if board.board_fen() != row["fen"]:
            raise SystemExit(
                f"{path}:{number}: placement does not round-trip: {row['fen']}"
            )
        board_black = chess.Board(row["fen"] + " b - - 0 1")
        if not board.is_valid() and not board_black.is_valid():
            raise SystemExit(
                f"{path}:{number}: placement is not legal for either side to move: {row['fen']}"
            )
        actual_piece_count = len(board.piece_map())
        if (
            not isinstance(row["piece_count"], int)
            or isinstance(row["piece_count"], bool)
            or row["piece_count"] != actual_piece_count
        ):
            raise SystemExit(
                f"{path}:{number}: piece_count {row['piece_count']!r} does not match "
                f"placement count {actual_piece_count}"
            )
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no groups")
    return rows


def select_groups(groups, count, seed):
    """Seeded selection of groups that fit one standard set, at the benchmark's sparse cadence.

    Sparse and dense groups are shuffled separately with the seed; the sheet then takes a sparse
    position at every tenth slot (index 9, 19, ...) and a dense one elsewhere, so the sparse share
    matches the benchmark's one-in-ten and the paired arm (the first rows) inherits it.
    """
    eligible = {"sparse": [], "dense": []}
    seen = set()
    for row in groups:
        board = chess.Board(row["fen"] + " w - - 0 1")
        if board.board_fen() != row["fen"]:
            raise SystemExit(
                f"group {row['id']}: placement does not round-trip: {row['fen']}"
            )
        if row["fen"] in seen:
            raise SystemExit(
                f"group {row['id']}: duplicate placement in the group file"
            )
        seen.add(row["fen"])
        piece_count = row["piece_count"]
        in_declared_band = (
            SPARSE_BAND[0] <= piece_count <= SPARSE_BAND[1]
            or DENSE_BAND[0] <= piece_count <= DENSE_BAND[1]
        )
        if fits_standard_set(board) and in_declared_band:
            eligible[density_of(row["piece_count"])].append(row)
    rng = random.Random(seed)
    for rows in eligible.values():
        rng.shuffle(rows)
    chosen = []
    for index in range(count):
        density = "sparse" if index % SPARSE_EVERY == SPARSE_EVERY - 1 else "dense"
        if not eligible[density]:
            raise SystemExit(
                f"ran out of eligible {density} groups at position {index + 1}"
            )
        chosen.append(eligible[density].pop())
    rows = [
        {
            "fen": row["fen"],
            "piece_count": row["piece_count"],
            "density": density_of(row["piece_count"]),
            "source_group_id": row["id"],
        }
        for row in chosen
    ]
    total_eligible = len(chosen) + sum(len(rows_) for rows_ in eligible.values())
    return rows, total_eligible


def load_exclusions(paths):
    """Every placement in the given files, plus forms, and raw source hashes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "real_photo_exclusion", Path(__file__).with_name("real_photo_exclusion.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = set()
    sources = []
    for path in paths:
        source = Path(path).expanduser()
        raw = source.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SystemExit(f"{path}: exclusion file is not valid UTF-8") from exc
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            placement = module.placement_of(line)
            if placement is None:
                raise SystemExit(
                    f"{path}:{line_number}: exclusion entry is not a canonical placement"
                )
            forms = module.dihedral(placement)
            if not forms:
                raise SystemExit(
                    f"exclusion entry is not a readable placement: {placement}"
                )
            out |= forms
    return out, sources


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


def refuse_output_aliases(out, inputs):
    destinations = [
        Path(out) / "shoot_sheet.jsonl",
        Path(out) / "SHEET.md",
        Path(out) / "fens.txt",
        Path(out) / "sheet.manifest.json",
    ]
    for destination in destinations:
        for source in inputs:
            if paths_alias(destination, source):
                raise SystemExit(
                    f"generated output {destination} aliases an input; refusing to overwrite evidence"
                )
        if destination.exists() or destination.is_symlink():
            raise SystemExit(
                f"generated output already exists; choose a fresh output directory: {destination}"
            )


def verify_sources_unchanged(sources):
    for source in sources:
        digest = hashlib.sha256(
            Path(source["path"]).expanduser().read_bytes()
        ).hexdigest()
        if digest != source["sha256"]:
            raise SystemExit(
                f"input changed while the sheet was generated: {source['path']}"
            )


def assign(rows, paired):
    """Near side, lighting and the paired-convention arm, all by position index."""
    out = []
    for index, row in enumerate(rows):
        entry = {
            "id": f"p{index + 1:03d}",
            "fen": row["fen"],
            "piece_count": row["piece_count"],
            "density": row["density"],
            # two thirds white near, one third black near
            "near_side": "black" if index % 3 == 2 else "white",
            # alternating room lighting condition
            "lighting": "window_behind" if index % 2 == 0 else "lamp_side",
            # the first `paired` positions are shot under both board conventions
            "paired": index < paired,
            "conventions": ["a1_dark", "a1_light"] if index < paired else ["a1_dark"],
            # boards alternate so no arm sits on one physical board only
            "board": "A" if index % 2 == 0 else "B",
        }
        if "source_group_id" in row:
            entry["source_group_id"] = row["source_group_id"]
        out.append(entry)
    return out


def diagram(fen):
    """Eight text ranks, rank 8 first, '.' for an empty square."""
    lines = []
    for rank in fen.split("/"):
        line = []
        for char in rank:
            if char.isdigit():
                line.append("." * int(char))
            else:
                line.append(char)
        lines.append(" ".join("".join(line)))
    return lines


def render_sheet(rows, seed, source):
    files = "a b c d e f g h"
    out = [
        "# Real-photo shoot sheet",
        "",
        (
            f"Seed {seed}. {len(rows)} positions ({source}), {sum(r['paired'] for r in rows)} of them "
            "shot under both board conventions. Uppercase is white, lowercase is black, rank 8 is the "
            "top row, and the a-file is on the left as White sees it."
        ),
        "",
        (
            "For every position: set it up on the board the row names (A or B), then shoot from low "
            "(about 10 degrees, phone almost at table height), medium (about 35) and high (about 55) "
            "elevation. Paired positions are then shot again after rotating the board 90 degrees and "
            "rebuilding the same position on it, which puts a light square on a1."
        ),
        "",
    ]
    for index, row in enumerate(rows):
        if index % 10 == 0:
            out.append(f"## Page {index // 10 + 1}")
            out.append("")
        out.append(f"### {row['id']}  ({row['piece_count']} pieces, {row['density']})")
        out.append(
            f"board: {row['board']} | near side: {row['near_side']} | "
            f"lighting: {row['lighting']} | conventions: {', '.join(row['conventions'])}"
        )
        out.append("")
        out.append("```")
        for rank_index, line in enumerate(diagram(row["fen"])):
            out.append(f"{8 - rank_index}  {line}")
        out.append(f"   {files}")
        out.append("```")
        out.append(f"`{row['fen']}`")
        out.append("")
        if index % 10 == 9:
            out.append('<div style="page-break-after: always"></div>')
            out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--paired", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--from-groups",
        metavar="PATH",
        help="jsonl of benchmark v2 dev groups (id, fen, piece_count, partition=dev); "
        "positions are drawn from these instead of fresh self-play",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATH",
        help="fresh sampling only: file of placements to skip, with their symmetries; repeatable",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.paired < 0:
        raise SystemExit("--paired cannot be negative")
    if args.paired > args.count:
        raise SystemExit("--paired cannot exceed --count")
    if args.from_groups and args.exclude:
        raise SystemExit(
            "--exclude applies to fresh sampling only; dev groups are already "
            "exclusion-checked, run the gate on the output instead"
        )

    out = Path(args.out).expanduser()
    input_paths = [*(args.exclude or [])]
    if args.from_groups:
        input_paths.append(args.from_groups)
    refuse_output_aliases(out, input_paths)

    manifest = {
        "schema": "real-photo-sheet/v2",
        "seed": args.seed,
        "count": args.count,
        "paired": args.paired,
        "photographs": args.count * 3 + args.paired * 3,
        "exclusion_check": "run tools/paper/real_photo_exclusion.py against every pool before shooting",
    }
    sources = []
    if args.from_groups:
        groups_path = Path(args.from_groups).expanduser()
        groups_bytes = groups_path.read_bytes()
        groups_sha256 = hashlib.sha256(groups_bytes).hexdigest()
        groups = load_groups(args.from_groups, groups_bytes)
        sources.append({"path": str(args.from_groups), "sha256": groups_sha256})
        rows, eligible = select_groups(groups, args.count, args.seed)
        source = "benchmark v2 dev groups"
        manifest.update(
            {
                "source": "v2_dev_groups",
                "groups_path": str(args.from_groups),
                "groups_sha256": groups_sha256,
                "groups_total": len(groups),
                "groups_eligible": eligible,
                "selection": "seeded shuffle per density class; sparse at every tenth slot, dense elsewhere",
            }
        )
    else:
        if args.exclude:
            excluded, sources = load_exclusions(args.exclude)
        else:
            excluded = frozenset()
        rows = sample(args.count, args.seed, excluded)
        source = "fresh self-play"
        manifest.update(
            {
                "source": "self_play",
                "sparse_every": SPARSE_EVERY,
                "excluded_inputs": sorted(str(path) for path in args.exclude),
                "excluded_sources": sorted(sources, key=lambda source: source["path"]),
                "excluded_forms": len(excluded),
            }
        )
    rows = assign(rows, args.paired)

    verify_sources_unchanged(sources)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    with (out / "shoot_sheet.jsonl").open("x") as handle:
        handle.write(jsonl)
    with (out / "SHEET.md").open("x") as handle:
        handle.write(render_sheet(rows, args.seed, source) + "\n")
    with (out / "fens.txt").open("x") as handle:
        handle.write("".join(row["fen"] + "\n" for row in rows))
    manifest["sheet_sha256"] = hashlib.sha256(jsonl.encode()).hexdigest()
    manifest["density"] = {
        "sparse": sum(row["density"] == "sparse" for row in rows),
        "dense": sum(row["density"] == "dense" for row in rows),
    }
    manifest["paired_density"] = {
        "sparse": sum(row["density"] == "sparse" for row in rows if row["paired"]),
        "dense": sum(row["density"] == "dense" for row in rows if row["paired"]),
    }
    with (out / "sheet.manifest.json").open("x") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
