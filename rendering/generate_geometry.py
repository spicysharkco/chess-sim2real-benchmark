#!/usr/bin/env python3
"""Generate deterministic legal positions and cameras for geometry renders."""

import argparse
import json
import random
from pathlib import Path

import chess


def play_position(rng, sparse):
    board = chess.Board()
    stop_at = rng.randint(65, 125) if sparse else rng.randint(12, 46)
    for ply in range(stop_at):
        moves = list(board.legal_moves)
        if not moves:
            break
        captures = [move for move in moves if board.is_capture(move)]
        probability = 0.58 if sparse else 0.28
        pool = captures if captures and ply > 12 and rng.random() < probability else moves
        board.push(rng.choice(pool))
    return board


def sampled_positions(count, seed):
    rng = random.Random(seed)
    seen = set()
    rows = []
    game_id = 0
    while len(rows) < count:
        sparse = len(rows) % 10 == 9
        board = play_position(rng, sparse)
        game_id += 1
        fen = board.board_fen()
        pieces = len(board.piece_map())
        wanted = 6 <= pieces <= 15 if sparse else 20 <= pieces <= 34
        if not wanted or fen in seen:
            continue
        seen.add(fen)
        rows.append((fen, pieces, game_id, len(board.move_stack)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--elevation-min", type=float, default=25.0)
    parser.add_argument("--elevation-max", type=float, default=70.0)
    parser.add_argument(
        "--out", type=Path, default=Path("rendering/geometry_manifest.jsonl")
    )
    args = parser.parse_args()
    if args.count <= 0 or not 0 < args.elevation_min < args.elevation_max < 90:
        parser.error("count must be positive and elevations must satisfy 0 < min < max < 90")
    rng = random.Random(args.seed ^ 0x6E0E7)
    positions = sampled_positions(args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for index, (fen, pieces, game_id, ply) in enumerate(positions):
            bin_width = 360.0 / args.count
            azimuth = (
                (index + 0.5) * bin_width + rng.uniform(-0.42, 0.42) * bin_width
            ) % 360.0
            u = ((index * 307) % args.count + 0.5) / args.count
            row = {
                "id": f"{index:06d}",
                "fen": fen,
                "piece_count": pieces,
                "game_id": game_id,
                "ply": ply,
                "seed": args.seed,
                "camera": {
                    "azimuth_deg": azimuth,
                    "elevation_deg": args.elevation_min + (args.elevation_max - args.elevation_min) * u,
                    "roll_deg": rng.uniform(-15.0, 15.0),
                    "target_board_width_fraction": rng.uniform(0.45, 0.61),
                },
                "piece_geometry": {
                    "source": "Poly Haven chess_set 1k glTF",
                    "asset_url": "https://polyhaven.com/a/chess_set",
                    "license": "CC0",
                },
            }
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(positions)} unique legal geometry positions to {args.out}")


if __name__ == "__main__":
    main()
