#!/usr/bin/env python3
"""Generate legal positions, balanced color pairs, and cameras for colors_b."""

import argparse
import json
import random
from pathlib import Path

import chess


# Uppercase/white pieces are always the lighter group. Values are sRGB inputs
# to the renderer; roughness and metalness make copper/gold recognizable while
# keeping both sides separable. None of the dark warm colors is within the
# detector's 40-level distance threshold around either board square color.
PALETTES = (
    {
        "name": "copper_pale_teal",
        "white": {
            "name": "pale teal",
            "rgb": [188, 229, 218],
            "roughness": 0.38,
            "metalness": 0.08,
        },
        "black": {
            "name": "copper",
            "rgb": [150, 76, 45],
            "roughness": 0.28,
            "metalness": 0.68,
        },
    },
    {
        "name": "red_cream",
        "white": {
            "name": "cream",
            "rgb": [252, 248, 244],
            "roughness": 0.42,
            "metalness": 0.02,
        },
        "black": {
            "name": "deep red",
            "rgb": [132, 32, 45],
            "roughness": 0.40,
            "metalness": 0.04,
        },
    },
    {
        "name": "dark_green_ivory",
        "white": {
            "name": "ivory",
            "rgb": [250, 250, 246],
            "roughness": 0.44,
            "metalness": 0.02,
        },
        "black": {
            "name": "dark green",
            "rgb": [31, 88, 62],
            "roughness": 0.38,
            "metalness": 0.04,
        },
    },
    {
        "name": "light_grey_gold",
        "white": {
            "name": "light grey",
            "rgb": [210, 216, 224],
            "roughness": 0.36,
            "metalness": 0.16,
        },
        "black": {
            "name": "gold",
            "rgb": [150, 103, 24],
            "roughness": 0.25,
            "metalness": 0.72,
        },
    },
)


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


def luminance(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def validate_palettes():
    board_colors = ([215, 206, 190], [170, 148, 126])
    for palette in PALETTES:
        white = palette["white"]["rgb"]
        black = palette["black"]["rgb"]
        if luminance(white) <= luminance(black):
            raise ValueError(f"white group is not lighter in {palette['name']}")
        for side in ("white", "black"):
            rgb = palette[side]["rgb"]
            warm = rgb[0] - rgb[2] > 10
            near_board = any(max(abs(a - b) for a, b in zip(rgb, board)) < 40 for board in board_colors)
            if warm and near_board:
                raise ValueError(f"{palette['name']} {side} enters the board detector mask")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--out", type=Path, default=Path("rendering/colors_b_manifest.jsonl"))
    args = parser.parse_args()
    if args.count <= 0 or args.count % len(PALETTES):
        parser.error(f"--count must be a positive multiple of {len(PALETTES)}")
    validate_palettes()
    rng = random.Random(args.seed ^ 0xC010B)
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
                    "elevation_deg": 25.0 + 45.0 * u,
                    "roll_deg": rng.uniform(-15.0, 15.0),
                    "target_board_width_fraction": rng.uniform(0.45, 0.61),
                },
                "piece_palette": PALETTES[index % len(PALETTES)],
                "piece_color_contract": "uppercase white pieces are the lighter group",
                "piece_geometry": "original procedural Staunton primitives",
                "piece_geometry_license": "original project code; no external asset",
            }
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(positions)} unique balanced colors_b rows to {args.out}")


if __name__ == "__main__":
    main()
