"""Deterministic manifest design primitives for benchmark v2.

This module deliberately does not inspect assets, historical FEN pools, or the
filesystem.  Callers provide pinned geometry/palette descriptors and a
membership object containing the already-excluded D4 keys.
"""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Collection, Mapping, Sequence

try:
    from .generate_colors_b import PALETTES
    from .generate_geometry import play_position
except ImportError:
    from generate_colors_b import PALETTES
    from generate_geometry import play_position

try:
    from tools.paper.fen_pools import placement, sym_key
except ImportError:  # Executed with rendering/ as the import root.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.paper.fen_pools import placement, sym_key


BENCHMARK_SCHEMA = 2
BENCHMARK_SEED = 20260910
ADAPTATION_GENERATION_SEED = 20260911
ADAPTATION_SCHEDULE_SEED = 20260911
ADAPTATION_BUDGET_SEED = 20260912
ADAPTATION_BUDGETS = (500, 2_000, 8_000)

ELEVATION_BANDS = (
    ("08_15", 8.0, 15.0),
    ("15_25", 15.0, 25.0),
    ("25_45", 25.0, 45.0),
    ("45_60", 45.0, 60.0),
)
GEOMETRIES = (
    "polyhaven",
    "deplorable",
    "killgorack",
    "lyricsz",
    "omie",
    "phuong",
    "pjhanzlik",
)
FRAMING_BANDS = (
    ("compact", 0.30, 0.40),
    ("standard", 0.52, 0.61),
)
PARITIES = ("a1_dark", "a1_light")
REFERENCE_ELEVATION = "45_60"
REFERENCE_GEOMETRY = "polyhaven"
REFERENCE_FRAMING = "standard"
REFERENCE_BOARD_PALETTE = {
    "name": "default_board",
    "light_rgb": [215, 206, 190],
    "dark_rgb": [170, 148, 126],
}
REFERENCE_PIECE_PALETTE = PALETTES[0]


def canonical_position_group(fen: str) -> str:
    """Return the stable group ID for a canonical piece-placement FEN."""

    return hashlib.sha256(placement(fen).encode()).hexdigest()


def generate_groups(
    count: int,
    seed: int,
    excluded_keys: Collection[str],
) -> list[dict]:
    """Generate a prefix-stable sequence of legal, exact/D4-disjoint positions.

    ``excluded_keys`` is intentionally used only through membership checks so a
    caller can pass a large set, database-backed view, or other compact index
    without this function copying millions of keys.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = random.Random(seed)
    generated_keys: set[str] = set()
    groups: list[dict] = []
    game_id = 0
    while len(groups) < count:
        index = len(groups)
        sparse = index % 10 == 9
        board = play_position(rng, sparse)
        game_id += 1
        fen = placement(board.board_fen())
        pieces = len(board.piece_map())
        wanted = 6 <= pieces <= 15 if sparse else 20 <= pieces <= 34
        key = sym_key(fen)
        if not wanted or key in excluded_keys or key in generated_keys:
            continue
        generated_keys.add(key)
        groups.append(
            {
                "id": f"{index:06d}",
                "fen": fen,
                "piece_count": pieces,
                "game_id": game_id,
                "ply": len(board.move_stack),
                "seed": seed,
                "position_group": canonical_position_group(fen),
                "azimuth_deg": rng.uniform(0.0, 360.0),
                "roll_deg": rng.uniform(-5.0, 5.0),
                "elevation_u": rng.random(),
                "framing_u": rng.random(),
            }
        )
    return groups


def partition_benchmark_groups(
    groups: Sequence[Mapping],
    *,
    dev_count: int = 400,
    seed: int = BENCHMARK_SEED,
) -> list[dict]:
    """Assign whole position groups to seeded dev/final-test partitions."""

    if not 0 < dev_count < len(groups):
        raise ValueError("dev_count must leave non-empty dev and final_test sets")
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    dev = set(order[:dev_count])
    return [
        dict(group, partition="dev" if i in dev else "final_test")
        for i, group in enumerate(groups)
    ]


def _validate_geometries(geometries: Sequence[str]) -> tuple[str, ...]:
    names = tuple(geometries)
    if not names or len(names) != len(set(names)):
        raise ValueError("geometry names must be non-empty and unique")
    if not set(GEOMETRIES).issubset(names):
        raise ValueError("geometry names must retain all seven primary families")
    if not set(names) - set(GEOMETRIES) <= {"nailfighter"}:
        raise ValueError("Nailfighter is the only optional geometry family")
    return names


def factor_cells(
    palette_schemes: Sequence[Mapping],
    *,
    geometries: Sequence[str] = GEOMETRIES,
) -> tuple[tuple[str, str], ...]:
    """Return the fixed OFAT cells (19 primary, 20 with Nailfighter)."""

    geometry_names = _validate_geometries(geometries)
    if len(palette_schemes) != 6:
        raise ValueError("benchmark v2 requires exactly six palette schemes")
    palette_names = [scheme.get("name") for scheme in palette_schemes]
    if any(not isinstance(name, str) or not name for name in palette_names):
        raise ValueError("every palette scheme needs a non-empty name")
    if len(set(palette_names)) != 6:
        raise ValueError("palette scheme names must be unique")
    return tuple(
        [("elevation", name) for name, _, _ in ELEVATION_BANDS]
        + [("geometry", name) for name in geometry_names]
        + [("palette", name) for name in palette_names]
        + [("framing", name) for name, _, _ in FRAMING_BANDS]
    )


def _balanced_level_schedule(
    count: int,
    levels: Sequence[str],
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    result: list[str] = []
    while len(result) < count:
        cycle = list(levels)
        rng.shuffle(cycle)
        result.extend(cycle[: count - len(result)])
    return result


def adaptation_factor_schedule(
    count: int,
    palette_schemes: Sequence[Mapping],
    *,
    seed: int = ADAPTATION_SCHEDULE_SEED,
    geometries: Sequence[str] = GEOMETRIES,
) -> list[dict[str, str]]:
    """Return independently balanced marginal factor schedules.

    Unlike the benchmark's 19 one-factor-at-a-time cells, adaptation examples
    combine factor levels.  Each marginal is a shuffled-cycle schedule, so at
    every prefix its level counts differ by at most one per completed cycle.
    """

    if count < 0:
        raise ValueError("count cannot be negative")
    geometry_names = _validate_geometries(geometries)
    factor_cells(palette_schemes, geometries=geometry_names)
    levels = {
        "elevation_band": [name for name, _, _ in ELEVATION_BANDS],
        "geometry": list(geometry_names),
        "palette": ["reference", *[scheme["name"] for scheme in palette_schemes]],
        "framing": [name for name, _, _ in FRAMING_BANDS],
    }
    schedules = {
        name: _balanced_level_schedule(count, values, seed ^ salt)
        for salt, (name, values) in enumerate(levels.items(), start=1)
    }
    return [{name: schedules[name][index] for name in levels} for index in range(count)]


def adaptation_budget_groups(
    groups: Sequence[Mapping],
    budget: int,
    *,
    seed: int = ADAPTATION_BUDGET_SEED,
) -> list[Mapping]:
    """Select a nested prefix of one seeded permutation of 8,000 groups."""

    if budget not in ADAPTATION_BUDGETS:
        raise ValueError(f"budget must be one of {ADAPTATION_BUDGETS}")
    if len(groups) != ADAPTATION_BUDGETS[-1]:
        raise ValueError("budget selection requires the complete 8,000-group pool")
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    return [groups[index] for index in order[:budget]]


def _validate_groups(groups: Sequence[Mapping]) -> None:
    ids = [group.get("id") for group in groups]
    if any(not isinstance(group_id, str) or not group_id for group_id in ids):
        raise ValueError("every position group needs a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("position group ids must be unique")
    keys = [sym_key(group["fen"]) for group in groups]
    if len(keys) != len(set(keys)):
        raise ValueError("position groups must be exact/D4 unique")
    for group in groups:
        expected_group = canonical_position_group(group["fen"])
        if group.get("position_group") != expected_group:
            raise ValueError("position_group must hash the canonical placement FEN")


def _lookup_named(items: Sequence[Mapping], name: str, kind: str) -> Mapping:
    matches = [item for item in items if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {kind} named {name!r}")
    return matches[0]


def _factor_values(axis: str, level: str, palette_names: Sequence[str]) -> dict:
    values = {
        "elevation_band": REFERENCE_ELEVATION,
        "geometry": REFERENCE_GEOMETRY,
        "palette": "reference",
        "framing": REFERENCE_FRAMING,
    }
    key = {
        "elevation": "elevation_band",
        "geometry": "geometry",
        "palette": "palette",
        "framing": "framing",
    }[axis]
    if axis == "palette" and level not in palette_names:
        raise ValueError(f"unknown palette level: {level}")
    values[key] = level
    return values


def _row(
    group: Mapping,
    *,
    row_index: int,
    group_index: str,
    axis: str,
    level: str,
    parity: str,
    geometry_descriptors: Mapping[str, Mapping],
    palette_schemes: Sequence[Mapping],
    adaptation: bool,
    factor_values: Mapping[str, str] | None = None,
) -> dict:
    palette_names = [scheme["name"] for scheme in palette_schemes]
    factors = (
        dict(factor_values)
        if factor_values is not None
        else _factor_values(axis, level, palette_names)
    )
    factors["parity"] = parity
    elevation = {name: (low, high) for name, low, high in ELEVATION_BANDS}
    framing = {name: (low, high) for name, low, high in FRAMING_BANDS}
    low_e, high_e = elevation[factors["elevation_band"]]
    low_f, high_f = framing[factors["framing"]]
    if factors["palette"] == "reference":
        board_palette = REFERENCE_BOARD_PALETTE
        piece_palette = REFERENCE_PIECE_PALETTE
    else:
        scheme = _lookup_named(palette_schemes, factors["palette"], "palette")
        board_palette = scheme["board"]
        piece_palette = scheme["pieces"]
    partition = "adaptation_train" if adaptation else group.get("partition")
    if partition not in ({"adaptation_train"} if adaptation else {"dev", "final_test"}):
        raise ValueError(
            "groups must carry the correct partition before views are designed"
        )
    data_usage = {
        "dev": "benchmark_dev",
        "final_test": "benchmark_final_test",
        "adaptation_train": "adaptation_train",
    }[partition]
    cell = (
        ":".join(
            factors[name]
            for name in ("elevation_band", "geometry", "palette", "framing")
        )
        if adaptation
        else f"{axis}:{level}"
    )
    view_key = f"{partition}-{group_index}-{axis}-{level}-{parity}"
    return {
        "id": f"{row_index:06d}",
        "view_key": view_key,
        "benchmark_schema": BENCHMARK_SCHEMA,
        "experiment": "benchmark_v2",
        "fen": placement(group["fen"]),
        "piece_count": group["piece_count"],
        "game_id": group["game_id"],
        "ply": group["ply"],
        "seed": group["seed"],
        "position_group": group["position_group"],
        "partition": partition,
        "data_usage": data_usage,
        "factor_axis": axis,
        "factor_level": level,
        "factor_cell": cell,
        "factors": factors,
        "parity": parity,
        "camera": {
            "azimuth_deg": group["azimuth_deg"],
            "elevation_deg": low_e + (high_e - low_e) * group["elevation_u"],
            "roll_deg": group["roll_deg"],
            "target_board_width_fraction": low_f
            + (high_f - low_f) * group["framing_u"],
        },
        "piece_geometry": copy.deepcopy(geometry_descriptors[factors["geometry"]]),
        "board_palette": copy.deepcopy(board_palette),
        "piece_palette": copy.deepcopy(piece_palette),
        "piece_color_contract": "uppercase white is the lighter side",
        "exclusion_policy": "canonical_placement_exact_and_d4",
    }


def design_rows(
    groups: Sequence[Mapping],
    geometry_descriptors: Mapping[str, Mapping],
    palette_schemes: Sequence[Mapping],
    *,
    adaptation: bool = False,
    seed: int = ADAPTATION_SCHEDULE_SEED,
    geometries: Sequence[str] = GEOMETRIES,
) -> list[dict]:
    """Expand position groups into paired benchmark or adaptation rows."""

    geometry_names = _validate_geometries(geometries)
    missing = set(geometry_names) - set(geometry_descriptors)
    extra = set(geometry_descriptors) - set(geometry_names)
    if missing or extra:
        raise ValueError(
            f"geometry descriptor mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    _validate_groups(groups)
    cells = factor_cells(palette_schemes, geometries=geometry_names)
    assigned = (
        adaptation_factor_schedule(
            len(groups), palette_schemes, seed=seed, geometries=geometry_names
        )
        if adaptation
        else None
    )
    rows = []
    for index, group in enumerate(groups):
        selected = (
            (("joint", "sampled", assigned[index]),)
            if assigned is not None
            else ((axis, level, None) for axis, level in cells)
        )
        for axis, level, factor_values in selected:
            for parity in PARITIES:
                rows.append(
                    _row(
                        group,
                        row_index=len(rows),
                        group_index=group["id"],
                        axis=axis,
                        level=level,
                        parity=parity,
                        geometry_descriptors=geometry_descriptors,
                        palette_schemes=palette_schemes,
                        adaptation=adaptation,
                        factor_values=factor_values,
                    )
                )
    return rows
