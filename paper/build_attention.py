"""Select a fixed, outcome-independent attention excerpt from frozen sheet pixels."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = (
    "tools/reports/paper/residual_shift_2026-09-10/r1/residual_r1.contact_sheets.json"
)
FULL_SHEETS = Path("tools/reports/paper/attention_full_2026-09-10")
QUERIES = tuple(f"{chr(97 + i)}{8 - i}" for i in range(8))
SHEET_NAME = re.compile(
    r"confirmation__([a-z]+)__parity__(\d{6})\.(source1920x1080|model644)\.png"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_boards(manifest):
    """First two families, then first board ID within each; never inspect outcomes."""
    families = {}
    for relative in manifest["files_sha256"]:
        path = Path(relative)
        match = SHEET_NAME.fullmatch(path.name)
        if path.parent.as_posix() != "residual_r1.contact_sheets" or not match:
            raise ValueError("Unexpected frozen sheet path")
        family, board_id, projection = match.groups()
        if projection == "source1920x1080":
            families.setdefault(family, []).append((board_id, path.name))
    if len(families) < 2:
        raise ValueError("At least two frozen geometry families are required")
    result = []
    for family in sorted(families)[:2]:
        board_id, filename = min(families[family])
        result.append({"family": family, "board_id": board_id, "file": filename})
    return result


def tile_box(query, tile_size):
    """Exact original tile bounds in the canonical a8-to-h1 contact sheet."""
    if query not in QUERIES:
        raise ValueError("Only the preregistered diagonal queries are selected")
    width, height = tile_size
    column, row = ord(query[0]) - ord("a"), 8 - int(query[1])
    return (column * width, row * height, (column + 1) * width, (row + 1) * height)


def verify_sources(root=ROOT):
    evidence = root / "evidence"
    index = json.loads((evidence / "index.json").read_text())["files"]
    manifest_path = evidence / MANIFEST
    receipt_path = manifest_path.with_name("residual_r1.complete.json")
    for path in [manifest_path, receipt_path]:
        key = path.relative_to(evidence).as_posix()
        if digest(path) != index[key]["sha256"]:
            raise ValueError("Attention evidence hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    parent = json.loads(receipt_path.read_text())
    # The export withholds the source digest of a copy it changed.
    bound = (
        "[redacted: withheld digest]"
        if index[MANIFEST]["derived_export_view"]
        else index[MANIFEST]["sha256"]
    )
    if parent["contact_sheet_manifest_sha256"] != bound:
        raise ValueError("Contact sheets are not bound by the original run receipt")
    if parent["attention_boards"] != 30 or parent["attention_queries_per_board"] != 64:
        raise ValueError("Unexpected frozen attention population")
    expected = {Path(p).name: sha for p, sha in manifest["files_sha256"].items()}
    directory = root.parent / FULL_SHEETS
    if len(expected) != 60 or {p.name for p in directory.glob("*.png")} != set(
        expected
    ):
        raise ValueError("Full frozen sheet inventory is incomplete or has extras")
    for name, sha in expected.items():
        path = directory / name
        if path.is_symlink() or digest(path) != sha:
            raise ValueError("Full attention sheet hash mismatch: " + name)
    return manifest, directory


def build_attention(out, root=ROOT):
    import matplotlib.pyplot as plt
    from PIL import Image

    manifest, directory = verify_sources(root)
    selected = select_boards(manifest)
    tile_size = manifest["source_frame_sheet_tile"]
    tiles = []
    fig = plt.figure(figsize=(6.5, 5.5), facecolor="white")
    for board_index, board in enumerate(selected):
        title_y = 0.97 - board_index * 0.5
        fig.text(
            0.025,
            title_y,
            f"{board['family'].capitalize()} / board {board['board_id']}",
            va="top",
            fontsize=11,
            weight="bold",
            color="#182c3c",
        )
        with Image.open(directory / board["file"]) as sheet:
            sheet.load()
            if sheet.size != (8 * tile_size[0], 8 * tile_size[1]):
                raise ValueError("Sheet dimensions do not match frozen tile geometry")
            for query_index, query in enumerate(QUERIES):
                row, column = divmod(query_index, 4)
                box = tile_box(query, tile_size)
                tile = sheet.crop(box).convert("RGB")
                axis = fig.add_axes(
                    (
                        0.025 + column * 0.245,
                        title_y - 0.245 - row * 0.215,
                        0.235,
                        0.175,
                    )
                )
                axis.imshow(tile, interpolation="none", aspect="equal")
                axis.set_title(query, loc="left", fontsize=10, pad=3, color="#182c3c")
                axis.set_axis_off()
                tiles.append(
                    {
                        "board": board["board_id"],
                        "family": board["family"],
                        "query": query,
                        "source_file": board["file"],
                        "box_px": list(box),
                        "rgb_pixels_sha256": hashlib.sha256(tile.tobytes()).hexdigest(),
                    }
                )
    target = out / "attention_excerpt.pdf"
    fig.savefig(
        target,
        metadata={
            "Creator": "Fixed-rule attention excerpt",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    caption = r"""% receipt: records/attention_excerpt.complete.json; original contact-sheet manifest
\begin{figure}[!ht]
  \centering
  \includegraphics[width=\linewidth]{figures/attention_excerpt.pdf}
  \caption{Fixed, outcome-independent excerpt. Select the first two geometry names
  lexicographically in the frozen attention sample, then the lowest zero-padded
  board ID in each: Deplorable 000009 and Lyricsz 000010. For both boards use the
  same eight diagonal queries, a8, b7, c6, d5, e4, f3, g2 and h1. No prediction,
  correctness, confidence or attention magnitude enters this selection. Tiles are
  unchanged crops of the recorded overlays, normalized per query, not a shared
  saliency scale. The full set has 60 sheets (30 boards, 64 queries, two
  projections). Three sheets ship in the package, in \texttt{figures/attention/}.
  The full set is not released.}
  \label{fig:attention-excerpt}
\end{figure}
"""
    (root / "generated/attention_excerpt.tex").write_text(caption)
    receipt = {
        "schema": "paper-attention-excerpt/v1",
        "script_sha256": digest(Path(__file__)),
        "source_manifest": "evidence/" + MANIFEST,
        "source_manifest_sha256": digest(root / "evidence" / MANIFEST),
        "full_sheet_directory": FULL_SHEETS.as_posix(),
        "full_sheets": 60,
        "full_boards": 30,
        "queries_per_full_board": 64,
        "selection_rule": "lexicographically first two geometry names, then lowest zero-padded board ID within each",
        "query_rule": "same a8-to-h1 diagonal; one query per rank and file",
        "queries": list(QUERIES),
        "selected_boards": selected,
        "tiles": tiles,
        "outcomes_consulted": False,
        "model_run": False,
        "source_sha256": manifest["files_sha256"],
        "artifacts_sha256": {
            "figures/attention_excerpt.pdf": digest(target),
            "generated/attention_excerpt.tex": digest(
                root / "generated/attention_excerpt.tex"
            ),
        },
        "published": False,
    }
    path = root / "records/attention_excerpt.complete.json"
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return {
        "receipt": "records/attention_excerpt.complete.json",
        "sha256": digest(path),
        "script_sha256": digest(Path(__file__)),
    }
