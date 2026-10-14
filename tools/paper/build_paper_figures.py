"""Deterministic paper PDF/CSV artifacts from receipt-bound summary JSON only.

No image, frame manifest, prediction array, checkpoint or GPU is opened. Legacy
summary inputs are explicitly labelled as lacking original run receipts.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

PARITIES = ("a1_dark", "a1_light")
RUNS = [
    f"b{n}_{arm}_s{seed}"
    for n, seeds in [(500, [303]), (2000, [303, 304, 305]), (8000, [303])]
    for seed in seeds
    for arm in ("dark", "both")
]
INPUTS = {
    "lora",
    "timing",
    "corner_v1",
    "router_full",
    "router_raw",
    "router_warp_high",
    "router_warp_low",
}
META = {
    "Creator": "Receipt-driven figure pipeline",
    "Producer": "Vector PDF",
    "CreationDate": None,
    "ModDate": None,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("input path escapes the declared report root")
    return path


def finite(value, *, lower=0, upper=100):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not lower <= value <= upper
    ):
        raise ValueError(f"invalid numeric summary: {value!r}")
    return value


def load_inputs(repo, config):
    """Read only the explicitly pinned report JSON and relevant receipt JSON."""
    repo, config = Path(repo).resolve(), Path(config).resolve()
    specification = json.loads(config.read_text())
    if (
        specification.get("schema") != "paper-figure-inputs/v1"
        or set(specification.get("inputs", {})) != INPUTS
    ):
        raise ValueError("unexpected figure input specification")
    data, provenance = (
        {},
        {"config_sha256": digest(config), "inputs": {}, "receipts": {}},
    )
    for key, item in specification["inputs"].items():
        path = safe_path(repo, item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError(f"input hash mismatch: {key}")
        if item["format"] == "jsonl" and path.suffix == ".jsonl":
            data[key] = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        elif item["format"] == "json" and path.suffix == ".json":
            data[key] = json.loads(path.read_text())
        else:
            raise ValueError("only JSON summary inputs are accepted")
        provenance["inputs"][key] = {
            "sha256": item["sha256"],
            "provenance": item["provenance"],
        }
        if "receipt" in item:
            receipt_path = safe_path(repo, item["receipt"])
            if digest(receipt_path) != item["receipt_sha256"]:
                raise ValueError(f"receipt hash mismatch: {key}")
            receipt = json.loads(receipt_path.read_text())
            expected_status = {
                "lora": "lora_curve_report_complete",
                "timing": "c4_timing_complete",
            }[key]
            if (
                receipt.get("status") != expected_status
                or receipt.get("final_test_images_read") is not False
            ):
                raise ValueError("invalid completion or final-test boundary")
            if receipt["artifacts_sha256"].get(path.name) != digest(path):
                raise ValueError("report is not bound to its completion receipt")
            provenance["receipts"][key] = item["receipt_sha256"]
    # Child timing receipts contain hardware scope; do not reopen timing observations.
    parent = safe_path(repo, specification["inputs"]["timing"]["path"]).parent
    data["timing_receipts"] = {}
    for entry in data["timing"]["runs"]:
        path = parent / "timing.runs" / Path(entry["receipt"]).name
        if (
            not path.name.endswith(".complete.json")
            or digest(path) != entry["receipt_sha256"]
        ):
            raise ValueError("timing child receipt mismatch")
        receipt = json.loads(path.read_text())
        if (
            receipt.get("status") != "c4_timing_complete"
            or receipt.get("final_test_images_read") is not False
            or receipt.get("model_alias") != entry["model_alias"]
            or receipt["summary"] != entry["summary"]
        ):
            raise ValueError("timing summary differs from child receipt")
        data["timing_receipts"][entry["model_alias"]] = receipt
        provenance["receipts"]["timing/" + entry["model_alias"]] = digest(path)
    return data, provenance


def build_tables(data):
    lora = data["lora"]
    if set(lora) != {"release", *RUNS}:
        raise ValueError("expected release and every registered seed/arm/budget")
    base = lora["release"]["metrics"]
    base_public = finite(base["chessred_test"]["exact_pct"])
    cells = sorted(base["dev"]["by_factor_cell"])
    if len(cells) != 20:
        raise ValueError("expected 20 development factor cells")
    expected_joint = {f"{cell}/{parity}" for cell in cells for parity in PARITIES}
    for point in lora.values():
        metrics = point["metrics"]
        if set(metrics) - {"dev", "chessred_test", "paired_vs_release"}:
            raise ValueError("unexpected scored partition")
        dev = metrics["dev"]
        if (
            dev["overall"]["n"] != 16000
            or set(dev["by_parity"]) != set(PARITIES)
            or set(dev["by_factor_cell_parity"]) != expected_joint
        ):
            raise ValueError("development inventory/cell/parity mismatch")
        for joint in dev["by_factor_cell_parity"].values():
            if joint["n"] != 400:
                raise ValueError("expected 400 observations per cell/parity")
            finite(joint["exact_pct"])
            finite(joint["per_square_pct"])
    adaptation = []
    for name in RUNS:
        budget, arm, seed = re.fullmatch(r"b(\d+)_(dark|both)_s(\d+)", name).groups()
        point, metric = lora[name], lora[name]["metrics"]
        public = metric["chessred_test"]
        floor, exact = (
            finite(public["regression_floor_pct"]),
            finite(public["exact_pct"]),
        )
        if public["n"] != 2129 or point["accepted"] is not (exact >= floor):
            raise ValueError("retention flag/count mismatch")
        for parity in PARITIES:
            current = metric["dev"]["by_parity"][parity]
            if current["n"] != 8000:
                raise ValueError("expected 8,000 cell-images per parity")
            adaptation.append(
                {
                    "run": name,
                    "arm": arm,
                    "budget": int(budget),
                    "seed": int(seed),
                    "parity": parity,
                    "exact_pct": finite(current["exact_pct"]),
                    "per_square_pct": finite(current["per_square_pct"]),
                    "reference_exact_pct": base["dev"]["by_parity"][parity][
                        "exact_pct"
                    ],
                    "chessred_exact_pct": exact,
                    "chessred_reference_pct": base_public,
                    "chessred_loss_pp": base_public - exact,
                    "retention_floor_pct": floor,
                    "retention_pass": point["accepted"],
                    "available_images": int(budget) * (1 if arm == "dark" else 2),
                    "image_presentations": 9000,
                }
            )
    heatmap = []
    for model, key in [
        ("release", "release"),
        ("adapted_both_8000", "b8000_both_s303"),
    ]:
        for cell in cells:
            for parity in PARITIES:
                value = lora[key]["metrics"]["dev"]["by_factor_cell_parity"][
                    f"{cell}/{parity}"
                ]
                heatmap.append(
                    {
                        "model": model,
                        "cell": cell,
                        "parity": parity,
                        "n": value["n"],
                        "exact_pct": value["exact_pct"],
                        "per_square_pct": value["per_square_pct"],
                    }
                )
    router = {}
    for variant in ["full", "raw", "warp_high", "warp_low"]:
        values = data["router_" + variant]
        if len(values) != 18 or len({x["split"] for x in values}) != 18:
            raise ValueError("expected all 18 historical splits")
        router[variant] = {r["split"]: r for r in values}
    if any(set(values) != set(router["full"]) for values in router.values()):
        raise ValueError("historical pipeline split inventories differ")
    router_rows = []
    for split in sorted(router["full"]):
        row = {"split": split.removeprefix("bench_"), "n": router["full"][split]["n"]}
        for variant, values in router.items():
            value = values[split]
            if value["n"] != row["n"] or value["n"] not in (200, 300, 500):
                raise ValueError("historical pipeline sample counts differ")
            row[variant + "_exact_pct"] = finite(value["exact"])
            row[variant + "_square_pct"] = finite(value["per_square"])
        row["full_minus_raw_pp"] = row["full_exact_pct"] - row["raw_exact_pct"]
        router_rows.append(row)
    timings = []
    for item in data["timing"]["runs"]:
        value, receipt = item["summary"], data["timing_receipts"][item["model_alias"]]
        if (
            value["images"] != 500
            or receipt["precision"] != "fp16"
            or receipt["batch_size"] != 1
            or receipt["warmup_images"] != 30
        ):
            raise ValueError("timing scope differs from the registered protocol")
        power = float(receipt["hardware"]["before"]["power.limit"])
        if power != 270 or float(receipt["hardware"]["after"]["power.limit"]) != power:
            raise ValueError("timing power limit mismatch")
        timings.append(
            {
                "model": item["model_alias"],
                "images": value["images"],
                "median_ms": finite(value["median_ms"], upper=100000),
                "p90_ms": finite(value["p90_ms"], upper=100000),
                "batch_size": 1,
                "precision": "fp16",
                "power_limit_w": power,
            }
        )
    if (
        len(timings) != 15
        or len({r["model"] for r in timings}) != 15
        or any(r["p90_ms"] < r["median_ms"] for r in timings)
    ):
        raise ValueError("incomplete or invalid timing table")
    corners = []
    for low, high in [(8, 15), (15, 25), (25, 45), (45, 60)]:
        key = f"bench_elev_{low:02d}_{high:02d}"
        value = data["corner_v1"]["splits"][key]
        if value["n"] != 500 or value["elevation_band"] != [low, high]:
            raise ValueError("historical elevation cohort mismatch")
        if (
            value["detector_usable"]
            + value["detector_not_found"]
            + value["detector_divergent_vertex"]
            != value["n"]
        ):
            raise ValueError("conditional corner denominator mismatch")
        corners.append(
            {
                "elevation_low_deg": low,
                "elevation_high_deg": high,
                "n": value["n"],
                "usable": value["detector_usable"],
                "not_found": value["detector_not_found"],
                "divergent": value["detector_divergent_vertex"],
                "median_edge_pct": finite(value["median_rel_pct"]),
                "p90_edge_pct": finite(value["p90_rel_pct"]),
                "median_px": finite(value["median_px"], upper=10000),
                "p90_px": finite(value["p90_px"], upper=10000),
            }
        )
    return {
        "adaptation": adaptation,
        "heatmap": heatmap,
        "router_harm": router_rows,
        "timing": timings,
        "corner_elevation": corners,
    }


def write_figures(tables, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager
    from matplotlib.ticker import NullLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "pdf.compression": 6,
            "svg.hashsalt": "fixed-paper-v1",
        }
    )

    def save(fig, name):
        fig.savefig(out / (name + ".pdf"), format="pdf", metadata=META)
        plt.close(fig)

    blue, orange, gray = "#2563eb", "#d97706", "#64748b"

    def point_x(row):
        jitter = (
            {303: -0.025, 304: 0, 305: 0.025}[row["seed"]]
            if row["budget"] == 2000
            else 0
        )
        return row["budget"] * (1 + jitter)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=True)
    for ax, arm, title in zip(
        axes, ["dark", "both"], ["Dark-only training", "Both-parity training"]
    ):
        cost = ax.twinx()
        records = [r for r in tables["adaptation"] if r["arm"] == arm]
        for parity, colour, label in zip(
            PARITIES, [blue, orange], ["Dark convention", "Light convention"]
        ):
            rows = [r for r in records if r["parity"] == parity]
            means = [
                statistics.mean(r["exact_pct"] for r in rows if r["budget"] == n)
                for n in [500, 2000, 8000]
            ]
            ax.plot([500, 2000, 8000], means, color=colour, lw=1.7, label=label)
            for r in rows:
                x = point_x(r)
                ax.scatter(
                    x,
                    r["exact_pct"],
                    color=colour,
                    s=24,
                    zorder=4,
                    edgecolor="white",
                    linewidth=0.4,
                )
        unique = [r for r in records if r["parity"] == "a1_dark"]
        means = [
            statistics.mean(r["chessred_loss_pp"] for r in unique if r["budget"] == n)
            for n in [500, 2000, 8000]
        ]
        cost.plot(
            [500, 2000, 8000], means, color=gray, lw=1.4, ls="--", label="ChessReD loss"
        )
        for r in unique:
            cost.scatter(
                point_x(r),
                r["chessred_loss_pp"],
                marker="x",
                color=gray,
                s=25,
            )
        allowed = unique[0]["chessred_reference_pct"] - unique[0]["retention_floor_pct"]
        cost.axhline(allowed, color="#b91c1c", ls=":", lw=1, label="Retention limit")
        ax.set(
            xscale="log",
            xlim=(420, 9600),
            ylim=(-3, 100),
            title=title,
            xlabel="Available position groups",
        )
        ax.set_xticks([500, 2000, 8000], labels=["500", "2,000", "8,000"])
        ax.xaxis.set_minor_locator(NullLocator())
        cost.set_ylim(0, 1.3)
        cost.spines["right"].set_visible(True)
        ax.grid(axis="y", alpha=0.18)
        ax.set_ylabel("Dev exact-board accuracy (%)")
        cost.set_ylabel("ChessReD exact loss (pp)", color=gray)
        if arm == "dark":
            ax.legend(loc="upper left", frameon=False, fontsize=8)
        else:
            cost.legend(loc="lower right", frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.25, top=0.86, wspace=0.50)
    fig.suptitle(
        "Fixed-step adaptation: recognition and retention", y=0.98, fontsize=13
    )
    fig.text(
        0.5,
        0.10,
        "Points: all seeds (2k x-jitter for visibility); lines: means. Same 1,500 updates / 9,000 presentations; available views: N vs 2N.",
        ha="center",
        fontsize=8,
    )
    base_dark, base_light = [
        next(r["reference_exact_pct"] for r in tables["adaptation"] if r["parity"] == p)
        for p in PARITIES
    ]
    failed = sum(
        not r["retention_pass"]
        for r in tables["adaptation"]
        if r["parity"] == "a1_dark"
    )
    floor = tables["adaptation"][0]["retention_floor_pct"]
    fig.text(
        0.5,
        0.055,
        f"Release: {base_dark:.3f}% dark / {base_light:g}% light exact. {failed}/10 adapted points fail the {floor:g}% ChessReD retention floor.",
        ha="center",
        fontsize=8,
    )
    save(fig, "adaptation")

    cells = sorted({r["cell"] for r in tables["heatmap"]})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 8.2), sharey=True)
    for ax, model, title in zip(
        axes, ["release", "adapted_both_8000"], ["Release", "Both-parity, 8,000 groups"]
    ):
        idx = {
            (r["cell"], r["parity"]): r["exact_pct"]
            for r in tables["heatmap"]
            if r["model"] == model
        }
        values = np.array([[idx[(c, p)] for p in PARITIES] for c in cells])
        im = ax.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_xticks([0, 1], labels=["Dark", "Light"])
        ax.set_yticks(
            range(len(cells)),
            labels=[c.replace(":", ": ").replace("_", " ") for c in cells],
        )
        ax.set_title(title)
        for i in range(20):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if values[i, j] > 58 else "white",
                )
    fig.subplots_adjust(left=0.29, right=0.86, bottom=0.10, top=0.91, wspace=0.35)
    cax = fig.add_axes([0.90, 0.19, 0.023, 0.63])
    fig.colorbar(im, cax=cax, label="Exact-board accuracy (%)")
    fig.text(
        0.5,
        0.035,
        "400 position groups shared across cells; 400 observations per cell/parity. Development only.",
        ha="center",
        fontsize=8,
    )
    save(fig, "heatmap")

    def table_pdf(name, title, headers, values, note, widths, size):
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")
        ax.set_title(title, pad=15, fontweight="bold")
        table = ax.table(
            cellText=values,
            colLabels=headers,
            loc="center",
            cellLoc="right",
            colWidths=widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.3)
        table.scale(1, 1.48)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#e2e8f0")
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f8fafc")
            if col == 0:
                cell.set_text_props(ha="left")
        fig.subplots_adjust(left=0.035, right=0.965, top=0.87, bottom=0.16)
        fig.text(0.04, 0.05, note, fontsize=8, va="bottom")
        save(fig, name)

    rows = tables["router_harm"]
    values = [
        [
            r["split"].replace("_", " "),
            str(r["n"]),
            *[
                f"{r[v + '_exact_pct']:.2f}"
                for v in ["full", "raw", "warp_high", "warp_low"]
            ],
            f"{r['full_minus_raw_pp']:+.2f}",
        ]
        for r in rows
    ]
    table_pdf(
        "router_harm",
        "Historical v1: routed pipeline versus members",
        ["Split", "n", "Full %", "Raw %", "High %", "Low %", "Full - raw pp"],
        values,
        "All 18 historical summaries; negative deltas favour raw. This changes both routing and member composition.\nOriginal run receipts unavailable; source summaries are hash-bound. Invalid historical latency is omitted.",
        [0.31, 0.055, 0.115, 0.115, 0.115, 0.115, 0.175],
        (10.6, 7.3),
    )
    values = [
        [r["model"], f"{r['median_ms']:.2f}", f"{r['p90_ms']:.2f}"]
        for r in tables["timing"]
    ]
    table_pdf(
        "timing",
        "Isolated end-to-end pipeline latency",
        ["System", "Median ms/image", "p90 ms/image"],
        values,
        "RTX 3090, 270 W; batch 1 FP16; 500 identical inputs; 30 excluded warm-up images. File decode excluded.\nNative preprocessing/routing/TTA/postprocessing included. Different pipelines are not equal-FLOP comparisons.",
        [0.54, 0.23, 0.23],
        (9.2, 6.7),
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    rows = tables["corner_elevation"]
    x = list(range(4))
    ax.plot(
        x, [r["median_edge_pct"] for r in rows], "o-", color=blue, label="Median", lw=2
    )
    ax.plot(
        x,
        [r["p90_edge_pct"] for r in rows],
        "s--",
        color=orange,
        label="90th percentile",
        lw=1.5,
    )
    ax.set_xticks(
        x,
        labels=[
            f"{r['elevation_low_deg']}-{r['elevation_high_deg']}\nusable {r['usable']}/{r['n']}"
            for r in rows
        ],
    )
    ax.set(
        xlabel="Elevation band (degrees)",
        ylabel="Corner error (% of mean projected edge)",
        title="Historical v1 localization by elevation",
        ylim=(0, 5),
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(left=0.105, right=0.97, top=0.86, bottom=0.30)
    fig.text(
        0.105,
        0.06,
        "Conditional on usable, non-divergent detections; failures are counted in CSV, not assigned zero error.\nHistorical summary, not a newly validated or re-run detector experiment.",
        fontsize=8,
    )
    save(fig, "corner_elevation")
    return {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "font_family": "DejaVu Sans",
        "font_sha256": digest(font_manager.findfont("DejaVu Sans")),
        "pdf_metadata": "timestamp-free",
    }


def build(repo, config, out):
    data, provenance = load_inputs(repo, config)
    tables = build_tables(data)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "INCOMPLETE").write_text("Not a completed figure build.\n")
    for name, rows in tables.items():
        with (out / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    environment = write_figures(tables, out)
    if load_inputs(repo, config)[1] != provenance:
        raise ValueError("report inputs changed during figure build")
    receipt = {
        "schema": "paper-figures/v1",
        "status": "paper_figures_complete",
        "release_ready": False,
        "gpu_used": False,
        "final_test_images_read": False,
        "provenance": provenance,
        "renderer": environment,
        "script_sha256": digest(__file__),
        "table_rows": {name: len(rows) for name, rows in tables.items()},
        "legacy_summary_warning": "v1 routing and corners have no original run receipt; current bytes are pinned, not retrospectively certified",
        "artifacts_sha256": {
            p.name: digest(p)
            for p in sorted(out.iterdir())
            if p.suffix in (".csv", ".pdf")
        },
    }
    (out / "figures.complete.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    (out / "INCOMPLETE").unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--input-map", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = args.input_map or args.repo / "tools/paper/figure_inputs_2026-09-10.json"
    receipt = build(args.repo, config, args.out)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "outputs": len(receipt["artifacts_sha256"]),
                "receipt_sha256": digest(args.out / "figures.complete.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
