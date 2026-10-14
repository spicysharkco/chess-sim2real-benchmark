"""Print-size vector figures from the validated receipt-to-CSV pipeline outputs."""

import csv
import io
import json
import statistics
from pathlib import Path

from build_attention import build_attention
from build_tables import ROOT, digest, load_evidence


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import NullLocator

    data, _ = load_evidence()
    tables = {
        name: list(
            csv.DictReader(
                io.StringIO(data["tools/reports/paper/figures/" + name + ".csv"])
            )
        )
        for name in ["adaptation", "heatmap", "timing", "corner_elevation"]
    }
    router = list(csv.DictReader(io.StringIO(data["reports/historical_pipeline.csv"])))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    meta = {
        "Creator": "Receipt-driven print figure pipeline",
        "CreationDate": None,
        "ModDate": None,
    }
    blue, orange, grey = "#007f80", "#b46d20", "#526775"

    def save(fig, name):
        fig.savefig(out / (name + ".pdf"), metadata=meta)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.15), sharey=True)
    for ax, arm, title in zip(
        axes, ["dark", "both"], ["Dark-only training", "Both-parity training"]
    ):
        records = [r for r in tables["adaptation"] if r["arm"] == arm]
        cost = ax.twinx()
        for p, color, label in [
            ("a1_dark", blue, "a1 dark"),
            ("a1_light", orange, "a1 light"),
        ]:
            rows = [r for r in records if r["parity"] == p]
            budgets = sorted({int(r["budget"]) for r in rows})
            ax.plot(
                budgets,
                [
                    statistics.mean(
                        float(r["exact_pct"]) for r in rows if int(r["budget"]) == n
                    )
                    for n in budgets
                ],
                color=color,
                label=label,
                lw=1.4,
            )
            for r in rows:
                jitter = 1 + (
                    {303: -0.025, 304: 0, 305: 0.025}[int(r["seed"])]
                    if int(r["budget"]) == 2000
                    else 0
                )
                ax.scatter(
                    int(r["budget"]) * jitter,
                    float(r["exact_pct"]),
                    color=color,
                    s=16,
                    zorder=4,
                    edgecolor="white",
                    linewidth=0.3,
                )
        cr = [r for r in records if r["parity"] == "a1_dark"]
        cost.plot(
            budgets,
            [
                statistics.mean(
                    float(r["chessred_loss_pp"]) for r in cr if int(r["budget"]) == n
                )
                for n in budgets
            ],
            color=grey,
            ls="--",
            lw=1,
            label="Photo loss",
        )
        cost.scatter(
            [int(r["budget"]) for r in cr],
            [float(r["chessred_loss_pp"]) for r in cr],
            color=grey,
            marker="x",
            s=16,
        )
        limit = float(cr[0]["chessred_reference_pct"]) - float(
            cr[0]["retention_floor_pct"]
        )
        cost.axhline(limit, ls=":", color="#b34040", lw=1, label="Retention limit")
        ax.set(
            xscale="log",
            xlim=(400, 10000),
            ylim=(-3, 101),
            title=title,
            xlabel="Available position groups",
        )
        ax.set_xticks(budgets, labels=["500", "2,000", "8,000"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", alpha=0.16)
        cost.set_ylim(0, 1.3)
        if arm == "dark":
            ax.set_ylabel("Dev exact-board (%)")
            ax.legend(loc="upper left", fontsize=7, frameon=False)
        if arm == "both":
            cost.set_ylabel("ChessReD loss (pp)")
            cost.legend(loc="lower right", fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.9, bottom=0.23, top=0.9, wspace=0.5)
    fig.text(
        0.5,
        0.04,
        "All seeds shown; lines are means. Fixed updates, not equal unique images seen.",
        ha="center",
        fontsize=7,
    )
    save(fig, "adaptation")

    cells = sorted({r["cell"] for r in tables["heatmap"]})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 5.1), sharey=True)
    for ax, model, title in zip(
        axes, ["release", "adapted_both_8000"], ["Release", "Both-parity, 8,000 groups"]
    ):
        idx = {
            (r["cell"], r["parity"]): float(r["exact_pct"])
            for r in tables["heatmap"]
            if r["model"] == model
        }
        values = np.array([[idx[c, p] for p in ["a1_dark", "a1_light"]] for c in cells])
        im = ax.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_yticks(
            range(len(cells)),
            labels=[
                c.replace("elevation:", "elev ")
                .replace("geometry:", "geom ")
                .replace("palette:", "pal ")
                .replace("framing:", "frame ")
                .replace("_", " ")
                for c in cells
            ],
            fontsize=7,
        )
        ax.set_xticks([0, 1], labels=["a1 dark", "a1 light"])
        ax.set_title(title)
        for i in range(len(cells)):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if values[i, j] < 50 else "#102f3a",
                )
    fig.subplots_adjust(left=0.28, right=0.84, bottom=0.06, top=0.94, wspace=0.27)
    cax = fig.add_axes((0.875, 0.12, 0.025, 0.74))
    fig.colorbar(im, cax=cax, label="Exact-board (%)")
    save(fig, "heatmap")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    delta = [
        float(r["shipped ensemble exact pct"]) - float(r["raw member exact pct"])
        for r in router
    ]
    y = np.arange(len(router))
    ax.barh(y, delta, color=[orange if d < 0 else blue for d in delta], height=0.65)
    ax.set_yticks(y, labels=[r["split"].replace("_", " ") for r in router], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=grey, lw=0.6)
    ax.set(xlabel="Shipped ensemble minus raw member (exact-board pp)", xlim=(-37, 4))
    for i, d in enumerate(delta):
        ax.text(
            d + (-0.5 if d < 0 else 0.4),
            i,
            f"{d:+.1f}",
            va="center",
            ha="right" if d < 0 else "left",
            fontsize=7,
        )
    fig.subplots_adjust(left=0.27, right=0.97, top=0.97, bottom=0.13)
    save(fig, "router_harm")

    names = {
        "cq_release": "ChessQueries release",
        "private_full": "shipped ensemble",
        "private_raw": "raw member",
        "private_warp_high": "warp-high member",
        "private_warp_low": "warp-low member",
    }
    rows = tables["timing"]
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    y = np.arange(len(rows))
    med = [float(r["median_ms"]) for r in rows]
    p90 = [float(r["p90_ms"]) for r in rows]
    ax.barh(y, med, color=blue, height=0.65)
    ax.scatter(p90, y, color=orange, marker="|", s=50, label="P90")
    ax.set_yticks(
        y,
        labels=[
            names.get(
                r["model"], r["model"].replace("lora_", "LoRA ").replace("_", " ")
            )
            for r in rows
        ],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set(xlabel="Single-image latency (ms)", xlim=(0, max(p90) * 1.07))
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.31, right=0.97, top=0.97, bottom=0.14)
    save(fig, "timing")

    rows = list(reversed(tables["corner_elevation"]))
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(rows))
    for key, color, label in [
        ("median_edge_pct", blue, "Median"),
        ("p90_edge_pct", orange, "P90"),
    ]:
        ax.plot(
            x,
            [float(r[key]) for r in rows],
            "-o",
            color=color,
            lw=1.4,
            ms=4,
            label=label,
        )
    ax.set_xticks(
        x,
        labels=[
            r["elevation_low_deg"]
            + "–"
            + r["elevation_high_deg"]
            + "°\nexcluded "
            + str(int(r["n"]) - int(r["usable"]))
            + "/"
            + r["n"]
            for r in rows
        ],
    )
    ax.set_ylabel("Corner error / mean edge (%)")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.94, bottom=0.25)
    save(fig, "corner_elevation")
    attention = build_attention(out)
    record = {
        "schema": "paper-print-figures/v1",
        "attention_excerpt": attention,
        "script_sha256": digest(Path(__file__)),
        "evidence_index_sha256": digest(ROOT / "evidence/index.json"),
        "matplotlib": matplotlib.__version__,
        "output_sha256": {p.name: digest(p) for p in sorted(out.glob("*.pdf"))},
        "width_inches": 6.5,
        "published": False,
    }
    (ROOT / "records/figures.complete.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
