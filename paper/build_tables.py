"""Generate paper tables from hash-bound, scrubbed receipt views (CPU only)."""

import hashlib
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "tools/reports/paper/"
LORA = BASE + "lora_curve_2026-09-10/lora_report.json"
TIMING = BASE + "timing_2026-09-10/timing.runs.json"
CONVENTIONS = "docs/records/PARITY_CONVENTIONS.md"
REDACTED = "[redacted: private artefact]"
WITHHELD = "[redacted: withheld digest]"
ASSETS = "docs/records/BENCHMARK_ASSET_INVENTORY_2026-09-10.json"
SUMMARY = "benchmark_v2/benchmark.summary.json"
PHUONG_STATUS = "docs/records/PHUONG_RELEASE_STATUS_2026-10-12.complete.json"
SECTION4_PREFIXES = (
    "data/holdout/leak_check",
    "tools/reports/paper/cq_baseline_",
    "tools/reports/paper/crop_diag_",
    "tools/reports/paper/member_",
    "tools/reports/paper/parity_probe_2026-09-09/analysis.json",
    "tools/reports/paper/real_photo_2026-09-11/a1_light_2026-10-04/",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence(root=ROOT / "evidence"):
    index = json.loads((root / "index.json").read_text())
    data = {}
    for name, record in index["files"].items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root.resolve())
            or digest(path) != record["sha256"]
        ):
            raise ValueError("Evidence hash or path mismatch: " + name)
        data[name] = (
            json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()
        )
    for name, receipt in data.items():
        if not name.endswith(".complete.json") or not isinstance(receipt, dict):
            continue
        for artifact, expected in receipt.get("artifacts_sha256", {}).items():
            relative = str(Path(name).parent / artifact)
            if relative not in index["files"]:
                continue
            record = index["files"][relative]
            prior = index["files"].get(record.get("prior_version"), record)
            # The export withholds the source digest of a copy it changed, so a
            # receipt binds such a copy by the marker and an unchanged one by hash.
            if record["derived_export_view"]:
                bound = expected == WITHHELD
            else:
                bound = expected == record["sha256"] or (
                    not prior["derived_export_view"] and expected == prior["sha256"]
                )
            if not bound:
                raise ValueError("Historical receipt binding mismatch: " + relative)
    return data, index


def percent_text(value):
    """Display stored percentages with decimal half-up rounding; never gate on display."""
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def tex(value):
    return "".join(
        {
            "_": r"\_\allowbreak{}",
            "%": r"\%",
            "&": r"\&",
            "#": r"\#",
            "{": r"\{",
            "}": r"\}",
        }.get(c, c)
        for c in str(value)
    )


def row(values, source, key):
    # Every cell is on its own source-annotated line, including numeric identifiers.
    decision_note = (
        f"; release qualification: evidence/{PHUONG_STATUS}"
        if source == ASSETS and key == "families/phuong"
        else ""
    )
    receipt = {
        LORA: BASE + "lora_curve_2026-09-10/lora_report.complete.json",
        TIMING: BASE + "timing_2026-09-10/timing.complete.json",
        CONVENTIONS: "docs/records/PARITY_CONVENTIONS.complete.json",
        ASSETS: "docs/records/RELEASE_LICENCE_REVIEW.complete.json",
        SUMMARY: "benchmark_v2/benchmark.complete.json",
        "docs/records/RELEASE_LICENCE_REVIEW_2026-09-10.md": "docs/records/RELEASE_LICENCE_REVIEW.complete.json",
    }.get(source, source)
    return (
        "\n".join(
            tex(value)
            + (" &" if i + 1 < len(values) else r" \\")
            + f" % receipt: evidence/{receipt}; source: evidence/{source}; {key}{decision_note}"
            for i, value in enumerate(values)
        )
        + "\n"
    )


def table(headers, rows, source, label, caption, widths=None, long=False):
    columns = widths or ("l" + "r" * (len(headers) - 1))
    columns = columns.replace("p{", r">{\raggedright\arraybackslash}p{")
    start = [r"\begingroup\small", r"\setlength{\tabcolsep}{4pt}"]
    if long:
        start += [
            r"\begin{longtable}{" + columns + "}",
            r"\caption{" + caption + r"}\label{" + label + r"}\\",
            r"\toprule",
        ]
    else:
        start += [
            r"\begin{table}[htbp]\centering",
            r"\caption{" + caption + r"}\label{" + label + "}",
            r"\begin{tabular}{" + columns + "}",
            r"\toprule",
        ]
    head = row(headers, source, "column definitions") + "\\midrule\n"
    start += [head]
    if long:
        start += [
            r"\endfirsthead",
            r"\toprule",
            head,
            r"\endhead",
            r"\bottomrule\endfoot",
        ]
    start.extend(row(values, source, key) for values, key in rows)
    start += [
        r"\end{longtable}" if long else "\\bottomrule\n\\end{tabular}\n\\end{table}",
        r"\endgroup",
    ]
    return "\n".join(start) + "\n"


def licensing_status(data):
    decision = data[PHUONG_STATUS]
    if (
        decision.get("scope") != "benchmark_data_asset_licensing"
        or decision.get("release_ready") is not True
        or decision.get("family_render_release_ready") is not True
        or decision.get("upstream_version_recovered") is not False
        or decision.get("gated_families") != []
        or decision.get("package_attribution_terms") != "CC BY 4.0"
        or hashlib.sha256(data[decision["decision_path"]].encode()).hexdigest()
        != decision["decision_sha256"]
        or hashlib.sha256(data[decision["attribution_path"]].encode()).hexdigest()
        != decision["attribution_sha256"]
    ):
        raise ValueError("Unbound or unsupported Phuong author decision")
    return decision


def build(data):
    licensing_status(data)
    out = {}
    summary = data[SUMMARY]
    cells = summary["cells"]
    factors = {
        axis: sorted(
            {
                key.split("/")[0].split(":")[1]
                for key in cells
                if key.startswith(axis + ":")
            }
        )
        for axis in ["elevation", "geometry", "palette", "framing"]
    }
    rows = [
        ([axis.capitalize(), len(levels), ", ".join(levels)], "cells/" + axis)
        for axis, levels in factors.items()
    ]
    rows += [
        (
            [
                "Convention",
                len({key.split("/")[1] for key in cells}),
                "a1 dark, a1 light",
            ],
            "cells/parity",
        )
    ]
    out["design.tex"] = table(
        ["Factor", "Levels", "Values"],
        rows,
        SUMMARY,
        "tab:design",
        "Paired factor panels, not a full factorial cross-product.",
        "p{.16\\linewidth}rp{.65\\linewidth}",
    )
    runs = data[LORA]
    if len(runs) != 11:
        raise ValueError("Expected release and every registered adaptation run")
    rows = []
    for name, value in runs.items():
        m = value["metrics"]
        p = m["dev"]["by_parity"]
        cr = m["chessred_test"]
        if m["dev"]["overall"]["n"] != 16000 or cr["n"] != 2129:
            raise ValueError("Evaluation inventory mismatch")
        rows.append(
            (
                [
                    name,
                    percent_text(p["a1_dark"]["exact_pct"]),
                    percent_text(p["a1_light"]["exact_pct"]),
                    percent_text(cr["exact_pct"]),
                    cr["exact_boards"],
                    "reference"
                    if name == "release"
                    else ("pass" if value["accepted"] else "fail"),
                ],
                name + "/metrics",
            )
        )
    out["adaptation.tex"] = table(
        ["Run", "Dark (%)", "Light (%)", "Photo (%)", "Photo exact", "Gate"],
        rows,
        LORA,
        "tab:adaptation",
        "All fixed-step runs. Synthetic cells are paired; photo counts use the same retention set.",
    )
    rows = []
    for name, value in runs.items():
        joint = value["metrics"]["dev"]["by_factor_cell_parity"]
        if len(joint) != 40:
            raise ValueError("Expected every development factor cell and parity")
        for cell, metric in sorted(joint.items()):
            rows.append(
                (
                    [
                        name,
                        cell.replace("a1_", ""),
                        metric["n"],
                        percent_text(metric["exact_pct"]),
                        percent_text(metric["per_square_pct"]),
                    ],
                    name + "/metrics/dev/by_factor_cell_parity/" + cell,
                )
            )
    out["per_cell.tex"] = table(
        ["Run", "Cell / convention", "N", "Exact (%)", "Square (%)"],
        rows,
        LORA,
        "tab:percell",
        "Complete development cell results for the release and each adaptation run.",
        "lp{.42\\linewidth}rrr",
        long=True,
    )
    names = {
        "cq_release": "ChessQueries release",
        "private_full": "shipped ensemble",
        "private_raw": "raw member",
        "private_warp_high": "warp-high member",
        "private_warp_low": "warp-low member",
    }
    rows = []
    for record in data[TIMING]["runs"]:
        m = record["summary"]
        alias = record["model_alias"]
        rows.append(
            (
                [
                    names.get(alias, alias.replace("lora_", "LoRA ").replace("_", " ")),
                    m["images"],
                    f"{m['median_ms']:.2f}",
                    f"{m['p90_ms']:.2f}",
                ],
                alias + "/summary",
            )
        )
    out["timing.tex"] = table(
        ["System", "Images", "Median (ms)", "P90 (ms)"],
        rows,
        TIMING,
        "tab:timing",
        "Synchronized single-image latency on the same isolated device.",
    )
    rows = []
    for line in data[CONVENTIONS].splitlines():
        if not line.startswith("|"):
            continue
        parts = [s.strip().replace("`", "") for s in line.strip("|").split("|")]
        if len(parts) == 6 and (
            parts[-1] == REDACTED or re.fullmatch("[a-f0-9]{64}", parts[-1])
        ):
            # The public copy carries no image hash for private sources: key those rows by alias.
            key = parts[0] if parts[-1] == REDACTED else parts[-1]
            rows.append(
                (
                    [
                        parts[0].replace("synth/", "").replace("holdout/", ""),
                        parts[3],
                        parts[4],
                    ],
                    "representative/" + key,
                )
            )
    if len(rows) < 40:
        raise ValueError("Convention sample inventory incomplete")
    out["conventions.tex"] = table(
        ["Source sample", "Named light corners", "Labels mirrored?"],
        rows,
        CONVENTIONS,
        "tab:conventions",
        "One representative per source; unresolved means no independent corner anchor.",
        "p{.47\\linewidth}p{.25\\linewidth}l",
        long=True,
    )
    rows = []
    for name, asset in data[ASSETS]["families"].items():
        desc = asset["source_descriptor"]
        rows.append(
            (
                [
                    name,
                    desc["license"],
                    "credited under CC BY 4.0" if name == "phuong" else "grant recorded",
                ],
                "families/" + name,
            )
        )
    out["licences.tex"] = table(
        ["Asset family", "Recorded licence", "Release qualification"],
        rows,
        ASSETS,
        "tab:licences",
        "Recorded source grants; Phuong package attribution follows the author decision, not a recovered source version. Renders of every family are CC BY-NC 4.0.",
        "lp{.40\\linewidth}l",
        long=True,
    )
    licence_source = "docs/records/RELEASE_LICENCE_REVIEW_2026-09-10.md"
    dataset_rows = []
    in_table = False
    for line in data[licence_source].splitlines():
        if line.startswith("| Source | What it supplies"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and not line.startswith("|"):
            break
        if in_table:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != 4:
                raise ValueError("Unexpected dataset licence register row")
            clean = lambda text: (
                re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
                .replace("**", "")
                .replace("`", "")
            )
            dataset_rows.append(
                (
                    [clean(cells[0]), clean(cells[2]), clean(cells[3])],
                    "dataset/" + cells[0],
                )
            )
    if len(dataset_rows) < 7:
        raise ValueError("Dataset licence register is incomplete")
    out["licences.tex"] += table(
        ["Dataset / source", "Recorded terms", "Distribution qualification"],
        dataset_rows,
        licence_source,
        "tab:datasetlicences",
        "Dataset and position-source register from the recorded licence review.",
        "p{.18\\linewidth}p{.39\\linewidth}p{.35\\linewidth}",
        long=True,
    )
    receipt_names = sorted(name for name in data if name.endswith(".complete.json"))
    summary_names = sorted(
        name
        for name in data
        if name.endswith(".json")
        and not name.endswith(".complete.json")
        and name.startswith(SECTION4_PREFIXES)
    )
    if len(summary_names) != 15:
        raise ValueError("Section 4 result summaries missing from evidence index")
    rows = [
        (
            [
                str(Path(name).name),
                data[name].get("status", data[name].get("schema", "record"))
                if isinstance(data[name], dict)
                else "record",
            ],
            name,
        )
        for name in receipt_names
    ] + [
        (
            [
                str(Path(name).name),
                "result summary",
            ],
            name,
        )
        for name in summary_names
    ]
    # Index is a directory locator; long hashes are retained in the machine-readable custody index.
    out["receipts.tex"] = table(
        ["Receipt filename", "Status / schema"],
        rows,
        "index.json",
        "tab:receipts",
        "Receipt and result-summary index. Paths and the hash of each copy are in the evidence index.",
        "p{.49\\linewidth}p{.44\\linewidth}",
        long=True,
    )
    return out


def main():
    data, _ = load_evidence()
    out = ROOT / "generated/tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in build(data).items():
        (out / name).write_text(text)
    receipt = {
        "schema": "paper-tables/v1",
        "evidence_index_sha256": digest(ROOT / "evidence/index.json"),
        "script_sha256": digest(Path(__file__)),
        "outputs_sha256": {p.name: digest(p) for p in sorted(out.glob("*.tex"))},
        "published": False,
    }
    (ROOT / "records/tables.complete.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
