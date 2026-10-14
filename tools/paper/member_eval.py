#!/usr/bin/env python3
"""Task 2 of the paper plan: run the shipped submission (extracted tarball) with a member subset over an image+label
set, recording per board the prediction, the quad's smallest interior angle (the router's elevation proxy) and the
wrong-square count. One variant per invocation (predict.py keeps module-level state), then --compare joins the tsvs.

      --images DIR --labels DIR [--ids val.txt] [--limit N] --out tools/reports/paper/member_<set>_<variant>.tsv
  python tools/paper/member_eval.py compare a.tsv b.tsv ... --out tools/reports/paper/member_compare_<set>.md

Variants rewrite model/config.json in a scratch copy (weights symlinked): `raw` keeps the raw members only,
`warp_high` the high-branch warp member(s), `warp_low` the low-branch warp member(s), `member<i>` one member by index;
all of them drop the router so the kept members vote on every board. Angle bins follow the router: <50, 50-58,
58-65, 65-70, >=70 degrees, "none" when the detector finds no quad.
"""
import argparse, importlib.util, json, os, shutil, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

BINS = [(-1, 50, "<50"), (50, 58, "50-58"), (58, 65, "58-65"), (65, 70, "65-70"), (70, 999, ">=70")]


def cells(fen):
    out = []
    for r in fen.split("/"):
        for ch in r:
            out.extend(["."] * int(ch) if ch.isdigit() else [ch])
    return out


def n_wrong(a, b):
    return sum(x != y for x, y in zip(cells(a), cells(b)))


def angle_bin(angle):
    if angle is None or angle != angle:
        return "none"
    for lo, hi, name in BINS:
        if lo <= angle < hi:
            return name
    return "none"


def make_variant(sub, variant):
    """Scratch copy of the submission with the member subset written into model/config.json."""
    d = Path(tempfile.mkdtemp(prefix=f"member_{variant}_"))
    for f in os.listdir(sub):
        if f == "model":
            continue
        src = Path(sub) / f
        if src.is_file():
            shutil.copy(src, d / f)
    (d / "model").mkdir()
    for f in os.listdir(Path(sub) / "model"):
        if f != "config.json":
            os.symlink((Path(sub) / "model" / f).resolve(), d / "model" / f)
    cfg = json.load(open(Path(sub) / "model" / "config.json"))
    members = cfg["members"]
    if variant == "full":
        pass
    else:
        if variant == "raw":
            keep = [m for m in members if "raw" in m["arch"]]
        elif variant == "warp_high":
            keep = [m for m in members if "raw" not in m["arch"] and m.get("branch") == "high"]
        elif variant == "warp_low":
            keep = [m for m in members if "raw" not in m["arch"] and m.get("branch") == "low"]
        elif variant.startswith("member"):
            keep = [members[int(variant[6:])]]
        else:
            raise SystemExit(f"unknown variant {variant}")
        assert keep, (variant, members)
        for m in keep:
            m["branch"] = "both"
        cfg["members"] = keep
        cfg["route_angle"] = None
        cfg["route_low"] = None
    json.dump(cfg, open(d / "model" / "config.json", "w"), indent=1)
    return d, cfg


def run(a):
    os.environ.setdefault("BUDGET_S", "1000000000")   # no staged rotation cuts in an offline eval
    d, cfg = make_variant(a.sub, a.variant)
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("predict", d / "predict.py")
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    import board  # noqa: E402  (the submission's detector, for the angle column)
    import cv2  # noqa: E402
    ids = sorted(p.stem for p in Path(a.labels).glob("*.txt"))
    if a.ids:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        ids = [i for i in ids if i in keep]
    if a.limit:
        ids = ids[: a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    t_pred = 0.0; ex = 0; sq = 0
    with open(out, "w") as f:
        f.write("id\tangle\tpred\tgt\tn_wrong\n")
        for i in ids:
            img = Path(a.images) / f"{i}.png"
            if not img.is_file():
                img = Path(a.images) / f"{i}.jpg"
            gt = Path(a.labels, f"{i}.txt").read_text().split()[0]
            t0 = time.time(); pred = P.predict(str(img)); t_pred += time.time() - t0
            bgr = cv2.imread(str(img)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            c = board.find_corners(rgb)
            ang = P._min_interior_angle(c) if c is not None else float("nan")
            nw = n_wrong(gt, pred); ex += nw == 0; sq += 64 - nw
            f.write(f"{i}\t{ang:.2f}\t{pred}\t{gt}\t{nw}\n")
    n = len(ids)
    summary = dict(variant=a.variant, members=[m["arch"] for m in cfg["members"]], n=n, exact=ex / n, per_square=sq / (64 * n),
                   ms_per_img=1000 * t_pred / n, sub=str(a.sub), images=str(a.images))
    print(json.dumps(summary))
    json.dump(summary, open(str(out) + ".json", "w"), indent=1)
    shutil.rmtree(d, ignore_errors=True)


def load_tsv(p):
    rows = {}
    for line in open(p):
        if line.startswith("id\t") or not line.strip():
            continue
        i, ang, pred, gt, nw = line.rstrip("\n").split("\t")
        rows[i] = dict(angle=float(ang), pred=pred, gt=gt, nw=int(nw))
    return rows


def compare(a):
    runs = {Path(p).stem: load_tsv(p) for p in a.tsvs}
    names = list(runs)
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    ref = runs[names[0]]
    lines = [f"# member comparison — {len(ids)} boards\n", "| variant | exact | per-square | " + " | ".join(f"exact {b[2]}" for b in BINS) + " | exact none |",
             "|---|---:|---:|" + "---:|" * (len(BINS) + 1)]
    bins_n = defaultdict(int)
    for i in ids:
        bins_n[angle_bin(ref[i]["angle"])] += 1
    for nm in names:
        r = runs[nm]
        ex = sum(r[i]["nw"] == 0 for i in ids); sq = sum(64 - r[i]["nw"] for i in ids)
        per = defaultdict(int)
        for i in ids:
            per[angle_bin(ref[i]["angle"])] += r[i]["nw"] == 0
        cells_ = [f"{per[b[2]]}/{bins_n[b[2]]}" for b in BINS] + [f"{per['none']}/{bins_n['none']}"]
        lines.append(f"| {nm} | {ex}/{len(ids)} = {100*ex/len(ids):.2f} | {100*sq/(64*len(ids)):.2f} | " + " | ".join(cells_) + " |")
    lines.append("")
    for x in names[1:]:
        only_ref = [i for i in ids if ref[i]["nw"] == 0 and runs[x][i]["nw"] > 0]
        only_x = [i for i in ids if runs[x][i]["nw"] == 0 and ref[i]["nw"] > 0]
        lines.append(f"- {names[0]} right / {x} wrong: {len(only_ref)} {only_ref[:10]}")
        lines.append(f"- {x} right / {names[0]} wrong: {len(only_x)} {only_x[:10]}")
        # wrong-square histogram of x on boards it misses
        hist = defaultdict(int)
        for i in ids:
            if runs[x][i]["nw"]:
                hist[min(runs[x][i]["nw"], 8)] += 1
        lines.append(f"- {x} wrong-square histogram (8 = 8+): {dict(sorted(hist.items()))}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--sub", required=True); r.add_argument("--variant", required=True)
    r.add_argument("--images", required=True); r.add_argument("--labels", required=True)
    r.add_argument("--ids", default=None); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", required=True)
    c = sp.add_parser("compare")
    c.add_argument("tsvs", nargs="+"); c.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
