#!/usr/bin/env python3
"""Paper Table 1: the ChessQueries release checkpoint on its own four test sets, scored with the
authors' own code path (minimal_reproduction: same loaders, eval_transform, fp32/TF32, argmax), plus
what their eval.py does not give us: ms/img on the local GPU, per-board predictions for miss analysis,
optional 4-rotation TTA, and any extra image+FEN set (our official held-out fold, holdouts).

  CHESSQUERIES_DATA_ROOT=$DATA_ROOT/ext/cqdata \
  python tools/paper/cq_baseline.py --repo $DATA_ROOT/ext/chessqueries/repo/minimal_reproduction \
      --ckpt $DATA_ROOT/ext/chessqueries/chessqueries-vitL14-644-joint.safetensors \
      --datasets chesscog slcc [chessred cvchess] [--extra name=images_dir:labels_dir ...] \
      [--tta] [--bs 16] --out tools/reports/paper/cq_baseline_<date>

Expected (their README / eval.py): chessred_test 2,129 boards 99.5 exact; chesscog_test 342 98.5;
slcc_test 373 87.1; cvchess_all 352 87.6. A row that misses its count or its number by more than noise
means the local data layout is wrong, not the model. Writes <out>.json (metrics + timing + environment),
<out>.md (table row per set) and <out>.<set>.jsonl (id, gt, pred, n_wrong per board).
"""
import argparse, datetime, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PIECES = ".PNBRQKpnbrqk"


def labels_to_fen(labels):
    rows = []
    for r in range(8):
        row, run = "", 0
        for c in labels[r * 8:(r + 1) * 8]:
            if c == 0:
                run += 1
            else:
                if run:
                    row += str(run); run = 0
                row += PIECES[c]
        if run:
            row += str(run)
        rows.append(row)
    return "/".join(rows)


def load_extra(spec, Sample, fen_to_labels):
    """name=images_dir:labels_dir[:ids.txt] with <id>.png + <id>.txt (our repo's external-set contract);
    the optional third field restricts to the ids listed one per line (e.g. our held-out fold)."""
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    images, labels = parts[0], parts[1]
    ids = sorted(p.stem for p in Path(labels).glob("*.txt"))
    if len(parts) > 2:
        keep = {l.strip() for l in open(parts[2]) if l.strip()}
        ids = [i for i in ids if i in keep]
    out = []
    for i in ids:
        img = Path(images) / f"{i}.png"
        if not img.is_file():
            img = Path(images) / f"{i}.jpg"
        fen = Path(labels, f"{i}.txt").read_text().split()[0]
        out.append(Sample(img, fen_to_labels(fen), i))   # their Sample: (image_path, labels, sample_id)
    return name, out


def git_rev(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to chessqueries/minimal_reproduction")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", nargs="*", default=["chesscog", "slcc"], choices=["chessred", "chesscog", "slcc", "cvchess"])
    ap.add_argument("--extra", nargs="*", default=[], help="name=images_dir:labels_dir")
    ap.add_argument("--tta", action="store_true", help="4 rotations of the raw input, logits averaged")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resolution", type=int, default=644)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu when both GPUs belong to other lanes")
    ap.add_argument("--out", required=True, help="output prefix (no extension)")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import data as cqdata  # noqa: E402
    from data import BoardDataset, Sample, eval_transform, fen_to_labels  # noqa: E402
    from eval import load_model, score_boards  # noqa: E402

    loaders = {
        "chessred": ("chessred_test", lambda: cqdata.load_chessred("test"), 2129),
        "chesscog": ("chesscog_test", lambda: cqdata.load_chesscog("test"), 342),
        "slcc": ("slcc_test", lambda: cqdata.load_slcc("test"), 373),
        "cvchess": ("cvchess_all", cqdata.load_cvchess, 352),
    }
    sets = []
    for d in a.datasets:
        name, load, expected = loaders[d]
        samples = load()
        if len(samples) != expected:
            print(f"WARNING {name}: {len(samples)} samples, the authors' eval expects {expected}", file=sys.stderr)
        missing = [s.image_path for s in samples if not Path(s.image_path).is_file()]
        if missing or not samples:
            print(f"SKIP {name}: {len(missing)}/{len(samples)} images missing" + (f" (first {missing[0]})" if missing else ""), file=sys.stderr)
            continue
        sets.append((name, samples, expected))
    for spec in a.extra:
        name, samples = load_extra(spec, Sample, fen_to_labels)
        if not samples:
            print(f"SKIP {name}: no samples", file=sys.stderr)
            continue
        sets.append((name, samples, len(samples)))
    if not sets:
        sys.exit("nothing to evaluate")

    dev = a.device
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    torch.set_float32_matmul_precision("high")
    model = load_model(Path(a.ckpt), dev)
    tf = eval_transform(a.resolution)
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else f"cpu x{torch.get_num_threads()}"
    env = dict(gpu=gpu, torch=torch.__version__, python=platform.python_version(), host=platform.node(),
               ckpt=os.path.basename(a.ckpt), ckpt_bytes=os.path.getsize(a.ckpt), tta=a.tta, bs=a.bs,
               resolution=a.resolution, cq_repo_rev=git_rev(Path(a.repo).parent),
               data_root=os.environ.get("CHESSQUERIES_DATA_ROOT"), date=datetime.datetime.now().astimezone().isoformat(timespec="minutes"))
    results = {}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for name, samples, expected in sets:
        if a.limit:
            samples = samples[: a.limit]
        ds = BoardDataset(samples, tf)
        dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=dev == "cuda")
        preds, gts, rows = [], [], []
        # warm-up so the first batch's CUDA init does not land in the timing
        with torch.no_grad():
            x0, _ = ds[0]
            model(x0[None].to(dev))
        sync(); t0 = time.time(); gpu_s = 0.0
        with torch.no_grad():
            for images, labels in dl:
                x = images.to(dev, non_blocking=True)
                sync(); tg = time.time()
                if a.tta:
                    lg = torch.stack([model(torch.rot90(x, k, (2, 3))) for k in range(4)]).mean(0)
                else:
                    lg = model(x)
                sync(); gpu_s += time.time() - tg
                preds.extend(lg.argmax(-1).cpu().tolist()); gts.extend(labels.tolist())
        wall = time.time() - t0
        m = score_boards(preds, gts)
        m.update(ms_per_img_gpu=1000 * gpu_s / len(preds), ms_per_img_wall=1000 * wall / len(preds), expected_n=expected)
        hist = {}
        with open(f"{out}.{name}.jsonl", "w") as f:
            for s, p, g in zip(samples, preds, gts):
                nw = int(sum(int(x != y) for x, y in zip(p, g)))
                hist[nw] = hist.get(nw, 0) + 1
                f.write(json.dumps(dict(id=str(s.sample_id), image=str(s.image_path), n_wrong=nw,
                                        gt=labels_to_fen(g), pred=labels_to_fen(p))) + "\n")
        m["n_wrong_hist"] = dict(sorted(hist.items()))
        results[name] = m
        print(f"{name:14s} boards {m['n_boards_exact']:4d}/{m['n_boards']:4d} = {m['board_accuracy']:.4%}  "
              f"squares {m['per_square_accuracy']:.4%}  {m['ms_per_img_gpu']:.1f} ms/img gpu ({m['ms_per_img_wall']:.1f} wall)  {gpu}")
    json.dump(dict(env=env, results=results), open(f"{out}.json", "w"), indent=2)
    with open(f"{out}.md", "w") as f:
        f.write(f"# ChessQueries release checkpoint, their eval path{' + 4-rot TTA' if a.tta else ''}\n\n")
        f.write(f"{env['date']} on {env['host']} {gpu}, torch {env['torch']}, bs {a.bs}, {a.resolution} px, fp32/TF32, ckpt {env['ckpt']}\n\n")
        f.write("| set | n | exact | per-square | ms/img (gpu) | wrong-square histogram |\n|---|---:|---:|---:|---:|---|\n")
        for name, m in results.items():
            f.write(f"| {name} | {m['n_boards']} | {100*m['board_accuracy']:.2f} | {100*m['per_square_accuracy']:.2f} | "
                    f"{m['ms_per_img_gpu']:.1f} | {m['n_wrong_hist']} |\n")
    print("wrote", f"{out}.json", f"{out}.md")


if __name__ == "__main__":
    main()
