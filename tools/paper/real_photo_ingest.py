#!/usr/bin/env python3
"""Turn a phone's camera roll into a scoreable real-photo set, with the label taken from the sheet.

Input: a directory of photographs as the phone produced them (HEIC, JPEG or PNG), the frozen shoot
sheet (`shoot_sheet.jsonl`, one row per position with `id` and `fen`), and a mapping from each
photograph to a sheet position, an elevation and a parity. Output: the external-set contract that
`tools/paper/cq_baseline.py --extra` reads (`images/<id>.jpg` plus `labels/<id>.txt`), one id list per
parity, a manifest that pins the original bytes and the converted bytes by SHA-256, and a README with
the scoring command. Originals are copied untouched into `originals/`; nothing is cropped or edited,
only decoded, EXIF-rotated upright and re-encoded.

Two ways to map photographs to positions:

  --card rotation   the twenty-photograph rotation test (positions p001-p010, medium elevation, a1_dark
                    then a1_light per position). Photographs named `r<nn>_a1_dark.*` / `r<nn>_a1_light.*`
                    are mapped by name; otherwise all photographs must be unnamed and exactly twenty,
                    and they are taken in capture order (EXIF DateTimeOriginal, then file name).
  --map map.tsv     the full protocol: tab-separated `file  position  elev  parity`, one row per
                    photograph, `elev` one of low|mid|high, `parity` one of a1_dark|a1_light.

Ids follow the protocol: `p<idx>_<elev>_<side>_<parity>`, the side being the sheet's `near_side`.
With `--map`, a second frame of the same position and elevation gets `_2`, a third `_3`, in map order.

  uv run --with pillow --with pillow-heif python tools/paper/real_photo_ingest.py ingest \\
      --photos <photo_dir> --sheet <sheet dir>/shoot_sheet.jsonl \\
      --card rotation --phone "<device>" --sitting <date> --out <sheet dir>/rotation_shoot
  uv run --with pillow python tools/paper/real_photo_ingest.py check --out <sheet dir>/rotation_shoot

`ingest` refuses to write into an existing output directory: a shot set is frozen once ingested.
`check` re-hashes every file, re-parses every label and exits 3 on any disagreement.
"""

import argparse
import datetime
import hashlib
import io
import itertools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .fen_pools import placement
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fen_pools import placement

PHOTO_SUFFIXES = (".heic", ".heif", ".jpg", ".jpeg", ".png")
ELEVATIONS = ("low", "mid", "high")
PARITIES = ("a1_dark", "a1_light")
ROTATION_POSITIONS = tuple(f"p{i:03d}" for i in range(1, 11))
ROTATION_NAME = re.compile(r"^r(\d{2})_(a1_dark|a1_light)$")
ID_NAME = re.compile(r"^p\d{3}_(low|mid|high)_(white|black)_(a1_dark|a1_light)(_[2-9]\d*)?$")
JPEG_QUALITY = 95


class IngestError(Exception):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_photos(directory):
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise IngestError(f"photo directory not found: {directory}")
    photos = sorted(p for p in directory.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    if not photos:
        raise IngestError(f"no photographs ({', '.join(PHOTO_SUFFIXES)}) under {directory}")
    return photos


def read_sheet(path):
    rows = {}
    with Path(path).expanduser().open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "fen", "near_side"):
                if key not in row:
                    raise IngestError(f"sheet row without {key!r}: {line.strip()[:80]}")
            placement(row["fen"])  # raises on a malformed sheet
            if row["id"] in rows:
                raise IngestError(f"duplicate sheet id {row['id']}")
            rows[row["id"]] = row
    if not rows:
        raise IngestError(f"empty sheet: {path}")
    return rows


# --- decoding -------------------------------------------------------------------------------------

def _register_heif():
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        return True
    except ImportError:
        return False


def decode_upright(path):
    """Return (PIL image in RGB, EXIF dict) with the EXIF orientation applied. HEIC goes through
    pillow-heif when installed, else through macOS `sips`."""
    from PIL import Image, ImageOps

    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() in (".heic", ".heif") and not _register_heif():
        sips = shutil.which("sips")
        if not sips:
            raise IngestError(f"{path.name}: HEIC needs pillow-heif (`--with pillow-heif`) or macOS sips")
        with _TempDir() as tmp:
            converted = Path(tmp) / (path.stem + ".jpg")
            subprocess.run(
                [sips, "-s", "format", "jpeg", "-s", "formatOptions", "100", str(path), "--out", str(converted)],
                check=True,
                capture_output=True,
            )
            data = converted.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        exif = _exif_summary(image)
        upright = ImageOps.exif_transpose(image)
        upright.load()
        return upright.convert("RGB"), exif


class _TempDir:
    def __enter__(self):
        import tempfile

        self.path = tempfile.mkdtemp(prefix="real_photo_ingest_")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def _exif_summary(image):
    out = {"datetime_original": None, "make": None, "model": None, "orientation": None}
    try:
        exif = image.getexif()
    except Exception:  # pragma: no cover - PIL raises on a few odd files
        return out
    if not exif:
        return out
    out["make"] = exif.get(0x010F)
    out["model"] = exif.get(0x0110)
    out["orientation"] = exif.get(0x0112)
    try:
        sub = exif.get_ifd(0x8769)
        out["datetime_original"] = sub.get(0x9003) or exif.get(0x0132)
    except Exception:
        out["datetime_original"] = exif.get(0x0132)
    return {k: (str(v).strip() if v is not None else None) for k, v in out.items()}


def capture_key(path, exif):
    stamp = exif.get("datetime_original") or ""
    return (stamp, path.name)


# --- mapping --------------------------------------------------------------------------------------

def rotation_plan(photos, exifs):
    """Map the rotation card: by `r<nn>_<parity>` names when every file carries one, else by capture
    order over exactly twenty unnamed files."""
    named = {}
    unnamed = []
    for photo in photos:
        match = ROTATION_NAME.match(photo.stem)
        if match:
            index, parity = int(match.group(1)), match.group(2)
            if not 1 <= index <= len(ROTATION_POSITIONS):
                raise IngestError(f"{photo.name}: rotation index outside r01-r{len(ROTATION_POSITIONS):02d}")
            key = (ROTATION_POSITIONS[index - 1], parity)
            if key in named:
                raise IngestError(f"{photo.name}: duplicates {named[key].name}")
            named[key] = photo
        else:
            unnamed.append(photo)
    expected = [(pos, parity) for pos in ROTATION_POSITIONS for parity in PARITIES]
    if named and unnamed:
        raise IngestError(
            f"mixed naming: {len(named)} named r<nn>_<parity> files and {len(unnamed)} unnamed "
            f"({unnamed[0].name}, ...). Name all or none."
        )
    if named:
        missing = [f"{p}/{q}" for p, q in expected if (p, q) not in named]
        if missing:
            raise IngestError(f"rotation card incomplete, missing: {', '.join(missing)}")
        return [(named[key], key[0], "mid", key[1]) for key in expected]
    if len(unnamed) != len(expected):
        raise IngestError(
            f"rotation card by capture order needs exactly {len(expected)} photographs, found {len(unnamed)}. "
            "Name them r<nn>_a1_dark / r<nn>_a1_light to map a partial or extra set."
        )
    ordered = sorted(unnamed, key=lambda p: capture_key(p, exifs[p]))
    stamps = [exifs[p]["datetime_original"] for p in ordered]
    if any(stamps) and not all(stamps):
        raise IngestError("some photographs carry EXIF DateTimeOriginal and some do not; capture order is ambiguous")
    return [(photo, pos, "mid", parity) for photo, (pos, parity) in zip(ordered, expected)]


def map_plan(map_path, photos):
    by_name = {p.name: p for p in photos}
    plan, seen = [], set()
    with Path(map_path).expanduser().open() as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise IngestError(f"{map_path}:{lineno}: expected 4 tab-separated fields, got {len(parts)}")
            name, position, elev, parity = (p.strip() for p in parts)
            if name not in by_name:
                raise IngestError(f"{map_path}:{lineno}: no photograph named {name!r}")
            if elev not in ELEVATIONS:
                raise IngestError(f"{map_path}:{lineno}: elev must be one of {ELEVATIONS}, got {elev!r}")
            if parity not in PARITIES:
                raise IngestError(f"{map_path}:{lineno}: parity must be one of {PARITIES}, got {parity!r}")
            if name in seen:
                raise IngestError(f"{map_path}:{lineno}: {name} listed twice")
            seen.add(name)
            plan.append((by_name[name], position, elev, parity))
    unmapped = sorted(set(by_name) - seen)
    if unmapped:
        raise IngestError(f"{len(unmapped)} photographs not in the map: {', '.join(unmapped[:5])}")
    return plan


# --- ingest ---------------------------------------------------------------------------------------

def ingest(args):
    out = Path(args.out).expanduser()
    if out.exists():
        raise IngestError(f"{out} exists; an ingested set is frozen. Choose a new --out.")
    photos = list_photos(args.photos)
    sheet = read_sheet(args.sheet)

    decoded = {}
    exifs = {}
    for photo in photos:
        image, exif = decode_upright(photo)
        decoded[photo] = image
        exifs[photo] = exif

    if args.card == "rotation":
        plan = rotation_plan(photos, exifs)
    elif args.map:
        plan = map_plan(args.map, photos)
    else:
        raise IngestError("pass --card rotation or --map map.tsv")

    rows, ids = [], set()
    for photo, position, elev, parity in plan:
        if position not in sheet:
            raise IngestError(f"{photo.name}: position {position} is not on the sheet")
        srow = sheet[position]
        if "conventions" in srow and parity not in srow["conventions"]:
            raise IngestError(f"{photo.name}: sheet allows {srow['conventions']} for {position}, not {parity}")
        image_id = f"{position}_{elev}_{srow['near_side']}_{parity}"
        if image_id in ids:
            if args.card:
                raise IngestError(f"two photographs map to {image_id} ({photo.name})")
            # A mapped protocol may hold several frames of one position at one elevation; the
            # second and later frames take a sequence suffix in map order.
            image_id = next(f"{image_id}_{k}" for k in itertools.count(2) if f"{image_id}_{k}" not in ids)
        ids.add(image_id)
        rows.append((photo, image_id, position, elev, parity, srow))

    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    (out / "originals").mkdir()
    manifest = []
    for photo, image_id, position, elev, parity, srow in rows:
        original_copy = out / "originals" / f"{image_id}{photo.suffix.lower()}"
        shutil.copyfile(photo, original_copy)
        image_path = out / "images" / f"{image_id}.jpg"
        decoded[photo].save(image_path, "JPEG", quality=JPEG_QUALITY, subsampling=0)
        fen = placement(srow["fen"])
        (out / "labels" / f"{image_id}.txt").write_text(fen + "\n")
        manifest.append(
            {
                "id": image_id,
                "position": position,
                "source_group_id": srow.get("source_group_id"),
                "board": srow.get("board"),
                "near_side": srow["near_side"],
                "elev": elev,
                "parity": parity,
                "fen": fen,
                "piece_count": srow.get("piece_count"),
                "source_name": photo.name,
                "source_sha256": sha256_file(photo),
                "original_copy": str(original_copy.relative_to(out)),
                "image_sha256": sha256_file(image_path),
                "width": decoded[photo].width,
                "height": decoded[photo].height,
                "exif": exifs[photo],
                "phone": args.phone,
                "sitting": args.sitting,
            }
        )
    manifest.sort(key=lambda r: r["id"])
    with (out / "manifest.jsonl").open("x") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    for parity in PARITIES:
        listed = [r["id"] for r in manifest if r["parity"] == parity]
        (out / f"ids_{parity}.txt").write_text("".join(i + "\n" for i in listed))
    (out / "README.md").write_text(readme(out, manifest, args))
    print(f"ingested {len(manifest)} photographs into {out}")
    for parity in PARITIES:
        print(f"  {parity}: {sum(r['parity'] == parity for r in manifest)}")
    return 0


def readme(out, manifest, args):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    per = {p: sum(r["parity"] == p for r in manifest) for p in PARITIES}
    positions = sorted({r["position"] for r in manifest})
    lines = [
        f"# Real-photo set `{out.name}`",
        "",
        f"Ingested {now} by `tools/paper/real_photo_ingest.py` from `{Path(args.photos).expanduser()}`.",
        f"Sheet: `{Path(args.sheet).expanduser()}`. Mapping: "
        + (f"`--card {args.card}`" if args.card else f"`--map {args.map}`") + ".",
        f"Phone: {args.phone or 'not recorded'}. Sitting: {args.sitting or 'not recorded'}.",
        "",
        f"{len(manifest)} photographs, {len(positions)} positions ({positions[0]}-{positions[-1]}), "
        + ", ".join(f"{p}: {n}" for p, n in per.items()) + ".",
        "",
        "Labels come from the sheet, not from annotation. `originals/` holds the untouched camera files;",
        "`images/` holds the same frames decoded, EXIF-rotated upright and re-encoded as JPEG quality 95,",
        "no crop, no edit. `manifest.jsonl` pins both by SHA-256; run `check` before scoring.",
        "",
        "## Score (render-host, ChessQueries release checkpoint, one row per parity)",
        "",
        "Both sets go after one `--extra`; a second `--extra` flag replaces the first (argparse `nargs='*'`).",
        "",
        "```",
        "R=~/bench_experiments/codex_parity_20260909",
        "S=<this directory on render-host>",
        "python tools/paper/cq_baseline.py --repo $R/upstream/minimal_reproduction --ckpt $R/upstream/checkpoint.safetensors \\",
        "    --datasets \\",
        "    --extra a1_dark=$S/images:$S/labels:$S/ids_a1_dark.txt \\",
        "            a1_light=$S/images:$S/labels:$S/ids_a1_light.txt \\",
        f"    --out tools/reports/paper/real_photo_{out.name}_<date>",
        "```",
        "",
        "Corners, when wanted: `tools/paper/real_photo_corners.py annotate --photos images --out corners.jsonl`.",
        "",
    ]
    return "\n".join(lines)


# --- check ----------------------------------------------------------------------------------------

def check(args):
    out = Path(args.out).expanduser()
    problems = []
    manifest_path = out / "manifest.jsonl"
    if not manifest_path.is_file():
        print(f"FAIL: {manifest_path} missing")
        return 3
    rows = []
    with manifest_path.open() as stream:
        for lineno, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    problems.append(f"manifest line {lineno}: {exc}")
    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids in manifest")
    for row in rows:
        image_id = row.get("id", "?")
        if not ID_NAME.match(str(image_id)):
            problems.append(f"{image_id}: id does not follow p<idx>_<elev>_<side>_<parity>")
        image = out / "images" / f"{image_id}.jpg"
        label = out / "labels" / f"{image_id}.txt"
        original = out / str(row.get("original_copy", ""))
        for path, key in ((image, "image_sha256"), (original, "source_sha256")):
            if not path.is_file():
                problems.append(f"{image_id}: {path.relative_to(out)} missing")
            elif sha256_file(path) != row.get(key):
                problems.append(f"{image_id}: {path.relative_to(out)} SHA-256 differs from manifest")
        if not label.is_file():
            problems.append(f"{image_id}: label missing")
        else:
            text = label.read_text().split()
            try:
                fen = placement(text[0] if text else "")
            except ValueError as exc:
                problems.append(f"{image_id}: label unreadable ({exc})")
            else:
                if fen != row.get("fen"):
                    problems.append(f"{image_id}: label differs from manifest fen")
        if row.get("parity") not in PARITIES or row.get("elev") not in ELEVATIONS:
            problems.append(f"{image_id}: parity/elev outside the protocol")
        base_id = f"{row.get('position')}_{row.get('elev')}_{row.get('near_side')}_{row.get('parity')}"
        if image_id != base_id and not re.fullmatch(re.escape(base_id) + r"_[2-9]\d*", image_id):
            problems.append(f"{image_id}: id disagrees with its manifest fields")
        if image.is_file():
            try:
                from PIL import Image

                with Image.open(image) as decoded:
                    if decoded.size != (row.get("width"), row.get("height")):
                        problems.append(f"{image_id}: decoded size {decoded.size} differs from manifest")
            except Exception as exc:  # unreadable jpeg
                problems.append(f"{image_id}: image unreadable ({exc})")
    for parity in PARITIES:
        listed = (out / f"ids_{parity}.txt")
        expected = sorted(r["id"] for r in rows if r.get("parity") == parity)
        got = sorted(l.strip() for l in listed.read_text().splitlines() if l.strip()) if listed.is_file() else None
        if got != expected:
            problems.append(f"ids_{parity}.txt disagrees with the manifest")
    stray = sorted(p.name for p in (out / "images").glob("*") if p.stem not in set(ids)) if (out / "images").is_dir() else []
    if stray:
        problems.append(f"images not in manifest: {', '.join(stray[:5])}")
    if args.card == "rotation":
        have = {(r.get("position"), r.get("parity")) for r in rows}
        missing = [f"{p}/{q}" for p in ROTATION_POSITIONS for q in PARITIES if (p, q) not in have]
        if missing:
            problems.append(f"rotation card incomplete: {', '.join(missing)}")
        if len(rows) != 20:
            problems.append(f"rotation card has {len(rows)} rows, expected 20")
    if problems:
        print(f"FAIL: {len(problems)} problem(s)")
        for problem in problems:
            print("  " + problem)
        return 3
    per = {p: sum(r["parity"] == p for r in rows) for p in PARITIES}
    print(f"OK: {len(rows)} photographs, " + ", ".join(f"{p}={n}" for p, n in per.items()))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="convert, label and pin a shot set")
    ing.add_argument("--photos", required=True, help="directory of camera files")
    ing.add_argument("--sheet", required=True, help="shoot_sheet.jsonl")
    ing.add_argument("--card", choices=["rotation"], default=None)
    ing.add_argument("--map", default=None, help="tab-separated file\\tposition\\telev\\tparity")
    ing.add_argument("--phone", default=None, help="phone model, as the author reports it")
    ing.add_argument("--sitting", default=None, help="date of the sitting, YYYY-MM-DD")
    ing.add_argument("--out", required=True, help="new directory to create")
    ing.set_defaults(func=ingest)
    chk = sub.add_parser("check", help="re-verify an ingested set")
    chk.add_argument("--out", required=True)
    chk.add_argument("--card", choices=["rotation"], default=None, help="also require the full rotation card")
    chk.set_defaults(func=check)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except IngestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: converter failed: {exc.stderr.decode(errors='replace').strip()}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
