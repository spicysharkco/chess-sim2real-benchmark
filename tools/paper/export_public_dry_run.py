"""Build and audit a reproducible, non-publishing research export tree.

PDF auditing requires the system Poppler executables pdftotext, pdfinfo and
pdfdetach, plus qpdf, on PATH, not Python packages. Missing or failing decoders
make the audit fail closed.
"""

import argparse
import ast
import csv
import fnmatch
import functools
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

SCHEMA = "public-export-policy/v1"
RECEIPT_SCHEMA = "public-export-dry-run/v1"
TOKEN_COUNT = 19
TOKEN_DIGEST = "[redacted: withheld digest]"
V2_POLICY_DIGEST = "[redacted: withheld digest]"
V3_POLICY_DIGEST = "[redacted: withheld digest]"
V4_POLICY_DIGEST = "[redacted: withheld digest]"
FINGERPRINT_COUNT = 84
FINGERPRINT_DIGEST = "[redacted: withheld digest]"
TEXT_RULE_KINDS = {"replace", "drop_lines", "md_table_columns"}
JSON_RULE_KINDS = {"json_entries", "json_fields", "json_drop_keys"}
RULE_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}
ALLOWED_TOOL_SUFFIXES = {".py", ".json", ".sh"}
BENCHMARK_SUFFIXES = {".json", ".jsonl", ".md"}
BENCHMARK_PREFIXES = ("benchmark", "adaptation", "design", "exclusion_audit")
PAPER_ROOT = PurePosixPath("paper")
EVIDENCE_ROOT = PurePosixPath("paper/evidence")
SIDECAR_SCHEMA = "paper-evidence-sources/v1"
WITHHELD_DIGEST = "[redacted: withheld digest]"
HEX_RUN = re.compile(r"[0-9A-Fa-f]{12,}")
LICENSING_PINS = ("required_notices_sha256", "decision_sha256", "attribution_sha256")
ROW_INVENTORIES = ("benchmark_v2/benchmark.jsonl", "benchmark_v2/adaptation.jsonl")
LAUNCHER_DIGEST = ("source_sha256", "render_batch.mjs")
INVENTORY_DIGESTS = {
    "launcher": LAUNCHER_DIGEST,
    "exclusion": ("excluded_fen_sha256",),
}
WRAPPED_HEX = re.compile(r"(?<=[0-9A-Fa-f])(?:[⌋↩]|,→|\s)+(?=[0-9A-Fa-f])")
DIGEST_SURFACES = frozenset(
    {"pdf_text", "pdf_metadata", "pdf_xmp", "pdf_custom_metadata", "pdf_urls"}
    | {"pdf_strings", "text", "decoded_json"}
)
RELEASE_GATES = ("digest_rule", "private_pools", "mandatory_paper_scans")
PIN = re.compile(r"[0-9a-f]{64}")
DIGEST64 = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])")
# The hex form of a PDF text string of 15 characters (the byte order mark and 30
# bytes), such as an outline title, is 64 hex digits long. It holds text, which
# the surfaces read in decoded form, and no digest.
PDF_TEXT_STRING_HEX = re.compile(r"<feff[0-9a-f]{60}>", re.IGNORECASE)
# Extensions of the file a digest is said to cover. A digest of a text file that
# does not ship lets a reader test guesses for its content.
TEXT_PRE_IMAGES = frozenset(
    {".json", ".jsonl", ".tsv", ".csv", ".txt", ".md", ".log", ".env", ".yaml"}
    | {".yml", ".py", ".mjs", ".js", ".sh", ".tex", ".html", ".toml", ".cfg"}
    | {".ini", ".fen", ".pgn", ".lock"}
)
BINARY_PRE_IMAGES = frozenset(
    {".png", ".jpg", ".jpeg", ".pt", ".pth", ".safetensors", ".npz", ".npy", ".bin"}
    | {".gz", ".zip", ".tar", ".pdf", ".glb", ".gltf", ".blend", ".ttf", ".woff"}
    | {".woff2", ".onnx", ".ckpt", ".webp", ".heic", ".hdr", ".exr", ".mp4", ".zst"}
)
FILE_SUFFIXES = TEXT_PRE_IMAGES | BINARY_PRE_IMAGES
# Fields that name the file a sibling digest covers.
NAME_FIELDS = (
    "path",
    "file",
    "name",
    "rel",
    "relative_path",
    "source",
    "filename",
    "source_path",
    "original_path",
    "output",
    "receipt",
    "src",
    "dst",
    "target",
)
DIGEST_RULE_KEYS = {"classes", "public", "withheld", "prose"}
PDF_STREAM_LIMIT = 1 << 28
# Views where a timestamp cannot be read reliably; the text views cover them.
SEPARATORS = re.compile(r"[^0-9A-Za-z]+")
TIMESTAMP_BLIND_SURFACES = {"binary", "pdf_stream"}
OFFSET_TIMESTAMP = re.compile(
    r"(?<![0-9])(?P<date>\d{4}-\d{2}-\d{2})(?P<sep>[T ])"
    r"(?P<time>\d{2}:\d{2}(?::\d{2})?)(?P<fraction>\.\d+)?"
    r"(?P<sign>[+-])(?P<hours>\d{2})(?P<colon>:?)(?P<minutes>\d{2})(?![0-9:])"
)
DATE_FIELD = re.compile(r'("date"\s*:\s*")(?P<stamp>[^"\\]*)(")')
DATE_WITH_TIME = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})[T ]\d{2}:?\d{2}\S*")
FILE_NAME_DATE = re.compile(r"(?<![0-9])(\d{4})-?(\d{2})-?(\d{2})(?![0-9])")
# Other writings of a local offset that the gate reads: ISO with a space before the
# offset or with hours only, the PDF date string, and RFC 2822.
GATE_OFFSET_TIMESTAMP = re.compile(
    r"(?<![0-9])\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?[ ]?"
    r"[+-](?P<hours>\d{2})(?::?(?P<minutes>\d{2}))?(?![0-9:])"
)
PDF_OFFSET_DATE = re.compile(
    r"(?<![A-Za-z0-9])D:\d{12,14}[+-](?P<hours>\d{2})(?:'?(?P<minutes>\d{2})'?)?(?![0-9])"
)
RFC2822_OFFSET = re.compile(
    r"(?<![0-9])\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4} "
    r"\d{2}:\d{2}(?::\d{2})? ?[+-](?P<hours>\d{2})(?P<minutes>\d{2})(?![0-9:])"
)
BASIC_OFFSET_TIMESTAMP = re.compile(
    r"(?<![0-9])\d{8}T\d{4}(?:\d{2})?(?:\.\d+)?[+-](?!0000)\d{4}(?![0-9])"
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IMAGE_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
PNG_DECODE_LIMIT = 1 << 20
PDF_HEX_STRING = re.compile(rb"<([0-9A-Fa-f\s]*)>")
PDF_LITERAL_STRING = re.compile(
    rb"\(((?:[^()\\]|\\[\s\S]|\((?:[^()\\]|\\[\s\S])*\))*)\)"
)
PDF_STRING_ESCAPE = re.compile(rb"\\(?:([0-7]{1,3})|(\r\n|[\s\S]))")
PDF_STRING_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f"}
PDF_EMBEDDED_FILE = re.compile(
    rb"/(?:EmbeddedFile|FileAttachment)|/EF\s*(?:<<|\d+\s+\d+\s+R)"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path, value):
    Path(path).write_bytes(_json_bytes(value))


def _relative(value, *, label):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid {label} path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"invalid {label} path")
    return path


def _root(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a non-symlink directory")
    return path.resolve()


def _source(root, relative, label):
    relative = _relative(str(relative), label=label)
    path = root.joinpath(*relative.parts)
    cursor = root
    has_symlink = False
    for part in relative.parts:
        cursor = cursor / part
        has_symlink = has_symlink or cursor.is_symlink()
    if has_symlink or not path.is_file():
        raise ValueError(f"missing or symlinked {label}: {relative}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes source root: {relative}")
    return relative, resolved


def _git_files(repo):
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        raise ValueError("git tracked-file inventory failed")
    values = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            values.append(_relative(raw.decode(), label="tracked"))
        except UnicodeDecodeError as error:
            raise ValueError("tracked path is not UTF-8") from error
    return sorted(values, key=str)


def _excluded(relative, policy):
    value = str(relative)
    prefixes = [
        _relative(x, label="excluded prefix")
        for x in policy.get("exclude_prefixes", [])
    ]
    if any(
        relative == prefix or relative.is_relative_to(prefix) for prefix in prefixes
    ):
        return True
    return any(
        fnmatch.fnmatchcase(value, pattern)
        for pattern in policy.get("exclude_globs", [])
    )


def select_repo_files(repo, policy):
    tracked = _git_files(repo)
    tracked_set = set(tracked)
    exact_key = "repo_files" if "repo_files" in policy else "exact_files"
    exact = [_relative(x, label="repository") for x in policy.get(exact_key, [])]
    prefixes = [
        _relative(x, label="repository prefix") for x in policy.get("repo_prefixes", [])
    ]
    missing = [str(path) for path in exact if path not in tracked_set]
    if missing:
        raise ValueError(
            "required tracked repository files missing: " + ", ".join(missing)
        )
    for prefix in prefixes:
        if not any(path == prefix or path.is_relative_to(prefix) for path in tracked):
            raise ValueError(f"required repository prefix is empty: {prefix}")
    selected = {
        path
        for path in tracked
        if path in exact
        or any(path == prefix or path.is_relative_to(prefix) for prefix in prefixes)
    }
    selected = {path for path in selected if not _excluded(path, policy)}
    # An evidence copy mirrors its original's path, so the exclusion list applies
    # to the mirrored path as well unless the policy names the original.
    allowed = set(policy.get("evidence_excluded_originals", []))
    mirrored = set()
    for path in selected:
        if not path.is_relative_to(EVIDENCE_ROOT) or path == EVIDENCE_ROOT:
            continue
        inner = path.relative_to(EVIDENCE_ROOT)
        if str(inner) in allowed:
            mirrored.add(str(inner))
        elif _excluded(inner, policy):
            raise ValueError(f"evidence copy of an excluded path: {path}")
    if mirrored != allowed:
        raise ValueError("unused excluded-original evidence allowance")
    for path in exact:
        if path not in selected:
            raise ValueError(f"required repository file was excluded: {path}")
    for path in selected:
        if (
            path.is_relative_to(PurePosixPath("tools/paper"))
            and path.suffix not in ALLOWED_TOOL_SUFFIXES
            and path != PurePosixPath("tools/paper/LICENSE.md")
        ):
            raise ValueError(f"unsupported paper-tool file type: {path}")
    return sorted(selected, key=str)


def select_benchmark_files(root, policy):
    selected = set()
    for path in root.iterdir():
        matching = path.suffix in BENCHMARK_SUFFIXES and any(
            path.name == prefix + path.suffix or path.name.startswith(prefix + ".")
            for prefix in BENCHMARK_PREFIXES
        )
        if matching and path.is_symlink():
            raise ValueError(f"symlinked benchmark input: {path.name}")
        if matching and path.is_file():
            selected.add(PurePosixPath(path.name))
    manifests = root / "manifests"
    if manifests.is_dir() and not manifests.is_symlink():
        for path in manifests.iterdir():
            if path.suffix == ".jsonl" and path.is_symlink():
                raise ValueError(f"symlinked benchmark input: manifests/{path.name}")
            if path.is_file() and path.suffix == ".jsonl":
                selected.add(PurePosixPath("manifests", path.name))
    pools = root / "inputs" / "pools.json"
    if pools.is_symlink():
        raise ValueError("symlinked benchmark input: inputs/pools.json")
    if pools.is_file():
        selected.add(PurePosixPath("inputs/pools.json"))

    required = policy.get("benchmark_required_files", [])
    required = {_relative(x, label="benchmark") for x in required}
    missing = required - selected
    if missing:
        raise ValueError(
            "required benchmark files missing: "
            + ", ".join(map(str, sorted(missing, key=str)))
        )
    if not selected:
        raise ValueError("benchmark selection is empty")
    for prefix in BENCHMARK_PREFIXES:
        if not any(
            path.parent == PurePosixPath(".") and path.name.startswith(prefix)
            for path in selected
        ):
            raise ValueError(f"benchmark category is empty: {prefix}")
    if not any(path.is_relative_to(PurePosixPath("manifests")) for path in selected):
        raise ValueError("benchmark manifests are missing")
    if PurePosixPath("inputs/pools.json") not in selected:
        raise ValueError("benchmark pool receipt is missing")
    locked = policy.get("benchmark_files_sha256")
    if locked is not None:
        if not isinstance(locked, dict):
            raise ValueError("benchmark file inventory is malformed")
        expected_paths = {
            _relative(name, label="benchmark inventory") for name in locked
        }
        if selected != expected_paths:
            added = sorted(map(str, selected - expected_paths))
            missing = sorted(map(str, expected_paths - selected))
            raise ValueError(
                f"benchmark inventory mismatch: added={added}, missing={missing}"
            )
        for relative in selected:
            if sha256(root.joinpath(*relative.parts)) != locked[str(relative)]:
                raise ValueError(f"benchmark inventory hash mismatch: {relative}")
    return sorted(selected, key=str)


def validate_benchmark_bindings(root, selected, inventory):
    """Validate metadata receipts without opening frame data."""
    selected = set(selected)
    hashes = {str(path): sha256(root.joinpath(*path.parts)) for path in selected}
    inventory_receipts = inventory.get("receipts_sha256", {})
    if not isinstance(inventory_receipts, dict):
        raise TypeError("asset receipt inventory is malformed")
    for name, expected in inventory_receipts.items():
        relative = _relative(name, label="asset receipt")
        if relative in selected and hashes[str(relative)] != expected:
            raise ValueError(f"asset receipt hash mismatch: {relative}")

    links = {
        "design_receipt_sha256": "design.complete.json",
        "benchmark_receipt_sha256": "benchmark.complete.json",
        "exclusion_audit_sha256": "exclusion_audit.json",
        "inventory_sha256": "inputs/pools.json",
    }
    receipts = sorted(
        (path for path in selected if path.name.endswith(".complete.json")), key=str
    )
    for receipt_path in receipts:
        try:
            receipt = json.loads(root.joinpath(*receipt_path.parts).read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid benchmark receipt JSON: {receipt_path}"
            ) from error
        artifacts = receipt.get("artifacts_sha256")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(
                f"benchmark receipt has no artifact bindings: {receipt_path}"
            )
        for name, expected in artifacts.items():
            artifact = _relative(name, label="receipt artifact")
            if artifact not in selected:
                raise ValueError(f"receipt-bound artifact is absent: {artifact}")
            if hashes[str(artifact)] != expected:
                raise ValueError(f"receipt artifact hash mismatch: {artifact}")
        for field, target in links.items():
            if receipt.get(field) is None:
                continue
            relative = PurePosixPath(target)
            if relative not in selected:
                raise ValueError(f"receipt-linked metadata is absent: {relative}")
            if receipt[field] != hashes[str(relative)]:
                raise ValueError(f"receipt link hash mismatch: {field}")


def select_frozen_files(repo, frozen, policy):
    inventory_value = policy.get(
        "asset_inventory", "docs/records/BENCHMARK_ASSET_INVENTORY_2026-09-10.json"
    )
    relative, inventory_path = _source(repo, inventory_value, "asset inventory")
    if (
        policy.get("asset_inventory_sha256")
        and sha256(inventory_path) != policy["asset_inventory_sha256"]
    ):
        raise ValueError("asset inventory hash mismatch")
    inventory = json.loads(inventory_path.read_text())
    hashes = dict(inventory.get("runtime_source_sha256", {}))
    for family in inventory.get("families", {}).values():
        for name, expected in family.get("actual_source_files_sha256", {}).items():
            if name in hashes and hashes[name] != expected:
                raise ValueError(f"conflicting frozen hash in inventory: {name}")
            hashes[name] = expected
    if not hashes:
        raise ValueError("frozen source inventory is empty")
    result = []
    for name, expected in sorted(hashes.items()):
        source_relative, source = _source(frozen, name, "frozen source")
        if sha256(source) != expected:
            raise ValueError(f"frozen source hash mismatch: {source_relative}")
        result.append((source_relative, source, expected))
    return relative, inventory_path, inventory, result


def _add_copy(copies, destination, source, category, expected=None):
    destination = _relative(str(destination), label="destination")
    actual = sha256(source)
    if expected is not None and actual != expected:
        raise ValueError(f"source hash mismatch: {source}")
    old = copies.get(destination)
    if old and old["sha256"] != actual:
        raise ValueError(f"conflicting export destination: {destination}")
    if old:
        old["categories"].add(category)
        if source not in {item[0] for item in old["sources"]}:
            old["sources"].append((source, actual))
    else:
        copies[destination] = {
            "source": source,
            "sha256": actual,
            "categories": {category},
            "sources": [(source, actual)],
        }


def plan_export(repo, benchmark, frozen, policy):
    copies = {}
    for relative in select_repo_files(repo, policy):
        _, source = _source(repo, relative, "repository source")
        _add_copy(copies, relative, source, "repository")
    inventory_relative, inventory_path, inventory, frozen_files = select_frozen_files(
        repo, frozen, policy
    )
    if policy.get("include_asset_inventory", True):
        _add_copy(copies, inventory_relative, inventory_path, "licence_inventory")
    updates = policy.get("updated_package_tools", {})
    used_updates = set()
    for relative, source, expected in frozen_files:
        name = str(relative)
        if name not in updates:
            _add_copy(copies, relative, source, "frozen_source", expected)
            continue
        # Only pinned paper tools, the launch wrapper and the per-family asset
        # README.md notes can supersede archived bytes. Scene renderers and asset
        # meshes or textures are never eligible for this exception.
        pin = updates[name]
        current = copies.get(relative)
        if (
            not (
                (
                    relative.parent == PurePosixPath("tools/paper")
                    and relative.suffix == ".py"
                )
                or relative == PurePosixPath("rendering/render_batch.mjs")
                or (
                    relative.parent.parent == PurePosixPath("rendering/assets")
                    and relative.name == "README.md"
                )
            )
            or not current
            or pin
            != {
                "frozen_sha256": expected,
                "current_sha256": current["sha256"],
            }
        ):
            raise ValueError(f"invalid package tool update: {name}")
        current["sources"].append((source, expected))
        current["superseded_frozen_source"] = source
        current["superseded_frozen_source_sha256"] = expected
        current["categories"].add("updated_package_tool")
        used_updates.add(name)
    if used_updates != set(updates):
        raise ValueError("unused package tool update")
    benchmark_files = select_benchmark_files(benchmark, policy)
    validate_benchmark_bindings(benchmark, benchmark_files, inventory)
    for relative in benchmark_files:
        _, source = _source(benchmark, relative, "benchmark source")
        _add_copy(copies, PurePosixPath("benchmark_v2") / relative, source, "benchmark")
    if policy.get("package_readme"):
        _, readme = _source(repo, policy["package_readme"], "public package README")
        _add_copy(copies, PurePosixPath("README.md"), readme, "repository")
        prefix = policy.get("package_readme_prefix")
        if prefix:
            _, intro = _source(repo, prefix["path"], "public README prefix")
            if sha256(intro) != prefix["sha256"] or not readme.read_bytes().startswith(
                intro.read_bytes()
            ):
                raise ValueError(
                    "public README prefix differs from the approved source"
                )
            copies[PurePosixPath("README.md")]["sources"].append(
                (intro, prefix["sha256"])
            )
    categories = Counter()
    for value in copies.values():
        for category in value["categories"]:
            categories[category] += 1
    required_categories = policy.get(
        "required_categories",
        ["repository", "licence_inventory", "frozen_source", "benchmark"],
    )
    missing = [name for name in required_categories if not categories[name]]
    if missing:
        raise ValueError("empty required export categories: " + ", ".join(missing))
    return copies


def _offset_summary(data, needle):
    count, first, start = 0, [], 0
    while True:
        position = data.find(needle, start)
        if position < 0:
            return count, first
        count += 1
        if len(first) < 3:
            first.append(position)
        start = position + 1


def _is_binary(data):
    if b"\0" in data:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


class PDFDecoderUnavailable(RuntimeError):
    """A required Poppler executable is absent; PDF audits must fail closed."""


def read_pdf(path):
    """Decode all PDF surfaces using the system Poppler tools, never raw bytes."""
    commands = {
        "pdf_text": ["pdftotext", str(path), "-"],
        "pdf_metadata": ["pdfinfo", str(path)],
        "pdf_xmp": ["pdfinfo", "-meta", str(path)],
        "pdf_custom_metadata": ["pdfinfo", "-custom", str(path)],
        "pdf_urls": ["pdfinfo", "-url", str(path)],
    }
    outputs = {}
    for surface, command in commands.items():
        try:
            proc = subprocess.run(command, capture_output=True, check=False)
        except FileNotFoundError as error:
            raise PDFDecoderUnavailable(
                "PDF decoding requires Poppler executables on PATH "
                f"(pdftotext and pdfinfo); missing: {command[0]}"
            ) from error
        except OSError as error:
            raise RuntimeError(f"PDF parser could not start: {command[0]}") from error
        if proc.returncode:
            raise RuntimeError(f"PDF parser failed: {command[0]}")
        outputs[surface] = proc.stdout
    return outputs


def _run_pdf_tool(command):
    try:
        proc = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as error:
        raise PDFDecoderUnavailable(
            f"PDF structure scan requires {command[0]} on PATH"
        ) from error
    except OSError as error:
        raise RuntimeError(f"PDF parser could not start: {command[0]}") from error
    if proc.returncode:
        raise RuntimeError(f"PDF parser failed: {command[0]}")
    return proc.stdout


def _pdf_string_texts(objects):
    """Decode every PDF string (literal or hex) as bytes, Latin-1, NUL-stripped and UTF-16BE."""

    def unescape(match):
        octal, other = match.groups()
        if octal:
            return bytes([int(octal, 8) & 0xFF])
        if other in (b"\r", b"\n", b"\r\n"):
            return b""
        return PDF_STRING_ESCAPES.get(other, other)

    values = {
        PDF_STRING_ESCAPE.sub(unescape, match.group(1))
        for match in PDF_LITERAL_STRING.finditer(objects)
    }
    for match in PDF_HEX_STRING.finditer(objects):
        digits = b"".join(match.group(1).split())
        values.add(bytes.fromhex((digits + b"0" * (len(digits) % 2)).decode()))
    texts = set()
    for value in values:
        texts.add(value.decode("latin-1"))
        texts.add(value.replace(b"\0", b"").decode("latin-1"))
        texts.add(value.removeprefix(b"\xfe\xff").decode("utf-16-be", "replace"))
    return "\n".join(sorted(texts))


def read_pdf_structure(path):
    """Decode the whole PDF object graph into text surfaces, failing closed.

    read_pdf sees only the text layer and the metadata calls. This adds what the
    file carries behind them: annotations, outlines, named destinations, orphan
    objects and every stream, decompressed through qpdf (QDF mode writes each
    object readable; unreferenced objects are kept), plus every string in its
    literal, hex, UTF-16BE and hex-of-UTF-16BE form. More than one %%EOF means an
    incremental update kept an earlier revision, and any embedded file is refused.
    Missing qpdf or pdfdetach, or a non-zero exit (qpdf exits 3 on warnings), is a
    parser failure.

    Known residual blind spots, not solved here: text drawn with subset-font glyph
    codes outside the page box is invisible both to pdftotext and to this byte
    scan; lossy image streams (DCT, JPX, CCITT, JBIG2) stay opaque; bytes after
    the final %%EOF are outside the object graph and are not read.
    """
    if Path(path).read_bytes().count(b"%%EOF") != 1:
        raise ValueError(
            "PDF must hold exactly one %%EOF marker (an incremental update keeps "
            "earlier revisions)"
        )
    objects = _run_pdf_tool(
        [
            "qpdf",
            "--qdf",
            "--object-streams=disable",
            "--preserve-unreferenced",
            str(path),
            "-",
        ]
    )
    listing = _run_pdf_tool(["pdfdetach", "-list", str(path)])
    count = re.match(rb"(\d+) embedded files?\b", listing)
    if count is None:
        raise ValueError("PDF embedded-file listing was not understood")
    if int(count.group(1)) or PDF_EMBEDDED_FILE.search(objects):
        raise ValueError("PDF carries an embedded file")
    return {
        "pdf_objects": objects.decode("latin-1"),
        "pdf_strings": _pdf_string_texts(objects),
    }


def _png_inflate(data):
    inflater = zlib.decompressobj()
    inflated = inflater.decompress(data, PNG_DECODE_LIMIT)
    if inflater.unconsumed_tail:
        raise ValueError("PNG metadata chunk is too large to decode")
    if inflater.unused_data:
        raise ValueError("PNG chunk has bytes after the end of its zlib stream")
    return inflated


def _png_check_image_data(chunks):
    """Drain the concatenated IDAT zlib stream; pixels are not scanned, tails fail."""
    inflater = zlib.decompressobj()
    produced = inflater.decompress(b"".join(chunks), PNG_DECODE_LIMIT)
    while produced and not inflater.eof:
        produced = inflater.decompress(inflater.unconsumed_tail, PNG_DECODE_LIMIT)
    if inflater.unused_data:
        raise ValueError("PNG image data has bytes after the end of its zlib stream")


def _png_chunk_text(kind, body):
    if kind == b"zTXt":
        keyword, _, rest = body.partition(b"\0")
        return keyword + b"\0" + _png_inflate(rest[1:])
    if kind == b"iCCP":
        name, _, rest = body.partition(b"\0")
        return name + b"\0" + _png_inflate(rest[1:])
    if kind == b"iTXt":
        keyword, _, rest = body.partition(b"\0")
        flag, rest = rest[:1], rest[2:]
        language, _, rest = rest.partition(b"\0")
        translated, _, text = rest.partition(b"\0")
        if flag == b"\x01":
            text = _png_inflate(text)
        return b"\0".join((keyword, language, translated, text))
    return body


def _png_decode(data):
    """Return (metadata text, bytes after IEND); raise on any other unread byte."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    position = len(PNG_SIGNATURE)
    parts, image_data = [], []
    while True:
        if position + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length = int.from_bytes(data[position : position + 4], "big")
        kind = data[position + 4 : position + 8]
        end = position + 8 + length
        if end + 4 > len(data):
            raise ValueError("truncated PNG chunk")
        if kind == b"IDAT":
            image_data.append(data[position + 8 : end])
        elif kind not in PNG_IMAGE_CHUNKS:
            text = _png_chunk_text(kind, data[position + 8 : end])
            parts.append(
                kind.decode("latin-1") + ": " + text.decode("utf-8", "replace")
            )
        position = end + 4
        if kind == b"IEND":
            break
    _png_check_image_data(image_data)
    trailing = data[position:]
    if trailing:
        parts.append("trailing: " + trailing.decode("latin-1"))
    return "\n".join(parts), trailing


def read_png_metadata(data):
    """Decode every non-image PNG chunk and any bytes after IEND into one text surface."""
    return _png_decode(data)[0]


def _tree_files(tree):
    files = []
    for root, directories, names in os.walk(tree, topdown=True, followlinks=False):
        base = Path(root)
        for name in sorted(directories):
            path = base / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError(
                    f"non-directory entry in export tree: {path.relative_to(tree)}"
                )
        for name in sorted(names):
            path = base / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError(
                    f"non-regular entry in export tree: {path.relative_to(tree)}"
                )
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(tree).as_posix())


def validate_tree_inventory(tree, expected):
    actual = {path.relative_to(tree).as_posix() for path in _tree_files(tree)}
    expected = {str(path) for path in expected}
    if actual != expected:
        added, missing = sorted(actual - expected), sorted(expected - actual)
        raise ValueError(
            f"export tree inventory mismatch: added={added}, missing={missing}"
        )


def audit_tree(tree, tokens, *, pdf_reader=read_pdf):
    encoded = []
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise ValueError("scrub tokens must be non-empty strings")
        encoded.append((token, token.encode()))
    matches = defaultdict(lambda: {"count": 0, "offsets": []})
    parser_failures = []
    for path in _tree_files(tree):
        relative = path.relative_to(tree).as_posix()
        path_bytes = relative.encode()
        data = path.read_bytes()
        surface = "binary_literal" if _is_binary(data) else "text_literal"
        streams = [("path", path_bytes), (surface, data)]
        if path.suffix.lower() == ".pdf":
            try:
                decoded = pdf_reader(path)
                expected = {
                    "pdf_text",
                    "pdf_metadata",
                    "pdf_xmp",
                    "pdf_custom_metadata",
                }
                if not isinstance(decoded, dict) or set(decoded) != expected:
                    raise ValueError("PDF parser returned incomplete surfaces")
                streams.extend(sorted(decoded.items()))
            except Exception as error:  # noqa: BLE001 - injected parsers fail closed
                parser_failures.append({"file": relative, "error": str(error)})
        for stream_name, content in streams:
            for token, needle in encoded:
                count, positions = _offset_summary(content, needle)
                if count:
                    key = (relative, token, stream_name)
                    matches[key]["count"] += count
                    matches[key]["offsets"].extend(positions)
    rows = [
        {
            "file": key[0],
            "token": key[1],
            "surface": key[2],
            "count": value["count"],
            "first_offsets": value["offsets"][:3],
        }
        for key, value in sorted(matches.items())
    ]
    return {
        "scrub_pass": not rows and not parser_failures,
        "matches": rows,
        "parser_failures": parser_failures,
        "match_count": sum(row["count"] for row in rows),
    }


def _summary(audit, inventory):
    grouped = Counter()
    for row in audit["matches"]:
        grouped[(row["token"], row["surface"])] += row["count"]
    lines = [
        "# Public export dry-run audit",
        "",
        f"- Scrub pass: `{str(audit['scrub_pass']).lower()}`",
        f"- Export files: {len(inventory)}",
        f"- Export bytes: {sum(row['bytes'] for row in inventory)}",
        f"- Literal matches: {audit['match_count']}",
        f"- PDF parser failures: {len(audit['parser_failures'])}",
        "- Release ready: `false`",
        "",
        "## Top matches",
        "",
    ]
    if grouped:
        lines.extend(
            f"- `{token}` / `{surface}`: {count}"
            for (token, surface), count in grouped.most_common(30)
        )
    else:
        lines.append("- None")
    if audit["parser_failures"]:
        lines.extend(["", "## Parser failures", ""])
        lines.extend(
            f"- `{row['file']}`: {row['error']}" for row in audit["parser_failures"]
        )
    lines.extend(["", "No publishing action was performed.", ""])
    return "\n".join(lines)


def build_export_v1(
    repo, benchmark_root, frozen_source, out, policy_path, *, pdf_reader=read_pdf
):
    repo = _root(repo, "repository")
    benchmark = _root(benchmark_root, "benchmark root")
    frozen = _root(frozen_source, "frozen source")
    policy_path = Path(policy_path)
    if policy_path.is_symlink() or not policy_path.is_file():
        raise ValueError("policy must be a non-symlink file")
    policy_path = policy_path.resolve()
    policy_bytes = policy_path.read_bytes()
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    policy = json.loads(policy_bytes)
    if policy.get("schema") != SCHEMA:
        raise ValueError("unsupported export policy schema")
    if policy.get("matching", "literal_case_sensitive") != "literal_case_sensitive":
        raise ValueError("unsupported scrub matching mode")
    tokens = policy.get("tokens")
    if (
        not isinstance(tokens, list)
        or len(tokens) != TOKEN_COUNT
        or len(set(tokens)) != TOKEN_COUNT
        or any(not isinstance(token, str) or not token for token in tokens)
        or hashlib.sha256(_json_bytes(tokens)).hexdigest() != TOKEN_DIGEST
    ):
        raise ValueError("export policy scrub-token contract mismatch")

    out_path = Path(out)
    if out_path.exists() or out_path.is_symlink():
        raise ValueError("output path already exists")
    out_resolved = out_path.resolve()
    if any(out_resolved.is_relative_to(root) for root in (repo, benchmark, frozen)):
        raise ValueError("output path is nested in a source tree")

    copies = plan_export(repo, benchmark, frozen, policy)
    source_before = {
        (str(destination), str(source)): sha256(source)
        for destination, value in copies.items()
        for source, _ in value["sources"]
    }
    expected_sources = {
        (str(destination), str(source)): expected
        for destination, value in copies.items()
        for source, expected in value["sources"]
    }
    if source_before != expected_sources:
        raise ValueError("an export source differs from its planned digest")
    out_path.mkdir(parents=True)
    tree = out_path / "tree"
    tree.mkdir()
    inventory = []
    for destination in sorted(copies, key=str):
        value = copies[destination]
        source, expected, categories = (
            value["source"],
            value["sha256"],
            value["categories"],
        )
        target = tree.joinpath(*destination.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        actual = sha256(target)
        if actual != expected:
            raise ValueError(f"target copy hash mismatch: {destination}")
        inventory.append(
            {
                "path": str(destination),
                "bytes": target.stat().st_size,
                "sha256": actual,
                "source_sha256": expected,
                "categories": sorted(categories),
            }
        )

    validate_tree_inventory(tree, copies)
    audit = audit_tree(tree, policy["tokens"], pdf_reader=pdf_reader)
    validate_tree_inventory(tree, copies)
    source_after = {
        (str(destination), str(source)): sha256(source)
        for destination, value in copies.items()
        for source, _ in value["sources"]
    }
    if source_after != source_before:
        raise ValueError("a source changed during export construction")
    if policy_path.read_bytes() != policy_bytes:
        raise ValueError("export policy changed during construction")
    for row in inventory:
        if sha256(tree / row["path"]) != row["sha256"]:
            raise ValueError(f"target changed during audit: {row['path']}")

    audit_document = {
        "schema": "public-export-scrub/v1",
        **audit,
        "tokens_sha256": hashlib.sha256(_json_bytes(policy["tokens"])).hexdigest(),
    }
    _write_json(out_path / "audit.json", audit_document)
    (out_path / "audit.md").write_text(_summary(audit, inventory))
    _write_json(
        out_path / "manifest.json",
        {"schema": "public-export-manifest/v1", "files": inventory},
    )
    category_counts = Counter()
    category_bytes = Counter()
    for row in inventory:
        for category in row["categories"]:
            category_counts[category] += 1
            category_bytes[category] += row["bytes"]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "audit_complete",
        "scrub_pass": audit["scrub_pass"],
        "release_ready": False,
        "published": False,
        "policy_sha256": policy_sha256,
        "script_sha256": sha256(Path(__file__)),
        "tree": {
            "files": len(inventory),
            "bytes": sum(row["bytes"] for row in inventory),
            "categories": {
                key: {"files": category_counts[key], "bytes": category_bytes[key]}
                for key in sorted(category_counts)
            },
            "aggregate_sha256": hashlib.sha256(_json_bytes(inventory)).hexdigest(),
        },
        "audit": {
            "match_count": audit["match_count"],
            "matched_files": len({row["file"] for row in audit["matches"]}),
            "parser_failures": len(audit["parser_failures"]),
        },
        "artifacts_sha256": {
            name: sha256(out_path / name)
            for name in ("audit.json", "audit.md", "manifest.json")
        },
    }
    if policy_path.read_bytes() != policy_bytes:
        raise ValueError("export policy changed before receipt write")
    _write_json(out_path / "complete.json", receipt)
    return receipt


def _pattern(token, *, boundary=None):
    body = re.escape(token)
    if boundary == "host":
        body = rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"
    elif boundary == "word":
        body = rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])"
    return re.compile(body)


def _deny_patterns(tokens, boundary_tokens, host_tokens):
    result = []
    for token in tokens:
        result.append(
            (
                token,
                _pattern(
                    token,
                    boundary=(
                        "word"
                        if token in boundary_tokens
                        else "host"
                        if token in host_tokens
                        else None
                    ),
                ),
            )
        )
    return result


def _has_pattern(text, patterns):
    return any(pattern.search(text) for _, pattern in patterns)


def _sub_with_lines(text, pattern, replacement):
    lines = []
    for match in pattern.finditer(text):
        lines.append(text.count("\n", 0, match.start()) + 1)
        if len(lines) == 3:
            break
    rewritten, count = pattern.subn(replacement, text)
    return rewritten, count, lines


def _regex_summary(text, pattern):
    count, first = 0, []
    for match in pattern.finditer(text):
        count += 1
        if len(first) < 3:
            first.append(match.start())
    return count, first


def _rewrite_text(text, policy):
    operations = []
    for rule in policy["home_path_rules"]:
        pattern = re.compile(rule["pattern"])
        text, count, lines = _sub_with_lines(text, pattern, rule["replacement"])
        if count:
            operations.append(
                {"kind": "data_root_rewrite", "count": count, "first_lines": lines}
            )
    for source, replacement in policy["host_rewrites"].items():
        pattern = _pattern(source, boundary="host")
        text, count, lines = _sub_with_lines(text, pattern, replacement)
        if count:
            operations.append(
                {"kind": "host_rewrite", "count": count, "first_lines": lines}
            )
    return text, operations


def _json_pointer(parts):
    return "/" + "/".join(
        str(part).replace("~", "~0").replace("/", "~1") for part in parts
    )


def _contains_removed(value, patterns):
    if isinstance(value, str):
        return _has_pattern(value, patterns)
    if isinstance(value, dict):
        return any(
            _has_pattern(str(key), patterns) or _contains_removed(item, patterns)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_removed(item, patterns) for item in value)
    return False


def _prune_json(value, patterns, operations, parts=()):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            pointer = _json_pointer((*parts, key))
            if _has_pattern(str(key), patterns) or (
                not isinstance(item, (dict, list)) and _contains_removed(item, patterns)
            ):
                operations.append({"kind": "json_field_removed", "pointer": pointer})
                continue
            result[key] = _prune_json(item, patterns, operations, (*parts, key))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            if not isinstance(item, (dict, list)) and _contains_removed(item, patterns):
                operations.append(
                    {
                        "kind": "json_list_record_removed",
                        "pointer": _json_pointer((*parts, index)),
                    }
                )
                continue
            result.append(_prune_json(item, patterns, operations, (*parts, index)))
        return result
    return value


def _validate_redactions(policy):
    """Check the path-scoped rewrite rules before any file is read."""
    seen = set()
    for rule in policy.get("redactions", []):
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen:
            raise ValueError("redaction rule needs a unique string id")
        seen.add(rule_id)
        paths = rule.get("paths")
        if (
            rule.get("kind") not in TEXT_RULE_KINDS | JSON_RULE_KINDS
            or not isinstance(paths, list)
            or not paths
            or not set(rule.get("flags", "")) <= set(RULE_FLAGS)
            or not isinstance(rule.get("where", {}), dict)
            or not isinstance(rule.get("keep_values", []), list)
            or ("key_pattern" in rule and not rule.get("each"))
            or (
                {"where", "keep_values", "key_pattern"} & set(rule)
                and rule.get("kind") != "json_fields"
            )
            or (
                rule.get("kind") == "json_drop_keys"
                and not (
                    rule.get("each")
                    and isinstance(rule.get("container"), str)
                    and isinstance(rule.get("keys"), list)
                    and rule["keys"]
                    and all(isinstance(key, str) for key in rule["keys"])
                )
            )
        ):
            raise ValueError(f"invalid redaction rule: {rule_id}")


def _redactions_for(relative, policy):
    return [
        rule
        for rule in policy.get("redactions", [])
        if any(fnmatch.fnmatchcase(str(relative), glob) for glob in rule["paths"])
    ]


def _rule_regex(rule):
    flags = 0
    for letter in rule.get("flags", ""):
        flags |= RULE_FLAGS[letter]
    source = re.escape(rule["text"]) if "text" in rule else rule["pattern"]
    return re.compile(source, flags)


def _redact_table_columns(rule, text, removed=None):
    """Replace named columns in every table whose header line matches.

    What the replaced cells held is noted in `removed` when it is given.
    """
    header = re.compile(rule["header_pattern"])
    lines = text.splitlines(keepends=True)
    count, index = 0, 0
    while index < len(lines):
        if not header.search(lines[index]):
            index += 1
            continue
        names = [cell.strip() for cell in lines[index].rstrip("\r\n").split("|")[1:-1]]
        targets = {names.index(name) + 1 for name in rule["columns"]}
        index += 1
        while index < len(lines) and lines[index].startswith("|"):
            body = lines[index].rstrip("\r\n")
            ending = lines[index][len(body) :]
            cells = body.split("|")
            if len(cells) != len(names) + 2:
                raise ValueError(f"table row shape mismatch: {rule['id']}")
            if not all(re.fullmatch(r"\s*:?-+:?\s*", cells[i]) for i in targets):
                for i in targets:
                    if removed is not None:
                        label = names[i - 1] + " "
                        removed.note(
                            label + cells[i],
                            len(label),
                            len(label) + len(cells[i]),
                            rule["replacement"],
                        )
                    cells[i] = " " + rule["replacement"] + " "
                count += len(targets)
                lines[index] = "|".join(cells) + ending
            index += 1
    return "".join(lines), count


REDACTED_NUMBER = re.compile(r"(?<![\w.,])\d+(?:,\d{3})*(?![\w]|\.\d|,\d)")
COUNT_NOUNS = (
    r"(?:files?|members?|hash(?:es)?|digests?|rows?|images?|entries|entry|"
    r"positions?|frames?|archives?|photos?|boards?)"
)


def _numbers_in(text):
    return {found.replace(",", "") for found in REDACTED_NUMBER.findall(text)}


# The words next to a number: one before and one after, across spaces, markup
# characters and at most one line break, and across the other end of a range.
NEIGHBOUR_GAP = r"(?:[ \t~*_{}$()\[\]%,.;:=\"'-]|\n(?!\n))*"
NEIGHBOUR_NUMBER = r"\d+(?:,\d{3})*"
NEIGHBOUR_LINK = r"(?:(?:to|and|or|through|until)\b" + NEIGHBOUR_GAP + r")"
NEIGHBOUR_WORD = r"([A-Za-z][A-Za-z-]+[A-Za-z])"
WORD_BEFORE = re.compile(
    NEIGHBOUR_WORD
    + NEIGHBOUR_GAP
    + rf"(?:{NEIGHBOUR_NUMBER}{NEIGHBOUR_GAP}{NEIGHBOUR_LINK}?)?\Z"
)
WORD_AFTER = re.compile(
    rf"\A{NEIGHBOUR_GAP}{NEIGHBOUR_LINK}?(?:{NEIGHBOUR_NUMBER}{NEIGHBOUR_GAP})?"
    + NEIGHBOUR_WORD
)
NEIGHBOUR_STOPWORDS = frozenset(
    {"and", "are", "but", "for", "from", "has", "have", "into", "not", "only"}
    | {"that", "the", "this", "was", "were", "which", "with", "than", "then"}
    | {"through", "until", "about", "also", "each", "all", "both", "its", "per"}
)
# Marks a tracked word that came from a JSON key, which only the word before matches.
KEY_WORD = "^"
# Numbers of one digit are not tracked: they are too common to name a value.
NEIGHBOUR_MIN_DIGITS = 2
# A number that is part of a name is not a count: a zero-padded id, a file stem,
# or a number after a word and a hyphen.
NAME_NUMBER = re.compile(r"[A-Za-z]-\Z")
NAME_FOLLOW = re.compile(r"\A\.[A-Za-z]")


def _word_pair(text, start, end):
    """The lower-case words before and after text[start:end]; None where there is none."""
    before = WORD_BEFORE.search(text[max(0, start - 96) : start])
    after = WORD_AFTER.match(text[end : end + 96])
    return tuple(
        found.group(1).lower()
        if found and found.group(1).lower() not in NEIGHBOUR_STOPWORDS
        else None
        for found in (before, after)
    )


def _neighbours(text, start, end):
    """The lower-case words next to text[start:end]: the one before and the one after."""
    return {word for word in _word_pair(text, start, end) if word}


class Removed:
    """What the redaction rules took out of the text.

    `numbers` maps each removed number to the words that stood next to it in the
    source. `hexes` holds the hex values of 32 digits or more. Both only grow,
    and neither is ever written anywhere.
    """

    def __init__(self):
        self.numbers, self.hexes = {}, set()

    def note(self, string, start, end, kept=""):
        """Record what string[start:end] held, except what `kept` still shows.

        Returns the set of numbers it recorded.
        """
        span = string[start:end]
        stays, gone = _numbers_in(kept), set()
        for found in REDACTED_NUMBER.finditer(span):
            number = found.group().replace(",", "")
            first, last = start + found.start(), start + found.end()
            if number not in stays:
                gone.add(number)
                words = self.numbers.setdefault(number, set())
                if not (
                    (number.startswith("0") and len(number) > 1)
                    or NAME_NUMBER.search(string[max(0, first - 2) : first])
                    or NAME_FOLLOW.match(string[last : last + 2])
                ):
                    words.update(_neighbours(string, first, last))
        self.hexes.update(
            run.lower()
            for run in HEX_RUN.findall(span)
            if len(run) >= 32 and run.lower() not in kept.lower()
        )
        return gone

    def note_json(self, value, key=""):
        """Record the integers and hex strings in a removed JSON value."""
        if isinstance(value, dict):
            for name, item in value.items():
                self.note_json(item, name)
        elif isinstance(value, list):
            for item in value:
                self.note_json(item, key)
        elif isinstance(value, int) and not isinstance(value, bool):
            # A key stands before its value, so it counts only as the word before.
            words = {
                KEY_WORD + word
                for word in re.findall(r"[a-z]{3,}", str(key).lower())
                if word not in NEIGHBOUR_STOPWORDS
            }
            self.numbers.setdefault(str(value), set()).update(words)
        elif isinstance(value, str):
            self.note(value, 0, len(value))

    def merge(self, other):
        for number, words in other.numbers.items():
            self.numbers.setdefault(number, set()).update(words)
        self.hexes |= other.hexes

    def tracked(self):
        """The numbers that gate shipped text, with the words each had."""
        return {
            number: words
            for number, words in self.numbers.items()
            if words and len(number) >= NEIGHBOUR_MIN_DIGITS
        }


def _remaining_counts(label, text, numbers, rules):
    """Refuse a number that a rule redacted when it stays elsewhere as a count.

    A rule that removes the count in "binds exactly N extracted items" leaves it
    in "all N extracted item names". A match stays on one line. The finding
    names file and line, not the number.
    """
    for number in sorted(numbers):
        # A zero-padded number is an identifier, not a count: only its padded form counts.
        spellings = (
            (number,)
            if number.startswith("0") and len(number) > 1
            else (f"{int(number):,}", number)
        )
        forms = "|".join(re.escape(form) for form in dict.fromkeys(spellings))
        value = rf"(?<![\w.,])(?:{forms})(?![\w]|\.\d|,\d)"
        found = re.search(
            rf"{value}(?:[ \t]+[A-Za-z][\w-]*){{0,3}}[ \t]+{COUNT_NOUNS}\b"
            rf"|\b{COUNT_NOUNS}\b[ \t]*(?:[:=(]|of|is|are)?[ \t]*{value}",
            text,
            re.IGNORECASE,
        )
        if found:
            line = text.count("\n", 0, found.start()) + 1
            raise ValueError(
                f"redacted number remains as a count: {label}:{line} "
                f"(rules {', '.join(sorted(rules))})"
            )


def _apply_text_rules(rules, text, operations, label="", removals=None):
    removed, sources = Removed(), set()
    for rule in rules:
        kind = rule["kind"]
        if kind == "replace":
            replacement = rule["replacement"]
            pattern = _rule_regex(rule)

            def swap(match, rule=rule, replacement=replacement):
                new = replacement if "text" in rule else match.expand(replacement)
                if removed.note(match.string, match.start(), match.end(), new):
                    sources.add(rule["id"])
                return new

            text, count = pattern.subn(swap, text)
        elif kind == "drop_lines":
            pattern = _rule_regex(rule)
            lines = text.splitlines(keepends=True)
            kept = []
            for line in lines:
                if not pattern.search(line):
                    kept.append(line)
                elif removed.note(line, 0, len(line)):
                    sources.add(rule["id"])
            count = len(lines) - len(kept)
            text = "".join(kept)
        elif kind == "md_table_columns":
            text, count = _redact_table_columns(rule, text, removed)
            if count:
                sources.add(rule["id"])
        else:
            continue
        if count:
            operations.append(
                {"kind": "path_scoped_rewrite", "rule": rule["id"], "count": count}
            )
    if removed.numbers:
        _remaining_counts(label, text, removed.numbers, sources)
    if removals is not None:
        removals.merge(removed)
    return text


def _json_target(value, pointer):
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("redaction pointer must be empty or start with a slash")
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def _redact_json_entries(rule, target, removed=None):
    entries = [(re.compile(entry["key_pattern"]), entry) for entry in rule["entries"]]
    result, count = {}, 0
    for key, item in target.items():
        for pattern, entry in entries:
            found = pattern.search(key)
            if found:
                if "key_replacement" in entry:
                    key = (
                        key[: found.start()]
                        + entry["key_replacement"]
                        + key[found.end() :]
                    )
                if removed is not None:
                    removed.note_json(item, key)
                item = entry["value"]
                count += 1
                break
        if key in result and result[key] != item:
            raise ValueError(f"redacted key collision: {rule['id']}")
        result[key] = item
    target.clear()
    target.update(result)
    return count


def _redact_json_fields(rule, target, removed=None):
    """Set the named fields of the target, or of each of its members.

    `where` limits the rule to members whose fields equal the given values,
    `key_pattern` to members whose key matches, and `keep_values` leaves a
    field that holds one of the listed values.
    """
    if not rule.get("each"):
        pairs = [(None, target)]
    elif isinstance(target, list):
        pairs = list(enumerate(target))
    else:
        pairs = list(target.items())
    key_pattern = re.compile(rule["key_pattern"]) if "key_pattern" in rule else None
    keep = rule.get("keep_values", [])
    count = 0
    for key, item in pairs:
        if not isinstance(item, dict):
            continue
        if key_pattern is not None and not key_pattern.search(str(key)):
            continue
        if any(
            field not in item or item[field] != expected
            for field, expected in rule.get("where", {}).items()
        ):
            continue
        for field in rule["fields"]:
            if (
                field in item
                and item[field] != rule["value"]
                and item[field] not in keep
            ):
                if removed is not None:
                    removed.note_json(item[field], field)
                item[field] = rule["value"]
                count += 1
    return count


def _redact_json_drop_keys(rule, target, removed=None):
    """Remove the named keys from the `container` member of each member of the target.

    A member whose container would be left empty is refused: the dropped key was
    its only entry, so the row would still say that something matched.
    """
    members = target if isinstance(target, list) else list(target.values())
    count = 0
    for item in members:
        container = item.get(rule["container"]) if isinstance(item, dict) else None
        if not isinstance(container, dict):
            continue
        dropped = [key for key in rule["keys"] if key in container]
        if not dropped:
            continue
        if len(dropped) == len(container):
            raise ValueError(f"redaction would empty a row: {rule['id']}")
        for key in dropped:
            if removed is not None:
                removed.note_json(container[key], key)
            del container[key]
        count += len(dropped)
    return count


def _apply_json_rules(rules, value, operations, removals=None):
    changed = False
    for rule in rules:
        if rule["kind"] not in JSON_RULE_KINDS:
            continue
        target = _json_target(value, rule["pointer"])
        if rule["kind"] == "json_entries":
            count = (
                _redact_json_entries(rule, target, removals)
                if isinstance(target, dict)
                else 0
            )
        elif rule["kind"] == "json_drop_keys":
            count = (
                _redact_json_drop_keys(rule, target, removals)
                if isinstance(target, (list, dict))
                else 0
            )
        else:
            count = (
                _redact_json_fields(rule, target, removals) if target is not None else 0
            )
        if count:
            operations.append(
                {"kind": "path_scoped_rewrite", "rule": rule["id"], "count": count}
            )
            changed = True
    return changed


def _transform_text(relative, data, policy, removals=None):
    try:
        original = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"text input is not UTF-8: {relative}") from error
    text, operations = _rewrite_text(original, policy)
    text, named, named_lines = name_date_fields(relative, text)
    if named:
        operations.append(
            {"kind": "name_date_rewrite", "count": named, "first_lines": named_lines}
        )
    text, restated, restated_lines = utc_timestamps(text)
    if restated:
        operations.append(
            {
                "kind": "utc_timestamp_rewrite",
                "count": restated,
                "first_lines": restated_lines,
            }
        )
    rules = _redactions_for(relative, policy)
    if relative.suffix.lower() != ".json" and any(
        rule["kind"] in JSON_RULE_KINDS for rule in rules
    ):
        raise ValueError(f"JSON redaction rule matches a non-JSON file: {relative}")
    text = _apply_text_rules(rules, text, operations, str(relative), removals)
    remove_patterns = _deny_patterns(
        policy["remove_text_tokens"], policy["word_boundary_tokens"], set()
    )
    suffix = relative.suffix.lower()
    if suffix == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON export input: {relative}") from error
        redacted = _apply_json_rules(rules, value, operations, removals)
        if _contains_removed(value, remove_patterns):
            value = _prune_json(value, remove_patterns, operations)
            redacted = True
        if redacted:
            text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    elif suffix == ".jsonl":
        rows = []
        dropped = []
        allowed = any(
            fnmatch.fnmatchcase(str(relative), pattern)
            for pattern in policy.get("jsonl_drop_record_patterns", [])
        )
        scene_manifest = relative.is_relative_to(PurePosixPath("benchmark_v2"))
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL export input: {relative}:{line_number}"
                ) from error
            if _contains_removed(value, remove_patterns):
                if scene_manifest:
                    raise ValueError(
                        f"benchmark scene row requires removal: {relative}:{line_number}"
                    )
                if not allowed:
                    return (
                        None,
                        operations,
                        {
                            "reason": "jsonl_record_removal_not_authorized",
                            "line": line_number,
                        },
                    )
                dropped.append(line_number)
                continue
            rows.append(json.dumps(value, sort_keys=True, separators=(",", ":")))
        if dropped:
            operations.append(
                {
                    "kind": "jsonl_records_removed",
                    "count": len(dropped),
                    "first_lines": dropped[:3],
                }
            )
            text = "\n".join(rows) + ("\n" if rows else "")
    else:
        kept, dropped = [], []
        for line_number, line in enumerate(text.splitlines(keepends=True), 1):
            if _has_pattern(line, remove_patterns):
                dropped.append(line_number)
            else:
                kept.append(line)
        if dropped:
            operations.append(
                {
                    "kind": "text_lines_removed",
                    "count": len(dropped),
                    "first_lines": dropped[:3],
                }
            )
            text = "".join(kept)
    if suffix == ".py":
        try:
            ast.parse(text, filename=str(relative))
        except SyntaxError as error:
            return (
                None,
                operations,
                {
                    "reason": "python_syntax_invalid_after_transform",
                    "line": error.lineno,
                },
            )
    return text.encode(), operations, None


def _drop_path(relative, policy):
    value = str(relative)
    if any(
        fnmatch.fnmatchcase(value, pattern) for pattern in policy["drop_file_patterns"]
    ):
        return "policy_drop_pattern"
    patterns = _deny_patterns(
        policy["remove_text_tokens"], policy["word_boundary_tokens"], set()
    )
    if _has_pattern(relative.name, patterns):
        return "forbidden_filename"
    return None


def _mapped_path(relative, policy):
    value, _ = _rewrite_text(str(relative), policy)
    return _relative(value, label="mapped export")


def _router_view(source, expected):
    if sha256(source) != expected:
        raise ValueError("neutral table source hash mismatch")
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    systems = [
        ("shipped ensemble", "full"),
        ("raw member", "raw"),
        ("warp-high member", "warp_high"),
        ("warp-low member", "warp_low"),
    ]
    if len(rows) != 18 or len({row["split"] for row in rows}) != 18:
        raise ValueError("neutral table requires exactly 18 distinct numeric rows")
    output = []
    for row in rows:
        item = {"split": row["split"], "n": int(row["n"])}
        for label, prefix in systems:
            item[label + " exact pct"] = float(row[prefix + "_exact_pct"])
            item[label + " per-square pct"] = float(row[prefix + "_square_pct"])
        output.append(item)
    header = ["split", "n"] + [
        label + suffix
        for label, _ in systems
        for suffix in (" exact pct", " per-square pct")
    ]
    csv_lines = [",".join(header)]
    for row in output:
        csv_lines.append(",".join(str(row[key]) for key in header))
    md = [
        "# Historical pipeline comparison",
        "",
        "Values are preserved from the receipt-bound numeric figure input.",
        "System composition and routing differ, so differences are descriptive pipeline comparisons rather than isolated member effects.",
        "",
        "| Split | N | Shipped ensemble exact / square | Raw member exact / square | Warp-high member exact / square | Warp-low member exact / square |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in output:
        values = [
            f"{row[label + ' exact pct']} / {row[label + ' per-square pct']}"
            for label, _ in systems
        ]
        md.append(f"| {row['split']} | {row['n']} | " + " | ".join(values) + " |")
    return ("\n".join(csv_lines) + "\n").encode(), ("\n".join(md) + "\n").encode()


def decoded_json_strings(path, text):
    """Expose escaped JSON keys/values; ordinary unescaped text was already scanned."""
    if "\\" not in text or path.suffix.lower() not in {".json", ".jsonl"}:
        return

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in privacy audit")
            result[key] = value
        return result

    if path.suffix.lower() == ".json":
        yield from strings(json.loads(text, object_pairs_hook=unique_object))
    else:
        for line in text.splitlines():
            if line.strip():
                yield from strings(json.loads(line, object_pairs_hook=unique_object))


def _fingerprint_pattern(value):
    """Return lowercase literals for a fast prefilter and the exact-match regex."""
    if re.fullmatch(r"[0-9a-fA-F]{32,}", value):
        pattern = rf"(?<![0-9A-Fa-f]){value}(?![0-9A-Fa-f])"
        return [value.lower()], re.compile(pattern, re.IGNORECASE)
    if value.isdigit():
        groups = [value[max(0, end - 3) : end] for end in range(len(value), 0, -3)]
        forms = list(
            dict.fromkeys([value, ",".join(groups[::-1]), "_".join(groups[::-1])])
        )
        pattern = r"(?<![0-9])(?:" + "|".join(forms) + r")(?![0-9])"
        return forms, re.compile(pattern)
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
    return [value.lower()], re.compile(pattern)


def private_fingerprint_matchers(policy, shipped_sha256=()):
    """Compile the unshipped fingerprint list into (label, literals, pattern) rows.

    A group may exempt values that equal the hash of a file the tree ships on
    purpose. Labels carry the group and index, never the value.
    """
    matchers = []
    for group in policy.get("private_fingerprints", {}).get("groups", []):
        for index, value in enumerate(group["values"]):
            if group.get("exempt_shipped_sha256") and value.lower() in shipped_sha256:
                continue
            needles, pattern = _fingerprint_pattern(value)
            matchers.append(
                (f"private_fingerprint:{group['name']}#{index}", needles, pattern)
            )
    return matchers


def check_fingerprint_contract(policy):
    """Refuse a policy whose private-fingerprint list is not the approved one."""
    if "private_fingerprints" not in policy:
        return
    groups = policy["private_fingerprints"]["groups"]
    values = [value for group in groups for value in group["values"]]
    if (
        len(values) != FINGERPRINT_COUNT
        or hashlib.sha256(_json_bytes(policy["private_fingerprints"])).hexdigest()
        != FINGERPRINT_DIGEST
    ):
        raise ValueError("export policy private-fingerprint contract mismatch")


def _audit_tree_v2(tree, policy, vendor_paths):
    files = _tree_files(tree)
    shipped = {path.relative_to(tree).as_posix(): sha256(path) for path in files}
    fingerprints = private_fingerprint_matchers(policy, set(shipped.values()))
    member_hashes = {
        value.lower()
        for group in policy.get("private_fingerprints", {}).get("groups", [])
        if group.get("exempt_shipped_sha256")
        for value in group["values"]
    }
    patterns = _deny_patterns(
        policy["tokens"],
        set(policy["word_boundary_tokens"]),
        set(policy["host_rewrites"]),
    )
    patterns.extend(
        (
            token,
            re.compile(
                r"\s+".join(re.escape(part) for part in token.split()), re.IGNORECASE
            ),
        )
        for token in policy.get("identifier_tokens", [])
    )
    patterns.extend(
        (name, re.compile(pattern))
        for name, pattern in policy.get("path_scrub_patterns", {}).items()
    )
    matches = []
    parser_failures = []
    for path in files:
        relative = path.relative_to(tree).as_posix()
        streams = [("path", relative)]
        if PurePosixPath(relative) not in vendor_paths:
            data = path.read_bytes()
            if path.suffix.lower() == ".pdf":
                try:
                    decoded = read_pdf(path)
                    streams.extend(
                        (surface, value.decode("utf-8", errors="replace"))
                        for surface, value in sorted(decoded.items())
                    )
                    streams.extend(sorted(read_pdf_structure(path).items()))
                except Exception as error:  # noqa: BLE001 - parser failure is an audit result
                    parser_failures.append({"file": relative, "error": str(error)})
            elif not _is_binary(data):
                text = data.decode("utf-8")
                streams.append(("text", text))
                try:
                    streams.extend(
                        ("decoded_json", value)
                        for value in decoded_json_strings(path, text)
                    )
                except (ValueError, TypeError) as error:
                    parser_failures.append({"file": relative, "error": str(error)})
            else:
                streams.append(("binary", data.decode("latin-1")))
                if path.suffix.lower() == ".png":
                    try:
                        text, trailing = _png_decode(data)
                        streams.append(("png_metadata", text))
                        if trailing:
                            raise ValueError("PNG has bytes after the IEND chunk")
                    except (ValueError, zlib.error) as error:
                        parser_failures.append({"file": relative, "error": str(error)})
        for surface, text in streams:
            found = []
            if surface != "binary":
                found.extend((token, pattern) for token, pattern in patterns)
            lowered = text.lower() if fingerprints else ""
            found.extend(
                (label, pattern)
                for label, needles, pattern in fingerprints
                if any(needle in lowered for needle in needles)
            )
            for token, pattern in found:
                count, offsets = _regex_summary(text, pattern)
                if count:
                    matches.append(
                        {
                            "file": relative,
                            "token": token,
                            "surface": surface,
                            "count": count,
                            "first_offsets": offsets,
                        }
                    )
    return {
        "scrub_pass": not matches and not parser_failures,
        "matches": matches,
        "parser_failures": parser_failures,
        "match_count": sum(row["count"] for row in matches),
        "private_fingerprint_values": len(fingerprints),
        "private_member_identical_paths": sorted(
            name for name, digest in shipped.items() if digest in member_hashes
        ),
    }


def digest_forms(data):
    """Every digest form a record may quote for one byte string."""
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "sha1": hashlib.sha1(data, usedforsecurity=False).hexdigest(),
        "git_blob_sha1": hashlib.sha1(
            b"blob %d\0" % len(data) + data, usedforsecurity=False
        ).hexdigest(),
        "md5": hashlib.md5(data, usedforsecurity=False).hexdigest(),
    }


def _hex_runs(text):
    """Distinct lowercase hex runs of 12 or more characters."""
    return frozenset(run.lower() for run in HEX_RUN.findall(text))


def _windows(values):
    """Every 12-character window of every value."""
    return {
        value[start : start + 12]
        for value in values
        for start in range(len(value) - 11)
    }


def withheld_digests(rows, seeds):
    """Close the set of digests this export may not quote.

    The set starts with the seeds and every digest form of the source of each
    rewritten row. A redactable row that quotes a member is rewritten by that
    redaction, so its own source digests join the set, until nothing is added.
    A quote is a hex run that holds any 12 consecutive characters of a member.
    Returns the set, the indexes of the quoting rows and the number of rounds.
    """
    fresh = set()
    for value in seeds:
        if not re.fullmatch(r"[0-9a-f]{32,128}", value):
            raise ValueError("withheld digest seed is not a lowercase hex digest")
        fresh.add(value)
    for row in rows:
        if row["rewritten"]:
            fresh.update(row["forms"].values())
    by_run = {}
    for index, row in enumerate(rows):
        if row["redactable"]:
            for run in row["runs"]:
                by_run.setdefault(run, []).append(index)
    withheld, quoting, rounds = set(), set(), 0
    while fresh:
        rounds += 1
        withheld |= fresh
        windows = _windows(fresh)
        fresh = set()
        for run, owners in by_run.items():
            if all(index in quoting for index in owners):
                continue
            if not any(
                run[start : start + 12] in windows for start in range(len(run) - 11)
            ):
                continue
            for index in owners:
                if index not in quoting:
                    quoting.add(index)
                    if not rows[index]["rewritten"]:
                        fresh.update(rows[index]["forms"].values())
        fresh -= withheld
    return withheld, quoting, rounds


def _quotes_withheld(run, prefixes, memo):
    hit = memo.get(run)
    if hit is None:
        hit = memo[run] = any(
            run[start : start + 12] in prefixes for start in range(len(run) - 11)
        )
    return hit


def redact_withheld(text, prefixes, memo):
    """Replace every hex run that quotes a withheld digest; return text and count."""
    count = 0

    def replace(match):
        nonlocal count
        if not _quotes_withheld(match[0].lower(), prefixes, memo):
            return match[0]
        count += 1
        return WITHHELD_DIGEST

    return HEX_RUN.sub(replace, text), count


def withheld_only(source, shipped):
    """True when `shipped` is `source` with nothing changed but withheld digests."""
    pieces = shipped.split(WITHHELD_DIGEST.encode())
    pattern = HEX_RUN.pattern.encode().join(re.escape(piece) for piece in pieces)
    return re.fullmatch(pattern, source, re.DOTALL) is not None


def _field(row, path):
    for part in path:
        row = row.get(part) if isinstance(row, dict) else None
    return row


def _with_field(row, path, value):
    if len(path) == 1:
        return {**row, path[0]: value}
    return {**row, path[0]: _with_field(row[path[0]], path[1:], value)}


def inventory_changes(relative, source, shipped):
    """Count the rows of a shipped row inventory and those that differ from the frozen file.

    The rows must match the frozen rows in number and order. A row may differ in
    the digest fields of `INVENTORY_DIGESTS` and in the `source_sha256` entries
    that pin a frozen manifest (a name ending `.jsonl`), and in each only by
    reading the withheld marker in place of the digest. A manifest pin reads the
    marker when the export rewrote that manifest, which gives the rewritten file
    another digest. Every other byte of the line must be frozen.
    """
    before, after = source.splitlines(), shipped.splitlines()
    if len(before) != len(after):
        raise ValueError(
            f"row inventory row count differs from the frozen file: {relative}"
        )
    marker = WITHHELD_DIGEST.encode()
    changed = dict.fromkeys([*INVENTORY_DIGESTS, "manifest"], 0)
    for number, (old_line, new_line) in enumerate(zip(before, after), 1):
        if old_line == new_line:
            continue
        old, new = json.loads(old_line), json.loads(new_line)
        expected, expected_line = old, old_line
        pins = _field(old, LAUNCHER_DIGEST[:1])
        paths = [
            *((kind, path) for kind, path in INVENTORY_DIGESTS.items()),
            *(
                ("manifest", (LAUNCHER_DIGEST[0], name))
                for name in (pins if isinstance(pins, dict) else ())
                if str(name).endswith(".jsonl")
            ),
        ]
        counted = set()
        for kind, path in paths:
            digest = _field(old, path)
            if (
                isinstance(digest, str)
                and HEX_RUN.fullmatch(digest)
                and _field(new, path) == WITHHELD_DIGEST
            ):
                expected = _with_field(expected, path, WITHHELD_DIGEST)
                expected_line = expected_line.replace(digest.encode(), marker)
                counted.add(kind)
        if new != expected or new_line != expected_line:
            raise ValueError(
                f"row inventory differs outside the withheld digests: {relative}:{number}"
            )
        for kind in counted:
            changed[kind] += 1
    return {
        "rows": len(after),
        **{f"{kind}_digest_withheld": count for kind, count in changed.items()},
    }


def _check_syntax(relative, text, original=None):
    """A digest redaction must leave Python, JSON and delimited files parseable.

    A CSV or TSV file must also keep the field count of every row of `original`.
    """

    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON key after digest redaction")
        return result

    suffix = relative.suffix.lower()
    try:
        if suffix == ".py":
            ast.parse(text, filename=str(relative))
        elif suffix == ".json":
            json.loads(text, object_pairs_hook=unique_object)
        elif suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    json.loads(line, object_pairs_hook=unique_object)
        elif suffix in {".csv", ".tsv"}:
            delimiter = "," if suffix == ".csv" else "\t"

            def widths(value):
                return [
                    len(fields)
                    for fields in csv.reader(
                        io.StringIO(value, newline=""), delimiter=delimiter
                    )
                ]

            if original is not None and widths(text) != widths(original):
                raise ValueError("delimited row changed shape after digest redaction")
    except (SyntaxError, ValueError, csv.Error) as error:
        raise ValueError(f"digest redaction broke the syntax of {relative}") from error


def _suffix_of(name):
    match = re.search(r"(\.[A-Za-z0-9]{1,11})$", name.strip())
    return match[1].lower() if match else ""


def digest_groups(relative, text, pools=(), pattern=DIGEST64):
    """Group every 64-hex string of one file by where it sits.

    Returns {(key, suffix, pooled): {lowercase value, ...}}. In JSON a key is
    the path of dictionary keys joined by `>`, with a numeric key read as `N`, a list
    level as `[]`, a digest used as a dictionary key as `<key>` and a digest
    inside a longer string with a trailing `~`. In a CSV or TSV file it is
    `col:` and the column name. In any other text it is `<text>`. `suffix` is
    the extension of the file the digest is said to cover: the key itself when
    it reads like a path, else the name field of the same object. `pooled` is
    true when the key or the object names a private pool. `pattern` picks the
    value width and is DIGEST64 for the rule.
    """
    groups = defaultdict(set)

    pool_pattern = pool_name_pattern(pools) if pools else None

    def named(value):
        return pool_pattern is not None and pool_pattern.search(value) is not None

    def scan(value, key, suffix, pooled):
        for match in pattern.finditer(value):
            whole = match.span() == (0, len(value))
            groups[(key if whole else key + "~", suffix, pooled)].add(match[0].lower())

    def walk(value, trail, pooled):
        if isinstance(value, dict):
            texts = [item for item in value.values() if isinstance(item, str)]
            pooled = pooled or any(named(item) for item in texts)
            sibling = next(
                (value[n] for n in NAME_FIELDS if isinstance(value.get(n), str)), ""
            )
            for key, item in value.items():
                part = DIGEST64.sub("<key>", key.replace(">", "?"))
                part = "N" if part.isdigit() else part
                named_file = "/" in key or _suffix_of(key) in FILE_SUFFIXES
                suffix = _suffix_of(key if named_file else sibling)
                here = pooled or named(key)
                path = ">".join([*trail, part])
                scan(key, path, suffix, here)
                if isinstance(item, str):
                    scan(item, path, suffix, here)
                else:
                    walk(item, [*trail, part], here)
        elif isinstance(value, list):
            path = ">".join([*trail, "[]"])
            for item in value:
                if isinstance(item, str):
                    scan(item, path, "", pooled)
                else:
                    walk(item, [*trail, "[]"], pooled)
        elif isinstance(value, str):
            scan(value, ">".join(trail) or "<root>", "", pooled)

    suffix = relative.suffix.lower()
    try:
        if suffix == ".json":
            walk(json.loads(text), [], False)
        elif suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    walk(json.loads(line), [], False)
        elif suffix in {".csv", ".tsv"}:
            delimiter = "," if suffix == ".csv" else "\t"
            table = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
            for row in table[1:]:
                pooled = any(named(cell) for cell in row)
                for index, cell in enumerate(row):
                    header = table[0][index] if index < len(table[0]) else "?"
                    scan(cell, "col:" + header, "", pooled)
        else:
            for match in pattern.finditer(text):
                groups[("<text>", "", False)].add(match[0].lower())
    except (ValueError, csv.Error) as error:
        raise ValueError(f"digest rule could not read {relative}") from error
    return groups


ODD_DIGESTS = {
    width: re.compile(rf"(?<![0-9A-Fa-f])[0-9A-Fa-f]{{{width}}}(?![0-9A-Fa-f])")
    for width in (32, 40, 128)
}


def odd_width_census(entries):
    """Count the 32, 40 and 128-hex values of the staged text by key pattern.

    The digest rule judges 64-hex values only. This counts the other widths
    where the rule does not look, so the private audit shows the size of that
    gap. Returns {width: {key pattern: number of distinct values}}.
    """
    seen = {width: defaultdict(set) for width in ODD_DIGESTS}
    for path, data, _ in entries:
        if data is None:
            continue
        text = data.decode("utf-8", errors="replace")
        for width, pattern in ODD_DIGESTS.items():
            groups = digest_groups(PurePosixPath(path), text, pattern=pattern)
            for (key, _, _), found in groups.items():
                seen[width][key] |= found
    return {
        str(width): {key: len(found) for key, found in sorted(keys.items())}
        for width, keys in seen.items()
    }


def _class_ok(item):
    """A public class states what its digests cover and which files may hash to them."""
    preimages = item.get("preimages") if isinstance(item, dict) else None
    return (
        isinstance(item, dict)
        and set(item) == {"description", "preimages"}
        and isinstance(item["description"], str)
        and len(item["description"]) >= 12
        and isinstance(preimages, list)
        and all(
            isinstance(name, str) and re.fullmatch(r"\.[a-z0-9]{1,11}", name)
            for name in preimages
        )
    )


def _validate_digest_rule(rule):
    """Check the digest classes before any file is read.

    A key pattern is `>`-joined components; a first component `**` lets any
    number of leading components precede the rest.
    """
    classes = rule.get("classes") if isinstance(rule, dict) else None
    if (
        not isinstance(rule, dict)
        or set(rule) != DIGEST_RULE_KEYS
        or not isinstance(classes, dict)
        or not classes
        or "withheld" in classes
        or not all(_class_ok(item) for item in classes.values())
    ):
        raise ValueError("invalid digest rule")

    def entry_ok(entry, keys):
        return (
            isinstance(entry, dict)
            and set(entry) <= keys
            and isinstance(entry.get("reason"), str)
            and len(entry["reason"]) >= 12
        )

    for kind in ("public", "withheld"):
        if not isinstance(rule[kind], list):
            raise TypeError(f"invalid digest rule: {kind}")
        keys = {"key", "reason", "files"} | ({"class"} if kind == "public" else set())
        for entry in rule[kind]:
            if (
                not entry_ok(entry, keys)
                or not isinstance(entry.get("key"), str)
                or not entry["key"]
                or not isinstance(entry.get("files", []), list)
                or (kind == "public" and entry.get("class") not in classes)
            ):
                raise ValueError(f"invalid digest rule entry: {kind}")
    seen = set()
    for entry in rule["prose"]:
        paths = entry.get("paths") if isinstance(entry, dict) else None
        pins = entry.get("digests") if isinstance(entry, dict) else None
        # A withheld prose path needs no pin: every digest in it is redacted,
        # so a swapped digest cannot pass. A pin would also be circular for the
        # exporter, whose source quotes the digest of the policy that holds it.
        pinned = isinstance(entry, dict) and entry.get("class") != "withheld"
        if (
            not entry_ok(entry, {"class", "reason", "paths", *(["digests"] * pinned)})
            or entry.get("class") not in {*classes, "withheld"}
            or not isinstance(paths, dict)
            or not paths
            or not all(isinstance(n, int) and n > 0 for n in paths.values())
            or seen & set(paths)
            or (pinned and not isinstance(pins, dict))
            or (pinned and set(pins) != set(paths))
            or (
                pinned
                and not all(
                    isinstance(n, str) and PIN.fullmatch(n) for n in pins.values()
                )
            )
        ):
            raise ValueError("invalid digest rule entry: prose")
        seen |= set(paths)


def prose_digest_pin(path, digests):
    """The sha256 over a prose path and its sorted digests.

    The rule lists one pin per public prose path next to the count, so swapping
    one digest of a prose file for another of the same number fails the export.
    """
    return hashlib.sha256("\n".join([path, *sorted(digests)]).encode()).hexdigest()


def digest_context(policy):
    """Return the validated digest rule and private pools of a policy, or None."""
    rule = policy.get("digest_rule")
    if rule is None:
        return None
    _validate_digest_rule(rule)
    pools = policy.get("private_pools", {}).get("names", [])
    if not isinstance(pools, list) or not all(isinstance(n, str) and n for n in pools):
        raise ValueError("invalid private pools")
    return {"rule": rule, "pools": pools}


def _entry_for(entries, path, key):
    """The index of the first entry whose key pattern and files match, else None."""
    parts = key.split(">")
    for index, entry in enumerate(entries):
        want = entry["key"].split(">")
        deep = want[0] == "**"
        if deep:
            want = want[1:]
        fits = len(parts) >= len(want) if deep else len(parts) == len(want)
        if fits and all(
            map(fnmatch.fnmatchcase, parts[len(parts) - len(want) :], want)
        ):
            files = entry.get("files")
            if not files or any(fnmatch.fnmatchcase(path, glob) for glob in files):
                return index
    return None


def classify_digests(entries, context, cache, blob_suffix=None):
    """Judge every 64-hex string in the text of a tree.

    `entries` yields (path, bytes or None, sha256 of the shipped bytes). A
    digest passes when it is the sha256 of a shipped file, or sits under a key
    the rule lists as public and neither covers a text file nor names a private
    pool, or sits in a prose file the rule lists with the right count. A public
    digest that equals a blob of the git history (`blob_suffix` maps the sha256
    of each to the suffix of its path) fails when the suffix is not among the
    pre-images its class lists. A digest under a key the rule withholds, under
    a public key that fails a test, or in a prose file the rule withholds, is
    returned to join the withheld set. Any other digest is a violation.
    Returns the withheld digests, the violations as (path, key) pairs, counts
    of distinct values and the digests a reader may find in the tree: shipped
    files and public classes.
    """
    entries = list(entries)
    shipped = {digest for _, _, digest in entries}
    rule = context["rule"]
    blob_suffix = blob_suffix or {}
    withheld, violations = set(), set()
    used = {"public": set(), "withheld": set()}
    values = defaultdict(set)
    prose = defaultdict(set)
    allowed = set(shipped)
    for path, data, _ in entries:
        if data is None:
            continue
        hit = cache.get(path)
        if hit is None or hit[0] != data:
            text = data.decode("utf-8")
            groups = digest_groups(PurePosixPath(path), text, context["pools"])
            hit = cache[path] = (data, groups)
        logical = path
        if path.startswith(f"{EVIDENCE_ROOT}/"):
            logical = path[len(str(EVIDENCE_ROOT)) + 1 :]
        for (key, suffix, pooled), found in hit[1].items():
            fresh = found - shipped
            values["shipped"] |= found & shipped
            if not fresh:
                continue
            if key == "<text>":
                prose[path] |= fresh
                continue
            index = _entry_for(rule["withheld"], logical, key)
            if index is not None:
                used["withheld"].add(index)
                withheld |= fresh
                values["withheld"] |= fresh
                continue
            index = _entry_for(rule["public"], logical, key)
            if index is None:
                violations.add((path, key))
                continue
            used["public"].add(index)
            name = rule["public"][index]["class"]
            preimages = rule["classes"][name]["preimages"]
            # A class with no pre-image files hashes content, not a file, and
            # reads no file name.
            if pooled or (preimages and suffix and suffix not in preimages):
                withheld |= fresh
                values["demoted"] |= fresh
                continue
            wrong = {
                value
                for value in fresh
                if value in blob_suffix and blob_suffix[value] not in preimages
            }
            withheld |= wrong
            values["demoted_blob"] |= wrong
            values["public:" + name] |= fresh - wrong
            allowed |= fresh - wrong
    listed = {path: entry for entry in rule["prose"] for path in entry["paths"]}
    for path, fresh in prose.items():
        entry = listed.get(path)
        if (
            entry is None
            or entry["paths"][path] != len(fresh)
            or (
                entry["class"] != "withheld"
                and entry["digests"][path] != prose_digest_pin(path, fresh)
            )
        ):
            violations.add((path, "<text>"))
        elif entry["class"] == "withheld":
            withheld |= fresh
            values["prose_withheld"] |= fresh
        else:
            values["public:" + entry["class"]] |= fresh
            allowed |= fresh
    counts = {name: len(group) for name, group in sorted(values.items())}
    counts["public_keys_used"] = len(used["public"])
    counts["withheld_keys_used"] = len(used["withheld"])
    return withheld, violations, counts, allowed - withheld


def reachable_blobs(repo):
    """Every blob reachable from HEAD as (digest forms, suffix of its path).

    The list is empty without a commit. A blob that sits under several paths
    carries the first one git lists.
    """

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, check=False
        )

    if git("rev-parse", "--quiet", "--verify", "HEAD").returncode:
        return []
    if git("rev-parse", "--is-shallow-repository").stdout.strip() != b"false":
        raise ValueError("git history is shallow")
    listing = git("rev-list", "--objects", "HEAD")
    if listing.returncode:
        raise ValueError("git object listing failed")
    suffix = {}
    for line in listing.stdout.splitlines():
        if line.strip():
            sha, _, path = line.decode().partition(" ")
            suffix.setdefault(sha, _suffix_of(path))
    return [
        (digest_forms(data), suffix.get(sha, ""))
        for sha, data in _read_blobs(repo, sorted(suffix))
    ]


def _shipped_digest(row):
    final = row["final"]
    return (
        row["original_sha256"] if final is None else hashlib.sha256(final).hexdigest()
    )


def _staged_digests(root, paths):
    """Yield (name, text bytes or None, sha256) for each staged file."""
    for path in paths:
        data = path.read_bytes()
        yield (
            path.relative_to(root).as_posix(),
            None if _is_binary(data) else data,
            hashlib.sha256(data).hexdigest(),
        )


def _digest_violations(violations):
    names = sorted(violations)
    return (
        f"unclassified digest in shipped bytes: {len(names)} key pattern(s) in "
        f"{len({path for path, _ in names})} file(s): "
        + ", ".join(f"{path} [{key}]" for path, key in names[:12])
    )


def settle_withheld(rows, seeds, earlier=(), digests=None, defer=()):
    """Redact every quote of a withheld digest from the redactable rows.

    Sets `final` and `withheld_quotes` on each row. `earlier` holds the rows of
    `earlier_versions`: they join the closure and nothing is written for them.
    `digests` holds the `digest_context` of the policy and the `blobs` of
    `reachable_blobs`. With it the settling repeats until no digest is added:
    a digest the rule withholds, and every form of a reachable blob whose
    bytes do not ship as they are once the shipped bytes have settled, join the
    withheld set. A digest the rule does not classify refuses the export.
    `defer` lists path prefixes whose
    unclassified digests are not refused here: the packager passes the paper,
    whose receipts are rebuilt after the packager runs and are judged by the
    export. The evidence copies stay judged. Returns the 12-character windows of
    the withheld set and the counts for the receipt. The set itself is never
    written anywhere.
    """
    extra, cache, census, violations = set(), {}, {}, set()
    if digests is not None:
        blobs = {forms["sha256"]: suffix for forms, suffix in digests["blobs"]}
    while True:
        withheld, quoting, rounds = withheld_digests([*rows, *earlier], seeds | extra)
        prefixes = _windows(withheld)
        memo = {}
        for index, row in enumerate(rows):
            row["final"], row["withheld_quotes"] = row["data"], 0
            if index in quoting:
                text, count = redact_withheld(
                    row["data"].decode("utf-8"), prefixes, memo
                )
                if not count:
                    raise ValueError(
                        f"withheld digest was not redacted: {row['export_path']}"
                    )
                _check_syntax(row["export_path"], text, row["data"].decode("utf-8"))
                row["final"], row["withheld_quotes"] = text.encode(), count
        if digests is None:
            break
        # A virtual row is the original of an evidence copy and ships at the
        # evidence path.
        entries = [
            (
                f"{EVIDENCE_ROOT}/{row['export_path']}"
                if row.get("virtual")
                else str(row["export_path"]),
                row["final"],
                _shipped_digest(row),
            )
            for row in rows
        ]
        found, violations, census, kept = classify_digests(
            entries, digests["context"], cache, blobs
        )
        # A row that quotes a digest added in this pass is redacted again in the
        # next one, so its bytes are not final while `found` still grows. A blob
        # is judged against the shipped bytes only once they have settled.
        fresh = found - extra
        if not fresh:
            fresh = {
                value
                for forms, _ in digests["blobs"]
                if forms["sha256"] not in kept
                for value in forms.values()
            } - extra
        if not fresh:
            break
        extra |= fresh
    deferred = {
        pair
        for pair in violations
        if pair[0].startswith(tuple(defer))
        and not pair[0].startswith(f"{EVIDENCE_ROOT}/")
    }
    if digests is not None and violations - deferred:
        raise ValueError(_digest_violations(violations - deferred))
    counts = {
        "values": len(withheld),
        "rounds": rounds,
        "earlier_versions": len(earlier),
        "earlier_versions_withheld": sum(
            row["forms"]["sha256"] in withheld for row in earlier
        ),
    }
    if digests is not None:
        counts["reachable_blobs"] = len(digests["blobs"])
        counts["reachable_blobs_withheld"] = sum(
            forms["sha256"] in withheld for forms, _ in digests["blobs"]
        )
        counts["digest_classes"] = census
        counts["digest_other_widths"] = odd_width_census(entries)
        counts["digest_violations_deferred"] = len(deferred)
    return prefixes, counts


def digest_gate(files, context, blob_suffix=None, shipped=()):
    """Classify the digests of a staged tree again, on the bytes on disk.

    `files` yields (path, bytes or None for a binary file, sha256 of the bytes
    on disk). `shipped` adds the digests of files that ship outside this tree.
    Any digest the rule would still withhold, or does not classify, is a
    problem. Returns the problems as strings, none of which quotes a digest, and
    the digests the tree may show.
    """
    found, violations, _, allowed = classify_digests(
        [*files, *((None, None, digest) for digest in shipped)],
        context,
        {},
        blob_suffix,
    )
    problems = []
    if violations:
        problems.append(_digest_violations(violations))
    if found:
        problems.append(f"{len(found)} withheld digest value(s) shipped in the tree")
    return problems, allowed


POOL_SEPARATOR = r"(?:\\textunderscore(?:\{\})?|\{\\_\}|\\_|[\s_-])*"


@functools.lru_cache(maxsize=None)
def _pool_pattern(names):
    alternatives = "|".join(
        POOL_SEPARATOR.join(
            re.escape(part) for part in re.split(r"[\s_-]+", name) if part
        )
        for name in names
    )
    return re.compile(rf"(?<![A-Za-z0-9_])(?i:{alternatives})(?![A-Za-z0-9_])")


def pool_name_pattern(names):
    """Compile the one pattern that finds a private pool by name.

    It ignores case and lets `_`, `-`, a space, a TeX-escaped underscore or nothing
    stand between the parts of a name. A longer name does not match a shorter pool
    and a shorter name does not match a longer pool. The pool gate and the
    demotion of pooled digests both read it.
    """
    return _pool_pattern(tuple(names))


def private_pools(policy):
    """Return the name pattern and the marker of the pools the policy marks private.

    Returns None when the policy names no private pool.
    """
    rule = policy.get("private_pools")
    if rule is None:
        return None
    names, marker = rule.get("names"), rule.get("marker")
    if (
        set(rule) != {"names", "marker"}
        or not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or not isinstance(marker, str)
        or not marker
    ):
        raise ValueError("private_pools needs a list of names and a marker")
    return pool_name_pattern(names), marker


def _pool_leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _pool_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _pool_leaves(item)
    else:
        yield value


def _pool_json(value, pattern, marker, in_list=False):
    """List the ways a parsed JSON value names a private pool other than by a marked key."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if not pattern.search(key):
                found += _pool_json(item, pattern, marker, in_list)
                continue
            leaves = list(_pool_leaves(item))
            if in_list:
                found.append("key inside a list member")
            elif not leaves or any(leaf != marker for leaf in leaves):
                found.append("key with a value other than the marker")
    elif isinstance(value, list):
        for item in value:
            found += _pool_json(item, pattern, marker, True)
    elif isinstance(value, str) and pattern.search(value):
        found.append("string value or list member")
    return found


TEX_UNDERSCORE = re.compile(r"\\textunderscore\s*(?:\{\})?|\{\\_\}|\\_")
TEX_WRAPPER = re.compile(
    r"\\(?:texttt|textsf|textrm|textit|textbf|emph|mbox|mathrm|mathtt)\{([^{}]*)\}"
)
TEX_TABLE_BEGIN = re.compile(
    r"\\begin\{(?:tabular[x*]?|longtable|array|supertabular|tabu)\}"
)
TEX_TABLE_END = re.compile(
    r"\\end\{(?:tabular[x*]?|longtable|array|supertabular|tabu)\}"
)


def _pool_normal(text):
    """Write the TeX spellings of an underscore and the font wrappers as plain text."""
    text = TEX_UNDERSCORE.sub("_", text)
    previous = None
    while previous != text:
        previous = text
        text = TEX_WRAPPER.sub(r"\1", text)
    return text


def _unescape_json_text(text):
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match[1], 16)), text)


def _pool_text(relative, text, pattern, marker):
    """List the table cells of one non-JSON text file that name a private pool.

    Prose may name the pool. A CSV or TSV cell, a Markdown table row or a TeX
    table row may not, except a Markdown row that gives the pool as its first
    cell and the marker in every other cell. The TeX and Markdown spellings of an
    underscore and the TeX font wrappers are read as plain text first. A TeX line
    that names a pool is a table row when it holds `&` or `\\\\`, sits inside a
    table environment, or continues a row from the line before or after it.
    """
    suffix = PurePosixPath(relative).suffix.lower()
    identity = re.compile(
        rf"\|\s*{pattern.pattern}\s*(?:\|\s*{re.escape(marker)}\s*)+\|"
    )
    found = []
    raw_lines = text.splitlines()
    depth = 0
    for number, raw in enumerate(raw_lines):
        opened = len(TEX_TABLE_BEGIN.findall(raw))
        closed = len(TEX_TABLE_END.findall(raw))
        inside = depth > 0 or opened > 0
        depth = max(depth + opened - closed, 0)
        line = _pool_normal(raw)
        if not pattern.search(line):
            continue
        stripped = line.strip()
        if suffix in (".csv", ".tsv"):
            found.append("delimited cell")
        elif "|" in stripped:
            if not identity.fullmatch(stripped):
                found.append("Markdown table row")
        elif suffix == ".tex":
            before = raw_lines[number - 1].rstrip() if number else ""
            after = ""
            if number + 1 < len(raw_lines):
                after = raw_lines[number + 1].lstrip()
            if (
                "&" in raw
                or "\\\\" in raw
                or inside
                or before.endswith("&")
                or after.startswith("&")
            ):
                found.append("TeX table row")
    return found


def pool_gate(files, pools):
    """Refuse a staged tree that names a private pool anywhere but in prose.

    `files` yields (path, bytes or None for a binary file, sha256) as
    `_staged_digests` does. `pools` is the result of `private_pools`. A
    structured file may name a pool only as a key outside any list whose
    whole subtree is the marker. A JSON or JSONL file is always parsed, a key
    that repeats in one object is a finding, and a file that does not parse is a
    finding when it names a pool. A PDF may not name a pool in its text layer.
    Returns the problems as strings that name the file and the kind of
    occurrence, never a value.
    """
    pattern, marker = pools

    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise KeyError("duplicate JSON key")
        return result

    problems = []
    for relative, data, _ in files:
        if data is None:
            continue
        text = data.decode("utf-8", errors="replace")
        suffix = PurePosixPath(relative).suffix.lower()
        found = None
        if suffix == ".pdf":
            named = sum(
                len(pattern.findall(_pool_normal(view)))
                for surface, view in scan_surfaces(relative, data)
                if surface == "pdf_text"
            )
            found = ["named in the text layer"] * (1 if named else 0)
        elif suffix in (".json", ".jsonl"):
            try:
                if suffix == ".json":
                    found = _pool_json(
                        json.loads(text, object_pairs_hook=unique_object),
                        pattern,
                        marker,
                    )
                else:
                    found = [
                        reason
                        for line in text.splitlines()
                        if line.strip()
                        for reason in _pool_json(
                            json.loads(line, object_pairs_hook=unique_object),
                            pattern,
                            marker,
                        )
                    ]
            except KeyError:
                found = ["key repeated in one object"]
            except json.JSONDecodeError:
                found = (
                    ["unparseable structured file"]
                    if pattern.search(_pool_normal(_unescape_json_text(text)))
                    else []
                )
        elif pattern.search(_pool_normal(text)):
            found = _pool_text(relative, text, pattern, marker)
        for reason, count in sorted(Counter(found or []).items()):
            problems.append(f"{relative} [{reason} x{count}]")
    return problems


def path_dates(relative):
    """The calendar dates that the name of a file or of a directory above it carries."""
    dates = set()
    for part in PurePosixPath(str(relative)).parts:
        for year, month, day in FILE_NAME_DATE.findall(part):
            try:
                datetime(int(year), int(month), int(day))
            except ValueError:
                continue
            dates.add(f"{year}-{month}-{day}")
    return dates


def _date_values(value):
    """Yield the value of every key named `date`, at any depth."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "date":
                yield item
            yield from _date_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _date_values(item)


def date_field_lines(name, data):
    """Return the line of the first `date` field that is not a date the path carries.

    The rule covers a JSON or JSONL file whose name or any parent directory name
    carries a date. Its `date` fields, at any depth, must equal one of those dates
    and hold no time of day. The result is empty when all of them do.
    """
    relative = PurePosixPath(name.rpartition(":")[2])
    dates = path_dates(relative)
    suffix = relative.suffix.lower()
    if not dates or suffix not in (".json", ".jsonl"):
        return []
    try:
        text = data.decode("utf-8")
        documents = (
            [json.loads(text)]
            if suffix == ".json"
            else [json.loads(line) for line in text.splitlines() if line.strip()]
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    wrong = [
        value
        for document in documents
        for value in _date_values(document)
        if not isinstance(value, str) or value not in dates
    ]
    if not wrong:
        return []
    first = text.find(json.dumps(wrong[0]))
    return [text.count("\n", 0, max(first, 0)) + 1] * len(wrong)


def name_date_fields(relative, text):
    """Write a `date` field that repeats a date of its file path as that date alone.

    A timestamp gives the time of day away, and with an offset the time zone. When
    the field and the name of the file or of a directory above it carry the same
    date, the date is all the field needs to say.
    """
    if Path(relative).suffix.lower() not in (".json", ".jsonl"):
        return text, 0, []
    named = path_dates(relative)
    lines = []

    def convert(match):
        stamp = DATE_WITH_TIME.fullmatch(match["stamp"])
        if stamp is None or stamp["date"] not in named:
            return match[0]
        lines.append(text.count("\n", 0, match.start()) + 1)
        return match[1] + stamp["date"] + match[3]

    rewritten = DATE_FIELD.sub(convert, text) if named else text
    return rewritten, len(lines), lines[:3]


def utc_timestamps(text):
    """Restate ISO timestamps that carry a local offset as the same instant in UTC."""
    lines = []

    def convert(match):
        offset = timedelta(hours=int(match["hours"]), minutes=int(match["minutes"]))
        if match["sep"] != "T" or not offset or offset > timedelta(hours=14):
            return match[0]
        try:
            local = datetime.fromisoformat(match["date"] + "T" + match["time"])
        except ValueError:
            return match[0]
        instant = local - offset if match["sign"] == "+" else local + offset
        clock = "%H:%M:%S" if match["time"].count(":") == 2 else "%H:%M"
        lines.append(text.count("\n", 0, match.start()) + 1)
        return (
            instant.strftime("%Y-%m-%dT" + clock)
            + (match["fraction"] or "")
            + "+00"
            + match["colon"]
            + "00"
        )

    rewritten = OFFSET_TIMESTAMP.sub(convert, text)
    return rewritten, len(lines), lines[:3]


def offset_timestamps(text):
    """Count timestamps whose numeric offset is not zero, in the forms listed above."""
    extended = sum(
        1
        for form in (GATE_OFFSET_TIMESTAMP, PDF_OFFSET_DATE, RFC2822_OFFSET)
        for match in form.finditer(text)
        if int(match["hours"]) or int(match["minutes"] or 0)
    )
    return extended + len(BASIC_OFFSET_TIMESTAMP.findall(text))


def _pdf_streams(data):
    """Inflate each Flate stream body; a body in another filter yields nothing."""
    for match in re.finditer(rb"(?<!end)stream\r?\n", data):
        end = data.find(b"endstream", match.end())
        if end < 0:
            break
        inflater = zlib.decompressobj()
        try:
            body = inflater.decompress(data[match.end() : end], PDF_STREAM_LIMIT)
        except zlib.error:
            continue
        if inflater.unconsumed_tail:
            raise ValueError("PDF stream exceeds the scan limit")
        yield body


def _joined(text):
    """The alphanumeric characters of `text` alone: a value split by any separator reads whole."""
    return SEPARATORS.sub("", text)


def scan_surfaces(name, data, path=None):
    """Yield (surface, text) for every decoded view of one shipped file.

    A surface whose name ends in `_joined` holds the alphanumeric characters of
    another surface. It is read for digests only, since a timestamp loses its
    shape there.
    """
    relative = PurePosixPath(name)
    suffix = relative.suffix.lower()
    yield "path", name
    if suffix == ".pdf":
        yield "binary", data.decode("latin-1")
        with tempfile.TemporaryDirectory() as scratch:
            if path is None:
                path = Path(scratch) / "member.pdf"
                path.write_bytes(data)
            decoded = read_pdf(path)
            structure = read_pdf_structure(path)
        for surface, value in sorted(decoded.items()):
            text = value.decode("utf-8", errors="replace")
            yield surface, text
            if surface != "pdf_text":
                yield surface + "_joined", _joined(text)
        # Strings and objects the text layer and the metadata calls never show.
        yield from sorted(structure.items())
        text = decoded["pdf_text"].decode("utf-8", errors="replace")
        # A digest may be broken across lines, with or without a hyphen, or by
        # the glyphs a listing sets at a wrapped line.
        yield "pdf_text_joined", re.sub(r"\s+", "", text)
        yield "pdf_text_joined", re.sub(r"\s+", "", re.sub(r"-\s*\n", "", text))
        yield "pdf_text_joined", _joined(text)
        for body in _pdf_streams(data):
            stream = body.decode("latin-1")
            yield "pdf_stream", stream
            if "\0" in stream:
                yield "pdf_stream", stream.replace("\0", "")
    elif not _is_binary(data):
        text = data.decode("utf-8")
        yield "text", text
        yield "text_joined", _joined(text)
        for value in decoded_json_strings(relative, text):
            yield "decoded_json", value
            yield "decoded_json_joined", _joined(value)
    else:
        yield "binary", data.decode("latin-1")
        if suffix == ".png":
            yield "png_metadata", read_png_metadata(data)


CONTAINER_MAGIC = (
    (b"\x1f\x8b", "gzip"),
    (b"PK\x03\x04", "zip"),
    (b"PK\x05\x06", "zip"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"(\xb5/\xfd", "zstd"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)


def container_kind(data):
    """Name the compressed container or UTF-16 text `data` is, or None.

    The scans read bytes as UTF-8, Latin-1 or the decoded forms listed above.
    A member that hides its text behind another encoding is refused instead.
    """
    for magic, kind in CONTAINER_MAGIC:
        if data.startswith(magic):
            return kind
    if data[257:262] == b"ustar":
        return "tar"
    half = len(data) // 2
    if half >= 8:
        even, odd = data[0::2].count(0), data[1::2].count(0)
        if (odd > half * 0.3 and even < half * 0.05) or (
            even > half * 0.3 and odd < half * 0.05
        ):
            return "utf-16"
    return None


def removed_number_lines(text, tracked):
    """Return the line of each number in `text` that `tracked` lists next to a word it had.

    `tracked` maps a removed number to the words that stood next to it in the
    source. A number with other neighbours is not a finding. The lines name the
    place, never the number.
    """
    lines = []
    for found in REDACTED_NUMBER.finditer(text):
        words = tracked.get(found.group().replace(",", ""))
        if not words:
            continue
        before, after = _word_pair(text, found.start(), found.end())
        near = {before, after}
        if before:
            near.add(KEY_WORD + before)
        if words & near:
            lines.append(text.count("\n", 0, found.start()) + 1)
    return lines


def release_gates(
    members, prefixes, exempt=(), allowed=None, removals=None, derivable=frozenset()
):
    """Count withheld-digest quotes and local-offset timestamps in shipped bytes.

    `members` yields (name, data, path). Third-party bytes are not this
    export's to restate, so they are checked for digests only. A compressed
    container or UTF-16 member is a hit, since the scans could not read it.
    With `allowed`, the digests the tree may show, a PDF or archive member
    that shows any other 64-hex string is a hit too, whether the string is
    whole or wrapped at a line end. With `removals`, what the rules took out,
    a text or PDF text surface that shows a removed number beside a word it had
    in the source, or a hex value a rule replaced, is a hit. A replaced hex that
    is the sha256 of a file the tree ships (`derivable`) hides nothing, so it is
    not. Returns the hits and the names of the members that could not be
    decoded.
    """
    memo, hits, unreadable = {}, [], []
    tracked = removals.tracked() if removals is not None else {}
    marked = removals.hexes - derivable if removals is not None else frozenset()
    for name, data, path in members:
        digests, stamps, strays = Counter(), Counter(), Counter()
        counted, clear = Counter(), Counter()
        first_line = {}
        kind = container_kind(data)
        if kind:
            hits.append(
                {"file": name, "gate": "container_member", "surface": kind, "count": 1}
            )
            continue
        try:
            outside = allowed is not None and (
                name.lower().endswith(".pdf") or ":" in name
            )
            for surface, text in scan_surfaces(name, data, path):
                runs = HEX_RUN.findall(text)
                digests[surface] += sum(
                    _quotes_withheld(run.lower(), prefixes, memo) for run in runs
                )
                if marked and surface in DIGEST_SURFACES:
                    clear[surface] += sum(run.lower() in marked for run in runs)
                if tracked and surface in ("text", "pdf_text") and name not in exempt:
                    lines = removed_number_lines(text, tracked)
                    counted[surface] += len(lines)
                    if lines:
                        first_line.setdefault(surface, lines[0])
                if outside and surface in DIGEST_SURFACES:
                    plain = PDF_TEXT_STRING_HEX.sub("<>", text)
                    strays[surface] += len(
                        {
                            value.lower()
                            for view in (plain, WRAPPED_HEX.sub("", plain))
                            for value in DIGEST64.findall(view)
                        }
                        - allowed
                    )
                if (
                    surface not in TIMESTAMP_BLIND_SURFACES
                    and not surface.endswith("_joined")
                    and name not in exempt
                ):
                    stamps[surface] += offset_timestamps(text)
            dated = date_field_lines(name, data) if name not in exempt else []
        except Exception:  # noqa: BLE001 - an undecodable member is a gate result
            unreadable.append(name)
            dated = []
        for gate, counts in (
            ("withheld_digest", digests),
            ("offset_timestamp", stamps),
            ("unclassified_digest", strays),
            ("marker_in_clear", clear),
        ):
            hits.extend(
                {"file": name, "gate": gate, "surface": surface, "count": count}
                for surface, count in sorted(counts.items())
                if count
            )
        hits.extend(
            {
                "file": name,
                "gate": "removed_number",
                "surface": surface,
                "count": count,
                "line": first_line[surface],
            }
            for surface, count in sorted(counted.items())
            if count
        )
        if dated:
            hits.append(
                {
                    "file": name,
                    "gate": "date_field",
                    "surface": "text",
                    "count": len(dated),
                    "line": dated[0],
                }
            )
    return hits, unreadable


# Files the arXiv builder adds to the tarball that the tree does not ship under paper/.
ARXIV_BUILDER_OUTPUTS = frozenset({"main.bbl"})


def require_paper_scans(extra_scans, shipped_pdf, shipped_paper=None):
    """Refuse a paper export whose arXiv tarball and PDF were not scanned too.

    `shipped_pdf` is the sha256 of the PDF the tree ships, or None. The scanned
    PDF must be those bytes, so the scan covers what the reader gets.
    `shipped_paper` maps each shipped `paper/` path to its sha256. Every member
    of a scanned tarball must equal the shipped file of the same name or be a
    builder output listed in `ARXIV_BUILDER_OUTPUTS`.
    """
    archives = [path for path in extra_scans if tarfile.is_tarfile(path)]
    pdfs = [path for path in extra_scans if path.read_bytes().startswith(b"%PDF-")]
    if not archives or not pdfs:
        raise ValueError(
            "paper ships: pass the arXiv tarball and the paper PDF to --also-scan"
        )
    if shipped_pdf is not None and shipped_pdf not in {sha256(path) for path in pdfs}:
        raise ValueError("the scanned PDF is not the PDF the tree ships")
    if shipped_paper is None:
        return
    unbound = []
    for path in archives:
        with tarfile.open(path) as archive:
            for entry in archive:
                if not entry.isfile() or entry.name in ARXIV_BUILDER_OUTPUTS:
                    continue
                expected = shipped_paper.get(str(PAPER_ROOT / entry.name))
                data = archive.extractfile(entry).read()
                if expected != hashlib.sha256(data).hexdigest():
                    unbound.append(f"{path.name}:{entry.name}")
    if unbound:
        raise ValueError(
            f"the scanned tarball holds {len(unbound)} member(s) that are not the "
            "shipped paper files: " + ", ".join(unbound[:12])
        )


def extra_members(path):
    """Yield the members of a staged archive, or the file itself."""
    path = Path(path)
    if not tarfile.is_tarfile(path):
        yield path.name, path.read_bytes(), path
        return
    with tarfile.open(path) as archive:
        for entry in archive:
            if entry.isdir():
                continue
            if not entry.isfile():
                raise ValueError(f"unsupported archive member: {entry.name}")
            yield f"{path.name}:{entry.name}", archive.extractfile(entry).read(), None


def scan_policy_tokens(path, policy):
    """Run the tree audit over a staged archive or PDF.

    The hits name the member and the surface, never the token.
    """
    path = Path(path)
    prefix = f"{path.name}:" if tarfile.is_tarfile(path) else ""
    hits = []
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        for name, data, _ in extra_members(path):
            relative = PurePosixPath(name.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive member name: {name}")
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_bytes(data)
        audit = _audit_tree_v2(root, policy, set())
    hits.extend(
        {
            "file": prefix + row["file"],
            "gate": "policy_token",
            "surface": row["surface"],
            "count": row["count"],
        }
        for row in audit["matches"]
    )
    hits.extend(
        {
            "file": prefix + row["file"],
            "gate": "policy_token",
            "surface": "parser",
            "count": 1,
        }
        for row in audit["parser_failures"]
    )
    return hits


def export_rows(
    copies, policy, exemptions, generated_binaries, skip=None, removals=None
):
    """Transform every planned file in memory; nothing is written.

    A text row outside `paper/` and the third-party pins is redactable. Every
    other row ships its source bytes. Returns the rows, with the two neutral
    table views appended, and the dropped records.
    """
    rows, dropped, taken = [], [], set()
    for original_path in sorted(copies, key=str):
        value = copies[original_path]
        if skip is not None and original_path.is_relative_to(skip):
            continue
        reason = _drop_path(original_path, policy)
        if reason:
            if original_path.is_relative_to(PAPER_ROOT):
                raise ValueError(f"export would drop a paper file: {original_path}")
            dropped.append(
                {
                    "original_path": str(original_path),
                    "original_sha256": value["sha256"],
                    "reason": reason,
                }
            )
            continue
        export_path = _mapped_path(original_path, policy)
        if export_path in taken:
            raise ValueError(f"mapped export path collision: {export_path}")
        source_data = value["source"].read_bytes()
        nontext = original_path.suffix.lower() == ".pdf" or _is_binary(source_data)
        if (
            nontext
            and original_path not in exemptions
            and original_path not in generated_binaries
        ):
            raise ValueError(f"unlisted non-text export input: {original_path}")
        operations = [{"kind": "path_rewrite"}] if export_path != original_path else []
        omission = None
        if original_path in exemptions or nontext:
            export_data = source_data
        else:
            export_data, content_operations, omission = _transform_text(
                export_path, source_data, policy, removals
            )
            operations.extend(content_operations)
        paper = original_path.is_relative_to(PAPER_ROOT)
        if paper and (
            omission or export_path != original_path or export_data != source_data
        ):
            raise ValueError(f"export would rewrite a paper file: {original_path}")
        if omission:
            dropped.append(
                {
                    "original_path": str(original_path),
                    "original_sha256": value["sha256"],
                    **omission,
                    "operations": operations,
                }
            )
            continue
        taken.add(export_path)
        redactable = not (nontext or paper or original_path in exemptions)
        rows.append(
            {
                "original_path": original_path,
                "export_path": export_path,
                "original_sha256": value["sha256"],
                "source": value["source"],
                "size": len(source_data),
                "categories": sorted(value["categories"]),
                "superseded": value.get("superseded_frozen_source_sha256"),
                "operations": operations,
                "artifact_kind": (
                    "third_party_hash_exempt"
                    if original_path in exemptions
                    else "generated_binary"
                    if nontext
                    else "text"
                ),
                "data": None if nontext else export_data,
                "redactable": redactable,
                "rewritten": export_data != source_data,
                "forms": digest_forms(source_data) if redactable else None,
                "runs": _hex_runs(export_data.decode("utf-8"))
                if redactable
                else frozenset(),
            }
        )

    router_relative = _relative(policy["router_table_source"], label="router source")
    if router_relative not in copies:
        raise ValueError("neutral table source is absent from export plan")
    router_source = copies[router_relative]["source"]
    router_csv, router_md = _router_view(router_source, policy["router_table_sha256"])
    router_forms = digest_forms(router_source.read_bytes())
    for name, data in (
        (PurePosixPath("reports/historical_pipeline.csv"), router_csv),
        (PurePosixPath("reports/historical_pipeline.md"), router_md),
    ):
        if name in taken:
            raise ValueError(f"generated report path collision: {name}")
        taken.add(name)
        rows.append(
            {
                "original_path": router_relative,
                "export_path": name,
                "original_sha256": policy["router_table_sha256"],
                "source": router_source,
                "size": len(data),
                "categories": ["neutral_historical_report"],
                "superseded": None,
                "operations": [{"kind": "neutral_four_system_view", "rows": 18}],
                "artifact_kind": "text",
                "data": data,
                "redactable": False,
                "rewritten": True,
                "forms": router_forms,
                "runs": frozenset(),
            }
        )
    return rows, dropped


def load_evidence_sidecar(repo, policy, copies):
    """Read the private digest record of the evidence copies; it never ships."""
    name = policy.get("evidence_sidecar")
    if name is None:
        return None
    relative = _relative(name, label="evidence sidecar")
    if relative in copies or not _excluded(relative, policy):
        raise ValueError("evidence sidecar is not excluded from the export")
    _, path = _source(repo, relative, "evidence sidecar")
    document = json.loads(path.read_bytes())
    files = document.get("files")
    if document.get("schema") != SIDECAR_SCHEMA or not isinstance(files, dict):
        raise ValueError("unsupported evidence sidecar")
    for entry in files.values():
        digests = [
            *entry.get("original", {}).values(),
            entry.get("export_sha256"),
            *entry.get("previous_export_sha256", []),
        ]
        if (
            set(entry)
            != {"original_path", "original", "export_sha256", "previous_export_sha256"}
            or set(entry["original"]) != {"sha256", "sha1", "git_blob_sha1", "md5"}
            or not all(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32,64}", value)
                for value in digests
            )
        ):
            raise ValueError("invalid evidence sidecar entry")
    return files


def evidence_copies(rows, evidence):
    """Match the shipped evidence copies to the private source record."""
    copies, index = {}, None
    for row in rows:
        path = row["export_path"]
        if not path.is_relative_to(EVIDENCE_ROOT):
            continue
        if row["data"] is None:
            raise ValueError(f"evidence copy is not text: {path}")
        name = str(path.relative_to(EVIDENCE_ROOT))
        if name == "index.json":
            index = json.loads(row["data"])
        else:
            copies[name] = row
    if evidence is None:
        if copies:
            raise ValueError("evidence copies need the private source record")
        return copies
    if set(copies) != set(evidence):
        raise ValueError(
            "evidence copies differ from their source record: "
            + ", ".join(sorted(set(copies) ^ set(evidence)))
        )
    for name, row in copies.items():
        if hashlib.sha256(row["data"]).hexdigest() != evidence[name]["export_sha256"]:
            raise ValueError(f"evidence copy differs from its source record: {name}")
    if index is not None:
        files = index.get("files", {})
        if set(files) != set(copies):
            raise ValueError("evidence index differs from the shipped copies")
        if any(
            {"original_sha256", "previous_export_sha256"} & set(record)
            for record in files.values()
        ):
            raise ValueError("evidence index ships source digests")
    return copies


def evidence_rows(repo, copies, rows, policy, evidence, removals=None):
    """Find the row each evidence copy mirrors, building one when none ships.

    An evidence name is the export path of its original. A built row is
    appended to `rows` flagged `virtual`: it takes part in the digest closure
    and is never written. Returns {name: row}.
    """
    shipped = {str(row["export_path"]): row for row in rows}
    allowed = set(policy.get("evidence_excluded_originals", []))
    tracked = None
    resolved = {}
    for name in sorted(evidence):
        entry = evidence[name]
        row = shipped.get(name)
        if row is None:
            original = _relative(entry["original_path"], label="evidence original")
            if str(original) not in allowed and (
                _excluded(original, policy) or _drop_path(original, policy)
            ):
                raise ValueError(f"evidence copy of an excluded path: {name}")
            if original in copies:
                source = copies[original]["source"]
            else:
                if tracked is None:
                    tracked = set(_git_files(repo))
                if original not in tracked:
                    raise ValueError(f"evidence original is not available: {name}")
                _, source = _source(repo, original, "evidence original")
            source_data = source.read_bytes()
            if _is_binary(source_data):
                raise ValueError(f"evidence original is not text: {name}")
            export_path = _relative(name, label="evidence copy")
            if _mapped_path(original, policy) != export_path:
                raise ValueError(f"evidence name is not its original's path: {name}")
            data, operations, omission = _transform_text(
                export_path, source_data, policy, removals
            )
            if omission:
                raise ValueError(f"evidence original cannot be exported: {name}")
            row = {
                "original_path": original,
                "export_path": export_path,
                "original_sha256": hashlib.sha256(source_data).hexdigest(),
                "source": source,
                "operations": operations,
                "data": data,
                "redactable": True,
                "rewritten": data != source_data,
                "forms": digest_forms(source_data),
                "runs": _hex_runs(data.decode("utf-8")),
                "virtual": True,
            }
            rows.append(row)
        elif row["data"] is None:
            raise ValueError(f"evidence original is not text: {name}")
        elif str(row["original_path"]) != entry["original_path"]:
            raise ValueError(f"evidence original differs from the shipped one: {name}")
        if (
            "original" in entry
            and entry["original"]["sha256"] != row["original_sha256"]
        ):
            raise ValueError(f"evidence source changed since it was bound: {name}")
        resolved[name] = row
    return resolved


def withheld_seeds(copies, evidence):
    """Digests withheld before any row is read.

    They are every form of each superseded frozen file, and the recorded source
    and earlier exports of each evidence copy that differs from its source.
    """
    seeds = set()
    for value in copies.values():
        if "superseded_frozen_source" in value:
            seeds.update(
                digest_forms(value["superseded_frozen_source"].read_bytes()).values()
            )
    for entry in evidence.values():
        if entry["original"]["sha256"] != entry["export_sha256"]:
            seeds.update(entry["original"].values())
            seeds.update(
                value
                for value in entry["previous_export_sha256"]
                if value != entry["export_sha256"]
            )
    return seeds


def _committed_blobs(repo):
    """Map each path to the blobs it held in any commit reachable from HEAD.

    A rename or copy links the new path to the old one. A repository without a
    commit has no history. A shallow one is refused: its history is cut off.
    Returns the blobs and the links.
    """

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, check=False
        )

    if git("rev-parse", "--quiet", "--verify", "HEAD").returncode:
        return {}, {}
    if git("rev-parse", "--is-shallow-repository").stdout.strip() != b"false":
        raise ValueError("git history is shallow")
    proc = git(
        "log", "--root", "-m", "-M", "--raw", "--no-abbrev", "-z", "--format=", "HEAD"
    )
    if proc.returncode:
        raise ValueError("git history listing failed")
    blobs, links = defaultdict(set), defaultdict(set)
    fields = iter(proc.stdout.decode("utf-8", errors="surrogateescape").split("\0"))
    for field in fields:
        field = field.lstrip("\n")
        if not field.startswith(":"):
            continue
        _, _, old, new, status = field.split()
        paths = [next(fields)]
        if status[0] in "RC":
            paths.append(next(fields))
            links[paths[1]].add(paths[0])
        for path, blob in ((paths[0], old), (paths[-1], new)):
            if blob.strip("0"):
                blobs[path].add(blob)
    return blobs, links


def _read_blobs(repo, names):
    """Yield (name, bytes) for each named git blob, one at a time."""
    with tempfile.TemporaryFile() as request:
        request.write("".join(f"{name}\n" for name in names).encode())
        request.seek(0)
        with subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as proc:
            for name in names:
                header = proc.stdout.readline().split()
                if len(header) != 3 or header[0].decode() != name:
                    raise ValueError("git object read failed")
                data = proc.stdout.read(int(header[2]) + 1)[:-1]
                if header[1] == b"blob":
                    yield name, data


def earlier_versions(repo, sources, policy):
    """Build a closure row for each earlier committed version of a shipped source.

    A record may quote the digest of a version that was committed before the one
    this export reads. `sources` yields (source file, export path). The history
    of a file is read under its repository path and under its export path, so a
    file that ships from outside the repository is matched by the path it ships
    under. Each version other than the shipped source is judged the way the
    export judges a file: it counts as rewritten when the transform would change
    or refuse it under any export path that descends from it, or when the audit
    or the timestamp gate matches it. The rows hold digests and hex runs only.
    `settle_withheld` adds them to the digest closure; nothing is written.
    """
    blobs, links = _committed_blobs(repo)
    wanted = defaultdict(set)
    for source, export_path in sources:
        if not blobs:
            continue
        current = digest_forms(source.read_bytes())["git_blob_sha1"]
        pending, seen = [str(export_path)], set()
        if source.is_relative_to(repo):
            pending.append(source.relative_to(repo).as_posix())
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            pending.extend(links.get(path, ()))
            for blob in blobs.get(path, ()):
                if blob != current:
                    wanted[blob].add(export_path)
    if not wanted:
        return []
    earlier = []
    with tempfile.TemporaryDirectory() as scratch:
        tree = Path(scratch)
        for index, (blob, data) in enumerate(_read_blobs(repo, sorted(wanted))):
            paths = sorted(wanted[blob], key=str)
            name = f"{index}{paths[0].suffix.lower()}"
            (tree / name).write_bytes(data)
            rewritten = False
            for export_path in paths:
                if export_path.suffix.lower() == ".pdf" or _is_binary(data):
                    continue
                try:
                    shipped, _, omission = _transform_text(export_path, data, policy)
                    rewritten = rewritten or bool(omission) or shipped != data
                except ValueError:
                    rewritten = True
            earlier.append({"name": name, "rewritten": rewritten})
        audit = _audit_tree_v2(tree, policy, set())
        flagged = {
            row["file"] for row in (*audit["matches"], *audit["parser_failures"])
        }
        for row in earlier:
            name = row.pop("name")
            data = (tree / name).read_bytes()
            runs = set()
            try:
                for surface, text in scan_surfaces(name, data, tree / name):
                    if surface.endswith("_joined"):
                        continue
                    runs.update(run.lower() for run in HEX_RUN.findall(text))
                    if surface not in TIMESTAMP_BLIND_SURFACES and offset_timestamps(
                        text
                    ):
                        flagged.add(name)
            except Exception:  # noqa: BLE001 - an undecodable version is withheld
                flagged.add(name)
            row.update(
                redactable=True,
                rewritten=row["rewritten"] or name in flagged,
                forms=digest_forms(data),
                runs=frozenset(runs),
            )
    return earlier


def benchmark_data_licensing(policy, copies):
    """Resolve a pinned current decision without rewriting historical freeze receipts."""
    binding = policy.get("benchmark_data_licensing")
    if binding is None:
        return None  # Historical policies/fixtures retain their original closed state.
    relative = _relative(binding["path"], label="licensing status")
    if relative not in copies or copies[relative]["sha256"] != binding["sha256"]:
        raise ValueError("licensing status hash mismatch")
    status = json.loads(copies[relative]["source"].read_text())
    if (
        status.get("schema") != "benchmark-data-licensing-status/v1"
        or status.get("scope") != "benchmark_data_asset_licensing"
        or status.get("release_ready") is not True
        or status.get("family_render_release_ready") is not True
        or status.get("upstream_version_recovered") is not False
        or status.get("gated_families") != []
        or status.get("remaining_blockers") != []
        or status.get("package_attribution_terms") != "CC BY 4.0"
        or status.get("package_release_ready") is not False
        or not status.get("package_blockers")
    ):
        raise ValueError("unsupported licensing status or publication scope")
    required = dict(status["required_notices_sha256"])
    required[status["decision_path"]] = status["decision_sha256"]
    required[status["attribution_path"]] = status["attribution_sha256"]
    required[str(relative)] = binding["sha256"]
    for name, expected in required.items():
        path = _relative(name, label="licensing input")
        if path not in copies or copies[path]["sha256"] != expected:
            raise ValueError("licensing input hash mismatch: " + name)
    return status, required


def _manifest_row(row):
    data = row["final"]
    digest = (
        row["original_sha256"] if data is None else hashlib.sha256(data).hexdigest()
    )
    operations = list(row["operations"])
    if row["withheld_quotes"]:
        operations.append(
            {"kind": "withheld_digest_redacted", "count": row["withheld_quotes"]}
        )
    return {
        "original_path": str(row["original_path"]),
        "export_path": str(row["export_path"]),
        "original_sha256": row["original_sha256"],
        "export_sha256": digest,
        "bytes": row["size"] if data is None else len(data),
        "categories": row["categories"],
        **(
            {"superseded_frozen_source_sha256": row["superseded"]}
            if row["superseded"]
            else {}
        ),
        "operations": operations,
        "derived_export_view": digest != row["original_sha256"],
        "embedded_hashes_rebound": False,
        "artifact_kind": row["artifact_kind"],
    }


def _generated_row(name, data, category, operation):
    return {
        "original_path": None,
        "export_path": str(name),
        "original_sha256": None,
        "export_sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "categories": [category],
        "operations": [{"kind": operation}],
        "derived_export_view": True,
        "embedded_hashes_rebound": False,
        "artifact_kind": "text",
    }


def load_policy(policy_path):
    """Read and check an export policy. Returns it, its sha256, path and bytes."""
    policy_path = Path(policy_path)
    if policy_path.is_symlink() or not policy_path.is_file():
        raise ValueError("policy must be a non-symlink file")
    policy_path = policy_path.resolve()
    policy_bytes = policy_path.read_bytes()
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    policy = json.loads(policy_bytes)
    required = {
        "tokens",
        "word_boundary_tokens",
        "host_rewrites",
        "home_path_rules",
        "remove_text_tokens",
        "drop_file_patterns",
        "third_party_exempt_sha256",
        "generated_binary_sha256",
        "router_table_source",
        "router_table_sha256",
    }
    if policy.get("schema") != "public-export-policy/v2" or not required <= set(policy):
        raise ValueError("unsupported export policy schema")
    if policy_sha256 not in (V2_POLICY_DIGEST, V3_POLICY_DIGEST, V4_POLICY_DIGEST):
        raise ValueError("unapproved v2 export policy digest")
    # The digest census, the pool gate and the mandatory paper scans each run only
    # when the policy asks for them. The first approved digest is the slot the
    # test fixtures fill; a release policy that omits one would skip it silently.
    missing = [key for key in RELEASE_GATES if not policy.get(key)]
    if policy_sha256 != V2_POLICY_DIGEST and missing:
        raise ValueError(f"release policy lacks gate: {', '.join(missing)}")
    if (
        len(policy["tokens"]) != TOKEN_COUNT
        or len(set(policy["tokens"])) != TOKEN_COUNT
        or hashlib.sha256(_json_bytes(policy["tokens"])).hexdigest() != TOKEN_DIGEST
    ):
        raise ValueError("export policy scrub-token contract mismatch")
    if not set(policy["word_boundary_tokens"]) <= set(policy["tokens"]):
        raise ValueError("boundary-token contract mismatch")
    check_fingerprint_contract(policy)
    _validate_redactions(policy)
    return policy, policy_sha256, policy_path, policy_bytes


def exemption_maps(policy, copies):
    """Check the third-party exemptions and generated binaries against the plan."""
    exemptions = {
        _relative(path, label="third-party exemption"): expected
        for path, expected in policy["third_party_exempt_sha256"].items()
    }
    for relative, expected in exemptions.items():
        if relative.is_relative_to(PAPER_ROOT):
            raise ValueError(f"a paper file cannot be third-party exempt: {relative}")
        if relative not in copies or copies[relative]["sha256"] != expected:
            raise ValueError(f"third-party exemption hash mismatch: {relative}")
    generated_binaries = {
        _relative(path, label="generated binary"): expected
        for path, expected in policy["generated_binary_sha256"].items()
    }
    for relative, expected in generated_binaries.items():
        if relative not in copies or copies[relative]["sha256"] != expected:
            raise ValueError(f"generated binary hash mismatch: {relative}")
    return exemptions, generated_binaries


def settle_export(
    repo, copies, policy, exemptions, generated_binaries, context, removals=None
):
    """Transform every file in memory and settle the digests the export may not quote.

    With `removals`, a `Removed`, the rules record there what they took out.
    """
    rows, dropped = export_rows(
        copies, policy, exemptions, generated_binaries, removals=removals
    )
    evidence = load_evidence_sidecar(repo, policy, copies)
    mirrored = evidence_copies(rows, evidence)
    originals = evidence_rows(
        repo, copies, rows, policy, evidence or {}, removals=removals
    )
    earlier = earlier_versions(
        repo, [(row["source"], row["export_path"]) for row in rows], policy
    )
    digests = (
        None
        if context is None
        else {"context": context, "blobs": reachable_blobs(repo)}
    )
    prefixes, withheld = settle_withheld(
        rows, withheld_seeds(copies, evidence or {}), earlier, digests
    )
    return rows, dropped, digests, mirrored, originals, prefixes, withheld


def withheld_windows(repo, benchmark_root, frozen_source, policy_path):
    """The 12-character windows of every digest an export of `repo` may not quote.

    The windows are the set the release scans match against. The digests behind
    them are never written anywhere.
    """
    repo = _root(repo, "repository")
    benchmark = _root(benchmark_root, "benchmark root")
    frozen = _root(frozen_source, "frozen source")
    policy, *_ = load_policy(policy_path)
    copies = plan_export(repo, benchmark, frozen, policy)
    exemptions, generated_binaries = exemption_maps(policy, copies)
    context = digest_context(policy)
    *_, prefixes, _ = settle_export(
        repo, copies, policy, exemptions, generated_binaries, context
    )
    return prefixes


def build_export_v2(
    repo, benchmark_root, frozen_source, out, policy_path, also_scan=()
):
    repo = _root(repo, "repository")
    benchmark = _root(benchmark_root, "benchmark root")
    frozen = _root(frozen_source, "frozen source")
    policy, policy_sha256, policy_path, policy_bytes = load_policy(policy_path)
    context = digest_context(policy)
    pools = private_pools(policy)

    out_path = Path(out)
    if out_path.exists() or out_path.is_symlink():
        raise ValueError("output path already exists")
    out_resolved = out_path.resolve()
    if any(out_resolved.is_relative_to(root) for root in (repo, benchmark, frozen)):
        raise ValueError("output path is nested in a source tree")
    extra_scans = [Path(path) for path in also_scan]
    for path in extra_scans:
        if path.is_symlink() or not path.is_file():
            raise ValueError("extra scan target must be a non-symlink file")

    copies = plan_export(repo, benchmark, frozen, policy)
    licensing = benchmark_data_licensing(policy, copies)
    source_before = {
        (str(destination), str(source)): sha256(source)
        for destination, value in copies.items()
        for source, _ in value["sources"]
    }
    expected_sources = {
        (str(destination), str(source)): expected
        for destination, value in copies.items()
        for source, expected in value["sources"]
    }
    if source_before != expected_sources:
        raise ValueError("an export source differs from its planned digest")
    exemptions, generated_binaries = exemption_maps(policy, copies)
    removals = Removed()

    # Every file is transformed in memory first. Nothing is written until the
    # digests this export may not quote are settled, and no directory named
    # `tree` exists until every gate has passed.
    (
        rows,
        dropped,
        digests,
        mirrored,
        originals,
        prefixes,
        withheld,
    ) = settle_export(
        repo, copies, policy, exemptions, generated_binaries, context, removals
    )

    applied = {
        operation["rule"]
        for row in (*rows, *dropped)
        for operation in row.get("operations", [])
        if operation["kind"] == "path_scoped_rewrite"
    }
    unused = [
        rule["id"]
        for rule in policy.get("redactions", [])
        if not rule.get("optional") and rule["id"] not in applied
    ]
    if unused:
        raise ValueError("redaction rule applied to no file: " + ", ".join(unused))

    stale = sorted(
        name
        for name, row in originals.items()
        if row["final"] != mirrored[name]["data"]
    )
    if stale:
        raise ValueError(
            "stale evidence copy: "
            + ", ".join(str(EVIDENCE_ROOT / name) for name in stale)
        )
    # An evidence copy whose original does not ship was derived when it was
    # packaged. No manifest row carries its rewrites, so they are counted here.
    derived = Counter()
    for row in originals.values():
        if row.get("virtual"):
            derived["sources"] += 1
            derived["withheld_digest_redacted"] += row["withheld_quotes"]
            for kind in ("utc_timestamp_rewrite", "name_date_rewrite"):
                derived[kind] += sum(
                    operation["count"]
                    for operation in row["operations"]
                    if operation["kind"] == kind
                )
    rows = [row for row in rows if not row.get("virtual")]
    output_paths = {row["export_path"] for row in rows}
    path_map = [
        {
            "original_path": str(row["original_path"]),
            "export_path": str(row["export_path"]),
        }
        for row in rows
        if row["categories"] != ["neutral_historical_report"]
        and row["export_path"] != row["original_path"]
    ]
    manifest = [_manifest_row(row) for row in rows]
    staged = {row["export_path"]: row for row in rows}
    if policy.get("mandatory_paper_scans") and any(
        path.is_relative_to(PAPER_ROOT) for path in output_paths
    ):
        shipped_pdf = next(
            (
                row["export_sha256"]
                for row in manifest
                if row["export_path"] == str(PAPER_ROOT / "main.pdf")
            ),
            None,
        )
        require_paper_scans(
            extra_scans,
            shipped_pdf,
            {
                row["export_path"]: row["export_sha256"]
                for row in manifest
                if PurePosixPath(row["export_path"]).is_relative_to(PAPER_ROOT)
            },
        )

    required_benchmark_outputs = {
        _mapped_path(
            PurePosixPath("benchmark_v2") / _relative(path, label="benchmark"), policy
        )
        for path in policy["benchmark_required_files"]
    }
    missing_required = required_benchmark_outputs - output_paths
    if missing_required:
        raise ValueError(
            "required benchmark output was removed: "
            + ", ".join(map(str, sorted(missing_required, key=str)))
        )

    # Each shipped row inventory is checked against its frozen file on every export.
    inventories = {}
    for row in rows:
        if str(row["export_path"]) in ROW_INVENTORIES:
            frozen_rows = row["source"].read_bytes()
            inventories[str(row["export_path"])] = inventory_changes(
                row["export_path"],
                frozen_rows,
                frozen_rows if row["final"] is None else row["final"],
            )
    changed_inventories = {
        name: counts
        for name, counts in inventories.items()
        if any(value for key, value in counts.items() if key != "rows")
    }

    notice_path = PurePosixPath("EXPORT_NOTICE.md")
    integrity_path = PurePosixPath("PACKAGE_INTEGRITY.json")
    notice = (
        "# Export notice\n\n"
        "This tree is a sanitised export. Receipts are dated records: flags such as "
        "`release_ready: false` inside them record the state on the day the receipt "
        "was written and were not rewritten. Hashes, sizes and file names of private "
        "artefacts and private paths are redacted in some copies. Digests of the "
        "unredacted form of any file this export rewrote, and of any earlier version "
        "of a shipped file that it would have changed or refused, are withheld and read `"
        + WITHHELD_DIGEST
        + "`, because such a digest would let a reader test guesses for a redacted "
        "value. Other embedded digests are historical and were not rebound to the "
        "exported bytes. The author's commit ids of 40 hex characters in the records "
        "refer to a private development repository and do not resolve publicly. "
        "Commit ids of third-party repositories are public. "
        "Timestamps that carried a local offset are restated in UTC. A date field "
        "that repeats the date in its file name reads that date alone. Dated documents "
        "name some files that are not part of this package.\n"
    )
    if context is not None:
        notice += (
            "\nA digest of 64 hex characters ships in this tree only in two cases. "
            "It is the sha256 of a file in this tree, or it is a hash that the "
            "benchmark publishes of a rendered image, model weights, a public "
            "third-party file, a binary output of a run (an array archive or a PDF), "
            "or a position, label or corner. Every other digest "
            "of that length, such as the digest of a receipt, a log, a plan, a "
            "selection file, a script or a text written for the project, reads `"
            + WITHHELD_DIGEST
            + "`.\n"
        )
    if changed_inventories:
        notice += (
            "\nThe row inventories differ from the frozen files in three kinds of "
            "field only, and each reads the marker in place of the digest. `"
            + "` -> `".join(LAUNCHER_DIGEST)
            + "` is the digest of the frozen render launcher. `"
            + "` -> `".join(INVENTORY_DIGESTS["exclusion"])
            + "` is the digest of the list of excluded positions. A `"
            + LAUNCHER_DIGEST[0]
            + "` entry that names a shard manifest is the digest of the frozen "
            "manifest, which this export rewrote. "
            + " ".join(
                f"In `{name}` the three kinds read the marker in "
                f"{counts['launcher_digest_withheld']}, "
                f"{counts['exclusion_digest_withheld']} and "
                f"{counts['manifest_digest_withheld']} of {counts['rows']} rows."
                for name, counts in sorted(changed_inventories.items())
            )
            + " Every other field of every row is unchanged, and the "
            "rows are the same in number and order. The shipped inventories are "
            "therefore not byte-identical to the frozen ones."
            + (
                " `PACKAGE_INTEGRITY.json` binds the shipped bytes."
                if policy.get("package_integrity_inventory")
                else ""
            )
            + "\n"
        )
    if licensing is not None:
        notice += (
            "\nCurrent benchmark-data asset licensing is ready under the "
            "author's decision. See "
            + policy["benchmark_data_licensing"]["path"]
            + ". "
            "No family remains asset-gated. "
            "That statement covers asset licensing only: it does not record that "
            "the package was reproduced or that its publication was approved. "
            "Base weights and adapters are withheld.\n"
        )
    notice = notice.encode()
    if notice_path in output_paths:
        raise ValueError("export notice path collision")
    generated = {notice_path: notice}
    output_paths.add(notice_path)
    manifest.append(
        _generated_row(notice_path, notice, "export_notice", "generated_notice")
    )
    if policy.get("package_integrity_inventory"):
        if integrity_path in output_paths:
            raise ValueError("package integrity inventory path collision")
        shipped_sha256 = {row["export_path"]: row["export_sha256"] for row in manifest}
        integrity_data = _json_bytes(
            {
                "schema": "public-package-files/v1",
                "scope": "shipped files only; inventory itself is bound by external export receipt",
                # A superseded file is declared by its shipped digest alone.
                "source_updates": {
                    name: {"current_sha256": shipped_sha256[name]}
                    for name in sorted(policy.get("updated_package_tools", {}))
                    if name.startswith("rendering/")
                },
                "files_sha256": shipped_sha256,
            }
        )
        generated[integrity_path] = integrity_data
        output_paths.add(integrity_path)
        manifest.append(
            _generated_row(
                integrity_path,
                integrity_data,
                "package_integrity",
                "generated_package_inventory",
            )
        )
    manifest.sort(key=lambda row: row["export_path"])

    out_path.mkdir(parents=True)
    try:
        staging = out_path / "staging"
        staging.mkdir()
        for export_path in sorted(output_paths, key=str):
            target = staging.joinpath(*export_path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if export_path in generated:
                target.write_bytes(generated[export_path])
                continue
            row = staged[export_path]
            data = row["final"]
            if data is None:
                data = row["source"].read_bytes()
                if hashlib.sha256(data).hexdigest() != row["original_sha256"]:
                    raise ValueError("a source changed during export construction")
            target.write_bytes(data)
        validate_tree_inventory(staging, output_paths)
        audit = _audit_tree_v2(staging, policy, exemptions)
        allowed = None
        digest_problems = pool_problems = 0
        if context is not None:
            # The generated files are in the staged tree too, so this pass sees
            # every byte the reader gets.
            problems, allowed = digest_gate(
                _staged_digests(staging, _tree_files(staging)),
                context,
                {forms["sha256"]: suffix for forms, suffix in digests["blobs"]},
            )
            digest_problems = len(problems)
            if problems:
                raise ValueError("; ".join(problems))
        if pools is not None:
            problems = pool_gate(_staged_digests(staging, _tree_files(staging)), pools)
            pool_problems = len(problems)
            if problems:
                raise ValueError(
                    f"private pool named outside a marked key: {len(problems)} "
                    f"finding(s): " + ", ".join(problems[:12])
                )
        hits, unreadable = release_gates(
            (
                (path.relative_to(staging).as_posix(), path.read_bytes(), path)
                for path in _tree_files(staging)
            ),
            prefixes,
            {str(path) for path in exemptions},
            allowed,
            removals,
            derivable={row["export_sha256"] for row in manifest},
        )
        scanned = []
        for path in extra_scans:
            members = list(extra_members(path))
            more, failed = release_gates(
                members,
                prefixes,
                allowed=allowed,
                removals=removals,
                derivable={row["export_sha256"] for row in manifest},
            )
            hits.extend(more)
            hits.extend(scan_policy_tokens(path, policy))
            if pools is not None:
                problems = pool_gate(members, pools)
                pool_problems += len(problems)
                if problems:
                    raise ValueError(
                        f"private pool named outside a marked key: {len(problems)} "
                        f"finding(s): " + ", ".join(problems[:12])
                    )
            unreadable.extend(failed)
            scanned.append(
                {
                    "name": path.name,
                    "sha256": sha256(path),
                    "members": len(members),
                }
            )
        unexplained = sorted(
            set(unreadable) - {row["file"] for row in audit["parser_failures"]}
        )
        if unexplained:
            raise ValueError(
                "release gate could not decode: " + ", ".join(unexplained[:10])
            )
        for gate, label in (
            ("withheld_digest", "withheld digest quoted in shipped bytes"),
            ("offset_timestamp", "local-offset timestamp in shipped bytes"),
            ("container_member", "compressed or UTF-16 member in shipped bytes"),
            ("unclassified_digest", "digest outside the shipped and public sets"),
            ("policy_token", "policy token or fingerprint in a scanned archive or PDF"),
            ("marker_in_clear", "value a rule replaced with a marker, shown in clear"),
            ("removed_number", "redacted number beside a word it had in the source"),
            ("date_field", "date field that is not a date its path carries, alone"),
        ):
            found = [row for row in hits if row["gate"] == gate]
            if found:
                raise ValueError(
                    f"{label}: {sum(row['count'] for row in found)} in "
                    f"{len({row['file'] for row in found})} file(s): "
                    + ", ".join(
                        f"{row['file']}"
                        + (f":{row['line']}" if "line" in row else "")
                        + f" [{row['surface']} x{row['count']}]"
                        for row in found[:12]
                    )
                )
        validate_tree_inventory(staging, output_paths)
        _check_staged(staging, manifest, "during audit")
        source_after = {
            (str(destination), str(source)): sha256(source)
            for destination, value in copies.items()
            for source, _ in value["sources"]
        }
        if source_after != source_before:
            raise ValueError("a source changed during export construction")
        if policy_path.read_bytes() != policy_bytes:
            raise ValueError("export policy changed during construction")

        if licensing is not None:
            status, required = licensing
            exported = {row["original_path"]: row for row in manifest}
            status_path = _relative(
                policy["benchmark_data_licensing"]["path"], label="licensing status"
            )
            for name, expected in required.items():
                row = exported.get(name)
                if row is not None and row["export_sha256"] == expected:
                    continue
                # The status record names the record it supersedes by digest, and
                # that digest may be withheld. Its pins and every other byte stay.
                shipped = None
                if row is not None and name == str(status_path):
                    shipped = (staging / row["export_path"]).read_bytes()
                if (
                    shipped is None
                    or not withheld_only(
                        copies[status_path]["source"].read_bytes(), shipped
                    )
                    or any(
                        json.loads(shipped)[key] != status[key]
                        for key in LICENSING_PINS
                    )
                ):
                    raise ValueError(
                        "required licensing record changed or omitted: " + name
                    )

        manifest_document = {
            "schema": "public-export-manifest/v2",
            "files": manifest,
            "dropped": sorted(dropped, key=lambda row: row["original_path"]),
            "third_party_exempt": [
                {"path": row["export_path"], "sha256": row["export_sha256"]}
                for row in manifest
                if row["artifact_kind"] == "third_party_hash_exempt"
            ],
            "generated_binary": [
                {"path": row["export_path"], "sha256": row["export_sha256"]}
                for row in manifest
                if row["artifact_kind"] == "generated_binary"
            ],
        }
        _write_json(out_path / "manifest.json", manifest_document)
        _write_json(
            out_path / "audit.json", {"schema": "public-export-scrub/v2", **audit}
        )
        (out_path / "audit.md").write_text(_summary(audit, manifest))
        path_lines = ["# Export path map", "", "## Host mappings", ""]
        path_lines.extend(
            f"- `{source}` -> `{replacement}`"
            for source, replacement in policy["host_rewrites"].items()
        )
        path_lines.extend(["", "## Data-root mappings", ""])
        path_lines.extend(
            f"- `{rule['pattern']}` -> `{rule['replacement']}`"
            for rule in policy["home_path_rules"]
        )
        path_lines.extend(["", "## Renamed paths", ""])
        path_lines.extend(
            f"- `{row['original_path']}` -> `{row['export_path']}`" for row in path_map
        )
        (out_path / "EXPORT_PATH_MAP.md").write_text("\n".join(path_lines) + "\n")
        counted = Counter()
        for row in manifest:
            for operation in row["operations"]:
                if operation["kind"] in (
                    "withheld_digest_redacted",
                    "utc_timestamp_rewrite",
                    "name_date_rewrite",
                ):
                    counted[operation["kind"] + "_files"] += 1
                    counted[operation["kind"]] += operation["count"]
        receipt = {
            "schema": "public-export-dry-run/v2",
            "status": "audit_complete",
            "scrub_pass": audit["scrub_pass"],
            "release_ready": False,
            "published": False,
            "policy_sha256": policy_sha256,
            "script_sha256": sha256(Path(__file__)),
            "tree": {
                "files": len(manifest),
                "bytes": sum(row["bytes"] for row in manifest),
                "derived_views": sum(row["derived_export_view"] for row in manifest),
                "dropped_files": len(dropped),
                "third_party_exempt_files": sum(
                    row["artifact_kind"] == "third_party_hash_exempt"
                    for row in manifest
                ),
                "generated_binary_files": sum(
                    row["artifact_kind"] == "generated_binary" for row in manifest
                ),
                "aggregate_sha256": hashlib.sha256(_json_bytes(manifest)).hexdigest(),
            },
            "audit": {
                "match_count": audit["match_count"],
                "matched_files": len({row["file"] for row in audit["matches"]}),
                "parser_failures": len(audit["parser_failures"]),
            },
            # Counts only: the withheld digests themselves are never written.
            "release_gates": {
                "withheld_digest_values": withheld["values"],
                "withheld_digest_rounds": withheld["rounds"],
                "earlier_versions_checked": withheld["earlier_versions"],
                "earlier_versions_withheld": withheld["earlier_versions_withheld"],
                "withheld_digest_redacted_files": counted[
                    "withheld_digest_redacted_files"
                ],
                "withheld_digest_redactions": counted["withheld_digest_redacted"],
                "withheld_digest_hits": _hit_count(hits, "withheld_digest"),
                **(
                    {}
                    if context is None
                    else {
                        "digest_rule": {
                            key: value
                            for key, value in withheld.items()
                            if key.startswith(("reachable", "digest"))
                        },
                        "unclassified_digest_hits": digest_problems
                        + _hit_count(hits, "unclassified_digest"),
                    }
                ),
                **({} if pools is None else {"private_pool_hits": pool_problems}),
                "utc_timestamp_rewritten_files": counted["utc_timestamp_rewrite_files"],
                "utc_timestamp_rewrites": counted["utc_timestamp_rewrite"],
                "name_date_rewritten_files": counted["name_date_rewrite_files"],
                "name_date_rewrites": counted["name_date_rewrite"],
                "offset_timestamp_hits": _hit_count(hits, "offset_timestamp"),
                "evidence_copies": len(mirrored),
                "evidence_copies_checked": len(originals),
                "evidence_only_sources": derived["sources"],
                "evidence_only_withheld_digest_redactions": derived[
                    "withheld_digest_redacted"
                ],
                "evidence_only_utc_timestamp_rewrites": derived[
                    "utc_timestamp_rewrite"
                ],
                "evidence_only_name_date_rewrites": derived["name_date_rewrite"],
                "extra_scans": scanned,
                "row_inventories": inventories,
            },
            "artifacts_sha256": {
                name: sha256(out_path / name)
                for name in (
                    "manifest.json",
                    "audit.json",
                    "audit.md",
                    "EXPORT_PATH_MAP.md",
                )
            },
        }
        if licensing is not None:
            status, _ = licensing
            receipt["benchmark_data_licensing"] = {
                "scope": status["scope"],
                "release_ready": True,
                "family_render_release_ready": True,
                "gated_families": [],
                "status_sha256": policy["benchmark_data_licensing"]["sha256"],
                "decision_sha256": status["decision_sha256"],
                "upstream_version_recovered": False,
                "package_attribution_terms": status["package_attribution_terms"],
            }
            receipt["remaining_release_blockers"] = status["package_blockers"]
            receipt["model_distribution"] = status["model_distribution"]
            receipt["historical_readiness_flags_preserved"] = True
        if policy_path.read_bytes() != policy_bytes:
            raise ValueError("export policy changed before receipt write")
        # A failed scrub keeps its audit report and no tree.
        if audit["scrub_pass"]:
            _check_staged(staging, manifest, "before placement")
            staging.rename(out_path / "tree")
        else:
            shutil.rmtree(staging)
        _write_json(out_path / "complete.json", receipt)
    except BaseException:
        shutil.rmtree(out_path, ignore_errors=True)
        raise
    return receipt


def _check_staged(staging, manifest, when):
    """Refuse an export whose staged file no longer has the audited bytes."""
    for row in manifest:
        if sha256(staging / row["export_path"]) != row["export_sha256"]:
            raise ValueError(f"export target changed {when}: {row['export_path']}")


def _hit_count(hits, gate):
    return sum(row["count"] for row in hits if row["gate"] == gate)


def build_export(
    repo, benchmark_root, frozen_source, out, policy_path, also_scan=(), **kwargs
):
    policy_path = Path(policy_path)
    try:
        schema = json.loads(policy_path.read_bytes()).get("schema")
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        raise ValueError("cannot read export policy schema") from error
    if schema == "public-export-policy/v2":
        if kwargs:
            raise ValueError("v2 export does not accept parser overrides")
        return build_export_v2(
            repo, benchmark_root, frozen_source, out, policy_path, also_scan
        )
    if also_scan:
        raise ValueError("extra scans need a v2 export policy")
    return build_export_v1(
        repo, benchmark_root, frozen_source, out, policy_path, **kwargs
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--frozen-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--policy", default="docs/PUBLIC_EXPORT_POLICY_2026-10-12_v4.json"
    )
    parser.add_argument(
        "--also-scan",
        action="append",
        default=[],
        metavar="PATH",
        help="staged file or tar archive that must also pass the release gates",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        receipt = build_export(
            args.repo,
            args.benchmark_root,
            args.frozen_source,
            args.out,
            args.policy,
            also_scan=args.also_scan,
        )
    except Exception as error:  # noqa: BLE001 - all CLI execution faults map to 1
        print(f"export dry-run failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["scrub_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
