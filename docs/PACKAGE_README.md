# Chess sim-to-real benchmark

A controlled render generator and a 20,000-image benchmark for chess-position recognition, built
to answer one question: when a perception model is trained on renders and used on a camera, what
does synthetic data buy, and what does it cost?

Paper: *What Synthetic Data Buys and Costs: A Sim-to-Real Evaluation Case Study in Chess-Position
Recognition* (Spicy Shark, 2026; PDF: [paper/main.pdf](https://github.com/spicysharkco/chess-sim2real-benchmark/blob/main/paper/main.pdf)). Rendered images: the Hugging Face dataset
`spicysharkco/chess-sim2real-benchmark`.

## What is here

- **Generator** (`rendering/`): procedural board, eight piece families with recorded asset grants,
  a perspective camera, explicit lighting and materials. Five factors vary one at a time: camera
  elevation, piece geometry, board palette, framing, and the board colour convention (which corner
  is light). Labels come from the scene, never from a detector.
- **Benchmark v2** (`benchmark_v2/`): 500 position groups, 400 development and 100 sealed; 20
  factor cells at both colour conventions; positions disjoint, exactly and under the eight board
  symmetries, from 67 inventoried sources including a 6.1-million-puzzle Lichess snapshot. "Sealed"
  names the author's no-scoring rule, not an access control. The final-test manifest rows
  (positions, seeds, cameras and image hashes) are in `benchmark_v2/`. The author did not score
  any model on that split. Its images follow later, and anyone can rebuild the pixels from the
  manifest and the generator.
- **Tools** (`tools/paper/`): scoring, crop diagnostics, freeze and exclusion checks.
- **Paper** (`paper/`): the LaTeX source, the compiled PDF, and the table and figure pipeline
  that built every generated table and figure in the paper from receipts. From the package root,
  `python3 paper/build_tables.py` rebuilds the tables byte for byte, and
  `python3 -m unittest discover -s paper/tests` runs the paper's unit tests. Run from `paper/`,
  `SOURCE_DATE_EPOCH=1789084800 tectonic -X compile main.tex` rebuilds `main.pdf` byte for byte.
  `uv run --no-project --with matplotlib==3.11.1 python paper/build_figures.py` rebuilds the five
  print plots byte for byte, then stops: the attention excerpt needs
  `tools/reports/paper/attention_full_2026-09-10/` (61 files, 54 MB), which this package does not
  hold. `paper/records/` holds the figure, table and attention receipts and the bibliography
  check. The PDF build receipt, the test receipt and the review records are not shipped.
- **Reports and receipts** (`reports/`, `docs/`): the dated records and receipts behind most of the
  numbers in the paper. A few receipts the paper cites are not part of this package.
  Commit identifiers in these records refer to the author's private development history and do
  not resolve in the public repository.

## Three things the benchmark found

1. **A convention confound was mistaken for a domain shift.** The public ChessQueries checkpoint
   produced zero exact boards on our renders under our original colour convention. Flipping one renderer expression, the light/dark
   assignment of the a1 square, raised per-square accuracy from 54.4 to 85.2 and removed 67.7% of
   wrong squares, replicated across six piece families.
2. **Synthetic adaptation always cost real-photo accuracy here.** Low-rank adaptation on 500
   rendered positions lifted the same checkpoint from 15.5 to 64.7 exact on standard-convention renders (dark-only arm), and 8,000 positions to 79.7. Every one of ten adapted checkpoints lost 11 to 23 boards on the 2,129-photo
   ChessReD set, a retention gate fixed before training. The lowest elevation band reaches only 14.5 exact after the largest both-convention budget.
3. **Two evaluation failures we made ourselves, and the checks that caught them.** A pipeline
   selected on a private leaderboard scored below its own single member on 16 of 18 controlled
   splits. Two sets named as holdouts (1,635 and 1,263 images) shared 1,446 and 1,128 image identifiers with the training mixes: 2,398 distinct identifiers, 2,574 counting both sets. An
   identifier-level leak check found it.

## Licences

Generator and renderers: PolyForm Noncommercial 1.0.0 (commercial licence and custom rendered
datasets on request: spicysharkco@gmail.com). Scoring and diagnostic tools: Apache-2.0. Manifests, receipts, hash lists,
rendered images, documentation and paper: CC BY-NC 4.0. An added permission in `LICENSE.md` lets anyone, companies
included, evaluate models on the benchmark and publish the scores. Piece assets keep their creators' terms; see `LICENSE.md` and
`docs/records/BENCHMARK_ATTRIBUTIONS_2026-09-10.md`. ChessQueries code and weights are not
redistributed.

## Reproduction

The rest of this file is the recorded, isolated reproduction procedure: render a subset of the v2
manifests through the packaged generator, compare image hashes to the freeze receipt, and score
the public checkpoint on CPU.

---

# Controlled chess-render benchmark: local reproduction smoke

This source package contains the generator/runtime, v2 manifests, reports,
licence notices and scoring tools. Rendered benchmark images and third-party model
weights are not bundled. Start here, not with historical operational commands in
old preregistrations. Nothing below publishes data or uses a GPU.

## Prerequisites and isolated installation

The recorded exact-image check uses Linux x86-64, Node 24.15.0, Playwright 1.61.0
and its Chromium 149.0.7827.55, CPU SwiftShader, and the packaged Three.js r128.
Different browser/renderer versions may change PNG bytes: do not silently accept a
perceptual match. System prerequisites are `curl`, `tar`, `xz`, `bwrap` (Bubblewrap), standard Chromium
shared libraries, and `uv` (https://docs.astral.sh/uv/getting-started/installation/).
No Python environment, browser cache or project checkout from another experiment
is required. Run every command from the extracted package root, in a fresh folder.

```sh
mkdir -p .runtime/home .runtime/cache .runtime/npm-cache .runtime/python
export HOME="$PWD/.runtime/home"
export XDG_CACHE_HOME="$PWD/.runtime/cache"
export UV_CACHE_DIR="$PWD/.runtime/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python"
export npm_config_cache="$PWD/.runtime/npm-cache"
unset NODE_COMPILE_CACHE
export PLAYWRIGHT_BROWSERS_PATH=0
export HF_HOME="$PWD/.runtime/cache/huggingface"
export TORCH_HOME="$PWD/.runtime/cache/torch"
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void PYTHONPATH=""
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

curl -fL https://nodejs.org/download/release/v24.15.0/node-v24.15.0-linux-x64.tar.xz \
  -o .runtime/node.tar.xz
printf '%s  %s\n' 472655581fb851559730c48763e0c9d3bc25975c59d518003fc0849d3e4ba0f6 \
  .runtime/node.tar.xz | sha256sum -c -
tar -xf .runtime/node.tar.xz -C .runtime
export PATH="$PWD/.runtime/node-v24.15.0-linux-x64/bin:$PATH"
npm install --prefix .runtime --save-exact playwright@1.61.0
node .runtime/node_modules/playwright/cli.js install chromium --no-shell
export PLAYWRIGHT_MODULE="$PWD/.runtime/node_modules/playwright/index.mjs"
export CHROME_BIN="$(node -e 'console.log(require("./.runtime/node_modules/playwright").chromium.executablePath())')"

uv venv --python 3.12.13 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv/bin/python \
  numpy==1.26.4 pillow==10.4.0 timm==1.0.27 safetensors==0.8.0
```

If Chromium reports missing system libraries, install the OS prerequisites listed
by `node .runtime/node_modules/playwright/cli.js install-deps chromium`; that system
operation can require administrator authority. Do not reuse a private browser
installation or change the pinned build as a workaround.

## Public model source and checkpoint

Obtain the following third-party code/weights from their publishers and review
their PolyForm Noncommercial terms. They are external runtime inputs, not part of
this package's permissive scoring-tool licence. The evaluation path loads the
checkpoint on CPU with `pretrained=False`; it needs no separate encoder download.

```sh
curl -fL https://codeload.github.com/JSeytre/chessqueries/tar.gz/2a9f3c00619133541c65de4315149037b5dece80 \
  -o .runtime/chessqueries.tar.gz
tar -xf .runtime/chessqueries.tar.gz -C .runtime
export CQ_REPO="$PWD/.runtime/chessqueries-2a9f3c00619133541c65de4315149037b5dece80/minimal_reproduction"
curl -fL https://huggingface.co/joelseytre/chessqueries/resolve/main/chessqueries-vitL14-644-joint.safetensors \
  -o .runtime/chessqueries.safetensors
printf '%s  %s\n' 6151bdd98fbe25f32080c097eba7ae75808b5615160e4867240931085fc762e5 \
  .runtime/chessqueries.safetensors | sha256sum -c -
```

The checkpoint URL names a moving branch; the checksum, not that branch name, is
its identity. A mismatch is a failure, not permission to substitute another model.
These evaluation dependencies follow the pinned upstream evaluator's versions;
we install only its inference path, not the upstream training/annotation stack.

## Reproduce and score four frozen development views

```sh
bwrap --ro-bind / / --dev /dev --tmpfs /home --tmpfs /tmp \
  --bind "$PWD" "$PWD" --chdir "$PWD" \
  .venv/bin/python tools/paper/package_smoke.py --package "$PWD" --out smoke \
  --playwright "$PLAYWRIGHT_MODULE" --browser "$CHROME_BIN"
bwrap --ro-bind / / --dev /dev --tmpfs /home --tmpfs /tmp \
  --bind "$PWD" "$PWD" --chdir "$PWD" \
  .venv/bin/python tools/paper/score_benchmark.py \
  --root smoke/renders --freeze smoke/selected.freeze.jsonl \
  --parent-freeze benchmark_v2/benchmark.jsonl \
  --system chessqueries --device cpu --repo-cq "$CQ_REPO" \
  --ckpt .runtime/chessqueries.safetensors --out smoke/score
```

These two commands mount a synthetic device directory, hide home directories and
other scratch trees, and rebind only this package as writable. This matters because
Chromium can probe hardware devices even when SwiftShader is selected; environment
variables alone are not a device-access boundary. Bubblewrap requires local user
namespace support. If that boundary cannot be created, fail rather than silently
rerun without it. Downloads happen before entering this boundary.

The fixed selection takes two a1-dark reference geometry views for Poly Haven,
then two for Phuong, ordered by frozen split/ID and with globally distinct exact/D4
positions. These exercise two renderer/asset paths. The runner checks the package
file inventory, the original scene and asset source hashes, the declared source updates
described below, source-manifest rows, every
rendered PNG hash against the original freeze, labels and camera projection. It
writes `smoke/render.complete.json` only after exact matches and a final source
integrity check. Failed attempts retain `INCOMPLETE`; use a fresh output directory.

Scoring uses the packaged `score_benchmark.py`, the public source above and the
public checkpoint, recording per-board predictions plus `smoke/score.complete.json`.
The selected freeze is an explicit four-row projection of the packaged inventory;
its rows/hashes are unchanged. The scorer binds those rows to the parent freeze,
records `selection_complete=true` and `benchmark_complete=false`, and reports
coverage for these four rows only,
**not** the full v2 benchmark, an independent performance result, or the final test.
The old v1 scorer's uniqueness gates are retained by selecting distinct positions.
No final-test image is rendered or scored by this smoke command.

## Full reproduction and provenance limits

This README verifies a bounded installation/render/scoring path, not every renderer,
all 20,000 cells, or the adaptation-training curve. Historical full-run tools under
`tools/paper/` may require inputs that are not part of this package, such as training data
and the checkouts they were run from; they are preserved records, not the smoke entry point.
Do not rewrite old receipt hashes to make this package stand in for those checkouts.
The published launch wrapper removes a machine-local dependency path and the machine
name it wrote into each render record. Five asset READMEs under `rendering/assets/` were
corrected after the freeze. `PACKAGE_INTEGRITY.json` lists these six files under
`source_updates`, each with the hash of the shipped file. They are the only
rendering-source exceptions. Scenes, meshes, textures and scripts stay frozen, and
every regenerated image still has to match its original hash.

The hash of a file is withheld when this package ships that file in changed form.
Where the frozen plan, a row or a receipt quoted such a hash, it reads
`[redacted: withheld digest]`. That covers the six files above, the tools updated for
this package and every sanitised copy. The hash of an earlier version of a shipped
file is withheld the same way when the export would have changed or refused that
version.

The row inventories `benchmark_v2/benchmark.jsonl` (20,000 rows) and
`benchmark_v2/adaptation.jsonl` (16,000 rows) differ from the frozen files in three kinds
of field only, and each reads the marker in place of the hash. `source_sha256` ->
`render_batch.mjs`, the hash of the frozen launch wrapper, reads the marker in every row.
`excluded_fen_sha256`, the hash of the list of excluded positions, reads it wherever a
row carries it. A `source_sha256` entry that names a shard manifest is the hash of the
frozen manifest; it reads the marker for each manifest this export rewrote. Every other
field of every row is unchanged, and the rows are the same in number and order. The
export checks this against the frozen files on every run. The shipped inventories are
therefore not byte-identical to the frozen ones. Receipts that bound the frozen bytes
read the marker as well, and the smoke runner accepts it there. `PACKAGE_INTEGRITY.json`
binds the shipped bytes.

Some checks compare against withheld hashes. They cannot be rerun from this package and
stop with a mismatch by design:

- `tools/paper/lora_curve.py` binds both row inventories to their freeze receipts by file
  hash and requires a 64-character wrapper hash in every row. The receipts and the rows
  read the marker.
- The input checks of `tools/paper/build_paper_figures.py` compare the timing report, its
  receipt, the 15 per-model timing receipts and the adaptation report receipt with
  recorded hashes that read the marker. The four router score files it reads are not in
  this package either. `paper/build_figures.py` is the supported figure path.
- `load_plan` in `tools/paper/benchmark_v2.py` compares each shard manifest with its pin
  in the design plan. The pin of a manifest this export rewrote reads the marker.
- Any other historical tool that verifies a recorded hash which now reads the marker.

`collect_exclusions` in `tools/paper/benchmark_v2.py` cannot be rerun for a different
reason. It re-reads the 67 exclusion source pools, and the pool files are not part of
this package; `benchmark_v2/inputs/pools.json` lists 38 of them without a path.

`PACKAGE_INTEGRITY.json` hashes the shipped files (excluding itself, installed
runtime inputs and generated outputs). Unlisted files outside `.runtime/`, `.venv/` and
the named smoke output directory fail the inventory check; Python bytecode writing is
disabled.

Receipts are dated records. Flags inside them, such as `release_ready: false`, record the
state on the day each receipt was written. Some receipts and documents are sanitised copies
in which hashes, sizes and file names of private artefacts and private paths are redacted.
Dated documents under `docs/` name some files that are not part of this package. See
`LICENSE.md`, `LICENSES/` and `docs/records/BENCHMARK_ATTRIBUTIONS_2026-09-10.md`
for the material-specific terms.

## Real-photo collection materials

The real-photo collection records, the collection guide and the collection protocol are not
part of this package. The real-photo set is planned for a later release under separate terms.
The scripts in `tools/paper/` (`real_photo_*.py`) are included as
source only. The one real-photo receipt the paper cites is `paper/evidence/tools/reports/paper/real_photo_2026-09-11/a1_light_2026-10-04/score_cq_cpu_20261004.json`.
