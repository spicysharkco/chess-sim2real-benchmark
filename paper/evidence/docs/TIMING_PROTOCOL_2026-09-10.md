# C4 preregistration: end-to-end latency on a fixed C1 development subset

Registered before timing and before C2 releases GPU 0. This protocol authorizes
preparation and CPU validation now; it does not authorize concurrent GPU work.
C4 runs immediately after the complete ten-point C2 matrix and before C3.

## Population

Use only C1 `dev`. Seed **20260914** selects 250 `(position_group,
factor_cell)` scenes, balanced across the 20 factor cells as closely as integer
counts allow. Both parity mates are retained, producing exactly **500 images**.
Position groups may repeat across factor cells because C1 has only 400 independent
groups; report unique groups, repeats, and the maximum scenes per group. This is a
latency sample, not 500 independent accuracy observations. No final-test image may
be read or timed, and the CLI exposes no partition or firewall override.

Before model import or CUDA allocation, verify the sealed 20,000-image C1 receipt,
all 16,000 dev image and label hashes, dimensions, partitions, selection receipt,
and source hashes. Recheck them after timing. Require the release-baseline receipt
and all ten completed C2 point receipts, including completed points that failed the
ChessReD retention gate. Every adapter must remain bound to that baseline, the
pinned source/recipe, and `train_head=false`.

## Systems

Run one fresh process per public alias:

- `cq_release`
- `private_full`, `private_raw`, `private_warp_high`, `private_warp_low`
- all ten `lora_{dark|both}_n{500|2000|8000}_s{303|304|305}` registered C2 points

Private aliases do not expose historical member numbers. Private variants are
made with `member_eval.make_variant` in scratch space; weights are symlinked and
never changed. The full system retains its native detector, warp, router, member
mix, rotation TTA, orientation and decoder. Subsets retain the existing
`make_variant` semantics. Consequently these systems do **not** have equal FLOPs,
and C4 must not claim that they do.

The private ensemble is extracted once by the controller from the frozen archive; its size and digests are [redacted: private artefact]. C4 requires that extraction's receipt and
hashes every extracted source/weight before and after each run; it neither
extracts again nor touches the source archive.
The frozen extraction receipt SHA-256 is [redacted: private artefact];
it binds exactly [redacted: private artefact] extracted files plus the separate identity JSON. Extra,
missing, or self-consistently re-receipted files are rejected.

## Timer

All systems receive the same harness-decoded contiguous uint8 RGB arrays. File
decode and BGR-to-RGB conversion are outside the primary timer. Each native
pipeline begins from that RGB input. ChessQueries resize, normalization, host-to-
device transfer, FP16 forward, argmax, prediction transfer to host and FEN
construction remain inside the timer. The private system includes its
detector/router/warp/TTA and CPU postprocess.

Use batch size **1** and CUDA FP16 autocast. Run 30 ordered warm-up images, excluded
from results, followed by all 500 images in manifest order. Immediately synchronize
CUDA before `time.perf_counter()` and after prediction. Save every finite per-image
duration and prediction in JSONL, then report median and linearly interpolated p90.
Model construction and image decode are excluded. A model-only secondary timer is
not required.

## Hardware and interference boundary

Run only on render-host with physical GPU 0 exposed as `CUDA_VISIBLE_DEVICES=0`. Record GPU
name, driver, total memory, the fixed **270 W** power limit, GPU/memory utilization, Torch/CUDA/Python
versions and UTC timestamp. Fail before allocation if another compute PID occupies
GPU 0; repeat that check during warm-up/timing and at completion. Never kill another
process. C2 owns GPU 0 until its matrix finishes, so C4 must wait rather than probe.

Every output prefix is no-clobber reserved. Successful runs receive artifact-hashed
completion receipts and raw timing JSONL. Execution/interference/nonfinite failures
retain the incomplete marker and any flushed per-image evidence. These measurements
describe this hardware/software/input pipeline only; they are not universal latency
or compute-efficiency claims.

The controller first runs `prepare`, then invokes `timing-all` once. `timing-all`
launches the 15 aliases serially as fresh child processes and publishes no aggregate
completion receipt until all 15 per-model `c4_timing_complete` receipts validate.
The aggregate receipt also uses `status=c4_timing_complete`,
`release_ready=false`, and `final_test_images_read=false`.

```sh
python tools/paper/timing_harness.py prepare \
  --c1-root "$R" --out "$T/selection"

CUDA_VISIBLE_DEVICES=0 CHESSQUERIES_DATA_ROOT=$HOME/bench_sources \
python tools/paper/timing_harness.py timing-all \
  --c1-root "$R" --selection "$T/selection.jsonl" \
  --matrix-root "$L" --baseline-receipt "$L/results/release.complete.json" \
  --lora-report-receipt "$T/lora_report.complete.json" \
  --cq-repo "$L/upstream" \
  --checkpoint $HOME/bench_experiments/codex_parity_20260909/upstream/checkpoint.safetensors \
  --submission "$T/private_ensemble" \
  --submission-receipt "$T/private_ensemble.complete.json" \
  --out "$T/timing"
```

## CPU-only preparation evidence

On 2026-09-10, before C4 authorization, the verified private extraction was
copied to fresh scratch space and imported with `CUDA_VISIBLE_DEVICES=""` using
the existing render-host runtime. Calling `_load()` initialized all three configured
models on CPU, including the DINOv3 member; Torch reported CUDA uninitialized.
No image was decoded, no `predict()` call was made, and no inference or timing
was performed. The extraction receipt and all [redacted: private artefact] extracted file hashes were
identical before and after; scratch was removed. This proves dependency and
checkpoint-load compatibility only, not latency, accuracy, or GPU readiness.
