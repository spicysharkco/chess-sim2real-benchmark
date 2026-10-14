# C1 preregistration: paired convention-controlled benchmark v2

Registered before generation/rendering. This benchmark is a diagnostic research
artifact, not a claim of model superiority. Historical results remain unchanged.

## Fixed design

- 500 fresh legal position groups, one position per independently sampled game.
  Shared across all factor panels. `position_group` is SHA-256 of canonical
  piece-placement FEN. A seeded permutation assigns 400 groups to `dev` and 100
  to `final_test`; every view of a group remains in its partition. Seed 20260910.
- Four elevation bands: 8–15, 15–25, 25–45, 45–60 degrees. Seven geometry families:
  Poly Haven, Deplorable, KillGorack, Lyricsz, Omie, Phuong, PJ Hanzlik. Six held-out
  palettes. Two framing bands: compact 0.30–0.40 and standard 0.52–0.61 board width.
- Every factor level appears at both parity levels: `a1_dark` reference and
  `a1_light` synthetic convention stress. 19 factor levels × 2 × 500 = **19,000
  cell-images**, comprising **15,200 dev and 3,800 final-test cell-images**.
- Reference: Poly Haven, elevation 45–60, standard framing, approved palette 0
  (`copper_pale_teal`), board light RGB [215,206,190], dark RGB [170,148,126].
  The shared reference geometry/elevation/framing views supply paired reference
  images for the six palette treatments. Identical reference views may occur in
  several panels; report duplicate image hashes, not fictitious independent data.
- Within each group, share azimuth, normalized elevation/framing draws, roll and
  position. Only the varied factor changes; elevation/framing require the
  corresponding physical camera parameters to change. Other panels use reference
  settings. Render-derived distance/projection metadata are recorded, not treated
  as independently held-fixed quantities.
- Nailfighter may be added only if its renderer and source-licence validation
  together take less than one hour, without changing any existing group/partition.
  Inclusion is decided before full rendering, not from model results. It adds
  1,000 cell-images (800 dev / 200 final), for totals 20,000 / 16,000 / 4,000.
  Otherwise report it deferred. No additional rendering-look or occlusion axes.

## Exclusions and lineage

Reject canonical exact and all eight D4 equivalents against every inventoried
training/public/reference pool, historical benchmark and prior experiment cohort,
including the complete 6,100,960-row Lichess source. Deduplicate paired views to
position groups for audits. Record source row counts, distinct counts, content
hashes, `excluded_fen_count`, `excluded_fen_sha256`, and policy. No silently omitted
or empty required source. Cross-partition and later adaptation-pool overlap is
forbidden. Material-only similarities are descriptive, not rejection conditions.

Use one explicit parity switch in each renderer lineage, keeping pinned geometry
identity and legacy guard behavior. New dataset roles are explicit and separate
from historical roles. Produce manifests before rendering, then commit the
implementation and render from a fresh clone on render-host. Record the actual checkout
commit, code-byte hashes, external asset hashes and CPU browser/backend identity.
No copied source edits after source freeze. External assets are separately pinned.

## Final-test firewall and validation

`final_test` is never used for recognition, localization-detector scoring,
attention inspection, selection, timing or training before the designated final
evaluation. Rendering and integrity/GT projection checks are permitted. The
schema-v2 freeze receipt includes the complete final-test image-hash list. Scorers
must reject final-test requests by default; this task provides no final-scoring
override. Freeze generation does not call a detector, even for development.

Validate paired labels/camera/factors, geometry guards, independent asymmetric
corner fixtures, all images/labels/hashes, exact/D4 separation and expected cell
counts. Run rendering and tool tests on render-host, classify baseline failures by exact
test identity, and reject new regressions. Known failure counts are not a waiver
for unrelated new failures. Completion requires an artifact-hashed schema-v2
receipt and complete exclusion audit. `release_ready` remains false pending the
separate licence review; no public upload or model result is part of C1.

## Execution record (append after completion)

### Inclusion decision before rendering

Nailfighter is included. Its existing converted mesh was reusable; renderer
support and validation completed inside the one-hour limit. The author's asset
page explicitly identifies the licence as **Creative Commons Zero v1.0 Universal**:
[LowPoly Chess Set](https://nailfighter.itch.io/lowpoly-chess-set), checked
2026-09-10 UTC. The converted asset digest is
`0d96781041e87b0330f7646cb0720f63acbb42a8f6806d935d81dace20a6951a`.
This is evidence for this asset's inclusion, not completion of the overall
release licence review. The locked C1 totals are therefore **20,000 cell-images,
16,000 dev and 4,000 final_test**, across 500 independent position groups.

The adaptation pool has eight geometry levels (1,000 groups each); its other
balanced marginals and 16,000-image total are unchanged. Palette levels denote
complete supplied material schemes, including RGB, roughness and metalness—not
an RGB-only intervention. Final-test performance remains sealed.

All execution is on render-host; rendering uses CPU SwiftShader from a script-file-backed
named tmux session. Preserve other jobs and retain incomplete artifacts on failure.
The final entry will give the locked cell table, source commit, exclusion counts,
receipt hashes, exact rerender command and test evidence. Emit a one-line status
when the C1 completion receipt lands. No model scoring is authorized by this task.

## C1 completion — 2026-09-10 UTC
Completed at **04:58:28 UTC** from clean source commit
`01c52fb9a69a5b36759b310ae18d26dd8ce662c1`. Renderer smoke fixtures covered all
eight geometries at high/low elevation and both parities (32 images), outside
the benchmark populations. No detector or recognizer was invoked on C1.
| Factor | Level | Dark dev | Dark final | Light dev | Light final |
|---|---|---:|---:|---:|---:|
| elevation | 08_15 | 400 | 100 | 400 | 100 |
| elevation | 15_25 | 400 | 100 | 400 | 100 |
| elevation | 25_45 | 400 | 100 | 400 | 100 |
| elevation | 45_60 | 400 | 100 | 400 | 100 |
| framing | compact | 400 | 100 | 400 | 100 |
| framing | standard | 400 | 100 | 400 | 100 |
| geometry | deplorable | 400 | 100 | 400 | 100 |
| geometry | killgorack | 400 | 100 | 400 | 100 |
| geometry | lyricsz | 400 | 100 | 400 | 100 |
| geometry | nailfighter | 400 | 100 | 400 | 100 |
| geometry | omie | 400 | 100 | 400 | 100 |
| geometry | phuong | 400 | 100 | 400 | 100 |
| geometry | pjhanzlik | 400 | 100 | 400 | 100 |
| geometry | polyhaven | 400 | 100 | 400 | 100 |
| palette | aubergine_periwinkle | 400 | 100 | 400 | 100 |
| palette | cobalt_porcelain | 400 | 100 | 400 | 100 |
| palette | indigo_lavender | 400 | 100 | 400 | 100 |
| palette | navy_coolrose | 400 | 100 | 400 | 100 |
| palette | plum_mintgreen | 400 | 100 | 400 | 100 |
| palette | wine_iceblue | 400 | 100 | 400 | 100 |

There are **20,000 cell-images but 18,000 unique PNG hashes**, as expected:
the three reference cells deliberately reuse identical FEN/camera/material
configurations. The independent sampling units are **500 position groups**, not
20,000 unrelated observations. Final-test performance remains unopened.

Exclusion evidence: **67 named source pools**, **6,199,823 distinct excluded
placements** for C1; the separate adaptation pool excludes those plus all 500 C1
positions (**6,200,323**). The source inventory includes the verified 6,100,960-row
pre-move Lichess snapshot. The combined 8,500 generated position groups are
exact/D4-unique and have **zero exact/D4 leakage**. Material-only matches (8,077)
are descriptive, not positional leakage or a rejection criterion.

- C1 receipt SHA-256: `[redacted: withheld digest]`.
- Exclusion basis SHA-256: `[redacted: withheld digest]`.
- The complete final-test image-hash list is inside the C1 receipt.
- Receipts and cell summaries: [`benchmark_v2_2026-09-10/`](../tools/reports/paper/benchmark_v2_2026-09-10/).
- `release_ready=false`: this is not a publication or completed licence review.

### Rerender command and software boundary

Use the clean source commit above and the frozen design manifests/assets. Set
`RUN` to the preserved run directory, `REPO` to that clean checkout,
`PLAYWRIGHT_MODULE` to the pinned Playwright entry point, and `CHROME_BIN` to the
Chromium executable for the recorded build. The standalone rerender command (the same committed entry point used by the launcher) is:

```sh
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/paper/run_benchmark_v2.py \
  --run "$RUN" --repo "$REPO" --cohort benchmark \
  --playwright "$PLAYWRIGHT_MODULE" --browser "$CHROME_BIN" --workers 4
```

The runner refuses existing output shards. A fresh rerender needs a separate run
directory with the same preserved design files and no existing `renders/` outputs;
it must not overwrite this freeze. Rendering uses Node v24.15.0, Chromium
149.0.7827.55, Three.js r128 and the verified CPU SwiftShader backend. Full
command scripts and private operational inventories remain with the run on render-host;
only the release allowlist should be exported, not historical repository content.

### Test and independent-review evidence

The fresh render-host source run produced **472 passed, 17 failed, 3 skipped**. All failure
identities, including the three failing Blender subtests, were present in the
recorded baseline; there were **zero new failure identities**. The old “2+7”
baseline count was incomplete, so this is not described as a globally green suite.
The transfer-test import repair removed all five recorded transfer failures;
execution-only image/checkpoint dependencies are now lazy-loaded.

Independent metadata review verified all source/asset and shard-manifest hashes,
paired cameras, partition counts, adaptation marginals/nested budgets, exact/D4
separation and exclusion-audit bindings. It did not inspect final-test images or
run performance scoring. The freeze validated every PNG/label, analytic corner
projection and parity pair. Focused CPU tests and static checks passed.
