# C2 data preregistration: separate matched-position adaptation pool

Registered before generation/rendering. This is the data prerequisite for the
later encoder-LoRA experiment, not authorization to begin training or to score
the sealed final partition. The complete training recipe is preregistered later.

## Population and exclusion

Generate **8,000 fresh position groups**, one position per independently sampled
legal game; render one scene per group at both `a1_dark` and `a1_light`, producing
**16,000 images**. All views are `partition=adaptation_train`. Generation seed
20260911; C1 generation/partition seed remains 20260910.

Before accepting any position, reject canonical exact and D4 overlap against all
C1 groups (dev and final_test), historical benchmark positions, all prior cohorts,
every inventoried training/public/reference pool and the complete Lichess corpus.
Publish count/hash evidence, not an unsupported blanket cleanliness claim. No
source may be silently omitted. The generator's own accepted-position set also
rejects duplicates and D4 equivalents within this pool.

## Scene sampling and nested budgets

Sample uniformly from the supported C1 factor **levels**, using independently
shuffled balanced marginal schedules: four elevation bands, every included
geometry family, reference palette plus the six palette treatments, and both
framing bands. Per-level counts differ by at most one across 8,000 groups. This
allows combinations of known factor levels, not additional factor levels. It
does not sample one-factor panel IDs uniformly, which would heavily overweight
the reference geometry and camera settings.

Draw nuisance camera parameters independently per group; both parity twins share
the identical scene and camera. Parity is the only within-group intervention.
The pool uses the same pinned renderer/source/asset lineage as C1.

A single permutation with seed **20260912** defines nested prefixes of **500,
2,000 and 8,000 position groups**, fixed before any training or evaluation.
Both arms use the same groups: dark-only uses N distinct images, parity-augmented
uses 2N distinct images, equally divided between parities. These are explicitly
**matched-position budgets**, not equal distinct-image counts. The later fixed
optimizer-step/batch-size recipe must report exposure and compute separately;
do not disguise duplicated images as a larger data budget.

## Evaluation boundary and completion

Future training consumes this pool only; selection/evaluation uses C1 dev, which
is group-disjoint by construction. C1 final_test is never opened for performance
measurement, fitting, timing or selection. No pool image is included in dev
evaluation merely because it originated from the same rendering vocabulary.

Commit this protocol and the implementation before rendering from a clean clone
on render-host. Validate all 16,000 labels, dimensions, hashes, parity twins, group counts,
marginal counts, nested membership and exclusions. The pool receives its own
artifact-hashed completion receipt, with the C1 receipt bound as a prerequisite.
Keep `release_ready=false` pending licence review. Emit a one-line status when
the pool receipt lands. Stop on missing sources, unexpected counts or new test
regressions rather than relaxing the specification.


## Completion — 2026-09-10 UTC

Completed at **05:12:48 UTC**, from clean rendering source commit
`01c52fb9a69a5b36759b310ae18d26dd8ce662c1`.

- **8,000 position groups; 16,000 distinct PNG hashes**, with exactly one image
  at each parity per group. No final-test rows occur in this training pool.
- Marginals: each elevation band 2,000 groups; each of eight geometry families
  1,000; each framing 4,000; each of seven palette schemes 1,142 or 1,143.
- The 500 / 2,000 / 8,000 budgets are the exact nested prefixes of the registered
  permutation. Small prefixes are randomly composed, not exactly balanced;
  record their realized composition in the adaptation reports.
- **6,200,323** distinct excluded placements, including every C1 group; exclusion
  basis SHA-256 `[redacted: withheld digest]`.
- Combined C1 + adaptation audit: 8,500 exact/D4-unique groups, zero positional
  leakage against all 67 inventoried sources.
- Pool receipt SHA-256:
  `[redacted: withheld digest]`.
  It binds C1 receipt
  `[redacted: withheld digest]`.

Independent streaming verification checked all 16,000 PNG hashes/dimensions,
label hashes, corner hashes, exact path inventories and parity pairs, with zero
mismatches. No image decoding, viewing or model/detector scoring was used by
that verification. The entire render/freeze chain exited zero.

Use the same standalone rerender command documented for C1, substituting
`--cohort adaptation`, in a fresh output directory carrying the preserved design
and manifests. The renderer version, asset hashes and source revision are the
same as C1. Summary and receipts are in
[`benchmark_v2_2026-09-10/`](../tools/reports/paper/benchmark_v2_2026-09-10/).
Dataset pixels, full manifests and private operational inputs remain on render-host.
`release_ready=false`; this data receipt does not authorize publication or
constitute an adaptation-model result.
