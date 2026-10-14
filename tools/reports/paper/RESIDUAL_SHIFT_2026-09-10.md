# Residual recognition shift — C3 complete after numerical-instrumentation repair

**Mechanism remains unresolved.** The fixed a1-dark confirmation cohort still
contains **0/48 exact boards for lyricsz and pjhanzlik**, versus **19/48 for
deplorable**. [All six family metrics, confusion matrices and histograms](residual_shift_2026-09-10/r1/residual_r1.md)
are bound by the [R1 receipt](residual_shift_2026-09-10/r1/residual_r1.complete.json).

GT-occupied exact-class accuracy is **69.49% / 66.11%** for lyricsz/pjhanzlik,
versus **87.16%** for deplorable. But this is not merely fine-grained piece-type
confusion: empty↔occupied mistakes account for **347/519** wrong squares in
lyricsz and **443/601** in pjhanzlik. Missed pieces, hallucinations and square
misassignment can all contribute; these counts do not uniquely identify a cause.
The historical families use different positions, unlike C1's paired geometry
panel, so cross-family differences here are descriptive rather than a matched
geometry intervention.

## Attention evidence and limits

All **30 preselected boards × 64 queries** were captured: ten each from lyricsz,
pjhanzlik and deplorable. All 60 source-frame/model-input sheets and the full
30×64×46×46 map tensor passed hash/shape/finite/normalization checks. An independent
check matched every captured argmax to its frozen prediction and recomputed all
six confusion matrices and wrong-square histograms.

These are the first preselected source-frame sheets in each family, not examples
chosen after inspecting correctness:
- [lyricsz](residual_shift_2026-09-10/examples/confirmation__lyricsz__parity__000179.source1920x1080.png)
- [pjhanzlik](residual_shift_2026-09-10/examples/confirmation__pjhanzlik__parity__000246.source1920x1080.png)
- [deplorable](residual_shift_2026-09-10/examples/confirmation__deplorable__parity__000205.source1920x1080.png)

The maps are descriptive query-to-patch weights, not causal attribution. Prefix
tokens are excluded and patch weights renormalized; display colour is scaled to
each query's maximum. These transformations must accompany any reproduced figure.

## Failed attempt and preserved recovery

The original `027d0c7` attempt **failed** ordinary/captured forward equivalence
under `high` matmul precision; its incomplete marker and log remain in
[failed_original](residual_shift_2026-09-10/failed_original/). It was not relabelled
as a pass. The [preregistered diagnostic](../../../docs/C3_NUMERICAL_DIAGNOSTIC_2026-09-10.md)
reproduced maximum logit error 0.001822 under high/auto, versus 0.0000324 under
highest/auto, with exact hook restoration and zero prediction changes.

The [R1 amendment](../../../docs/C3_PRECISION_AMENDMENT_2026-09-10.md) selected the
minimal precision-only correction before rerunning all 30 boards. **Atol/rtol
remain 1e-4; exact saved-argmax matching remains required.** All 15 batches passed;
maximum capture difference was **0.0000426**, restoration difference **zero**.
The original metrics and the entire 30-board selection are **byte-identical**.
Attention uses full FP32/highest during capture, restoring the caller's historical
high setting afterward; performance metrics still use the original saved predictions.

R1 source: `02051bb`. Receipt SHA-256:
`[redacted: withheld digest]`.
The full 60-sheet collection (54 MB) and raw maps/logits (14 MB) remain at
`render-host:~/bench_experiments/codex_post_matrix_20260910/residual_r1.*`;
Git holds compact reports, manifests, diagnostics and three visual examples.
No final-test image was read. `release_ready=false`.
