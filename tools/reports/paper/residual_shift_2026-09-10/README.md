# C3 artifact custody

- `r1/`: compact complete-attempt metrics, selection, per-batch numerical checks,
  report, and original completion receipt.
- `diagnostics/`: the fixed three-condition numerical probe, including logits.
- `failed_original/`: failed first C3 attempt and pre-allocation probe-launch failure.
- `examples/`: unchanged first-selected source-frame sheet in each attention family.
- `overall_recovery.*`: verified top-level completion after retaining the failed attempt.

The original R1 receipt also binds `residual_r1.attention_raw.npz` (14 MB). That
file and all 60 contact sheets (54 MB) are retained on render-host under
`~/bench_experiments/codex_post_matrix_20260910/residual_r1.*`; the raw NPZ is also
available in this working tree but intentionally not committed. Re-verifying the
original receipt from a fresh Git-only checkout requires obtaining that bound
NPZ. The contact-sheet manifest independently lists all 60 image hashes; all
were verified against render-host's files. This is not a claim that Git contains every
binary artifact.
