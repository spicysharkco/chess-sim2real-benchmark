# C3 R1: precision-only instrumentation correction

Registered before rerunning the full C3 attention sample. The first C3 attempt
at source `027d0c7` remains **failed/incomplete**, and its prefix/log are retained.
The completed C2 matrix, all model/retention results, and C4 are not changed or rerun.

The registered three-condition first-batch diagnostic reproduced the issue:

| Matmul / SDP | Capture maximum absolute logit difference | Original allclose gate | Saved argmax differences |
|---|---:|---|---:|
| high / automatic | 0.0018221736 | fail | 0 |
| highest / automatic | 0.0000324249 | pass | 0 |
| highest / math | 0.0000123978 | pass | 0 |

Ordinary-after versus ordinary-before was exactly zero in all three cases;
hooks were restored. This supports a precision/kernel explanation, not a
weight/state mutation. The diagnostic used the same first two already-selected
boards, not a newly selected easy sample. Its first launcher attempt was rejected
before GPU allocation because its log occupied the output prefix; that failure
is retained separately.

**Minimal correction selected by the preregistered priority:** set matmul
precision to `highest` during ordinary-before, upstream attention capture, and
ordinary-after. Keep the automatic SDP backend; forcing math is unnecessary on
this diagnostic. Assert FP32 model/input tensors and restore the caller's matmul
precision even on failure. Log every batch's numerical differences before raising
any validation failure.

All substantive C3 controls are unchanged: the same frozen checkpoint and source,
288 historical canonical rows, identical 30-board selection, all 64 queries,
unchanged **atol=1e-4, rtol=1e-4**, exact saved-prediction argmax agreement,
hook/source/input integrity checks, and no final-test access. A failed exact or
numerical check still stops the run; no further automatic tolerance changes.

Performance/confusion metrics continue to use the original saved default
predictions. The attention instrumentation now uses stricter numerical precision
than their historical TF32-enabled inference; this distinction must accompany
the maps. Attention remains descriptive, with mechanism unresolved.

Execute from a new clean source checkout and new `residual_r1` prefix only after
the original process has exited and GPU 0 is idle. Do not overwrite the original
attempt or relabel its failed gate as a pass. `release_ready=false`.

See [the preregistered diagnostic](C3_NUMERICAL_DIAGNOSTIC_2026-09-10.md) and
[PyTorch's TF32 explanation](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).
