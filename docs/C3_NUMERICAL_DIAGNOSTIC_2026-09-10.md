# C3 numerical instrumentation diagnostic — registered after the failed first attempt

The original C3 attempt at source `027d0c7` failed its first-batch ordinary versus
captured forward-equivalence check. Its outputs and `residual.incomplete` remain
untouched; no completed C3 receipt exists. C2 and C4 results are unaffected.

Before any diagnostic inference, register this bounded sequence on the **same
first two preselected attention boards**, same checkpoint/source/transform and
batch size. No new boards, learning, metric selection, or final-test access.

1. Reproduce `high` matmul precision with automatic SDP backend.
2. Change only matmul precision to `highest`; retain automatic SDP.
3. Retain `highest` and force the math SDP backend.

For each, record ordinary-before, captured and ordinary-after logits; maximum
absolute differences, original 1e-4 atol/rtol pass flags, hook restoration, and
argmax differences versus the frozen saved predictions. The latter requirement
is exact and unchanged. Weights are not modified. Inputs/sources/selection are
hashed; the probe is isolated on GPU 0 after the completed C4 run.

Ranked hypotheses: TF32 rounding in the non-fused path when requesting weights;
then fused/math reduction differences; finally hook/state mutation if restoration
or equivalence fails even with higher precision/math. This is not a model-quality
or retention test. Do not relax thresholds or choose a new sample to pass.

If a numerical-only setting passes both unchanged gates, preregister the minimal
setting correction for a new C3 attempt/prefix; retain the failed original beside
it. Otherwise report the unresolved instrumentation failure rather than claim
validated attention. Run no unregistered fourth condition.

Primary implementation context: PyTorch documents that requesting attention
weights disables the optimized SDP path, that backends can differ numerically,
and that TF32 reduces mantissa precision. This motivates the hypotheses; it does
not yet prove the cause of this run's failure.

- [MultiheadAttention, PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.MultiheadAttention.html)
- [SDP numerical behavior](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [TF32 numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)
