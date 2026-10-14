# C2 preregistration: upstream LoRA hyperparameters, matched-position budgets

Registered before adaptation training. The later instruction to use upstream
defaults with only the stated protocol changes is controlling: **the head,
queries, decoder and norm are frozen**, superseding the earlier trainable-head
proposal. No extra head-tuning arm is added. The independent unchanged-release
baseline is registered in `C2_BASELINE_2026-09-10.md`.

## Source and parameterization

Pin ChessQueries commit `2a9f3c00619133541c65de4315149037b5dece80` and release
checkpoint SHA-256
`6151bdd98fbe25f32080c097eba7ae75808b5615160e4867240931085fc762e5`.
Use the authors' LoRA implementation, not a newly written approximation:
[LoRA module](https://github.com/JSeytre/chessqueries/blob/2a9f3c00619133541c65de4315149037b5dece80/chessqueries/models/lora.py),
[adaptation configuration](https://github.com/JSeytre/chessqueries/blob/2a9f3c00619133541c65de4315149037b5dece80/chessqueries/models/adapt.py),
[few-shot CLI](https://github.com/JSeytre/chessqueries/blob/2a9f3c00619133541c65de4315149037b5dece80/scripts/lora_fewshot.py).

| Setting | Locked value |
|---|---|
| Trainable weights | Encoder LoRA A/B only; base encoder and entire readout frozen |
| Targets | `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` |
| Rank / alpha / dropout | 8 / 16 / 0.05 (few-shot CLI dropout, not the lower-level API's zero default) |
| Expected trainables | 96 adapted linear modules, 192 tensors, 3,145,728 parameters |
| Optimizer | AdamW, LR 0.0001, weight decay 0, betas (0.9, 0.999), epsilon 1e-8 |
| Scheduler / warm-up | None / zero |
| Training | Batch 6, bf16 autocast, exactly 1,500 optimizer updates |
| Loss | Mean 13-class cross-entropy over all 64 squares |
| Input transform | Authors' full-frame `eval_transform(644)`; no input augmentation or TTA |
| Sampling | Seeded epoch shuffle, drop incomplete batches; 9,000 images presented per run |
| Checkpoint | Fixed final update 1,500; no early stopping or best-on-dev checkpoint selection |
| Evaluation | Batch 4, fp32/TF32, argmax, evaluation mode |

The LR/weight-decay/no-scheduler values come from the upstream
[few-shot implementation](https://github.com/JSeytre/chessqueries/blob/2a9f3c00619133541c65de4315149037b5dece80/chessqueries/train/fewshot.py)
and adaptation defaults, not a learning rate printed in the README. This uses
upstream **adapter/optimizer hyperparameters**, not an unchanged upstream
experiment: the matched-position data, fixed 1,500 steps, batch/precision, two
arms and explicitly disabled input augmentation are this protocol. Fixed-step
evaluation replaces upstream early stopping/checkpoint selection. Do not hide
these differences behind an unqualified “identical upstream experiment” claim.

## Data and run matrix

Training uses only the completed, separate adaptation pool: 8,000 fresh position
groups with both parities, exact/D4-disjoint from the entire C1 benchmark and all
inventoried historical sources. Its receipt must bind the exact current C1 and
design receipts. The seed-20260912 permutation provides nested group prefixes.

| Groups | Dark-only images | Both-parity images | Seeds per arm |
|---:|---:|---:|---|
| 500 | 500 | 1,000 | 303 |
| 2,000 | 2,000 | 4,000 | 303, 304, 305 |
| 8,000 | 8,000 | 16,000 | 303 |

Ten runs, each initialized independently from the same release checkpoint;
larger budgets do not continue training from smaller budgets. Seed before LoRA
initialization and sampling. The arms match positions and optimizer exposure,
**not distinct-image count**: both-parity uses twice as many rendered views, with
9,000 presentations in either arm. Report this cost and realized factor counts
at every budget; the smaller random prefixes need not be exactly balanced.

Inject LoRA on CPU before moving the complete model to GPU, then rebuild the
optimizer parameter references. Frozen decoder weights must remain in the
autograd path to encoder adapters. Verify tensor counts, device/dtype, initial
base equivalence, and finite loss/gradients. Do not silently detach the encoder,
reduce batch size/resolution, alter the recipe, or resume from a different run.

## Evaluation and gates

Evaluate every final checkpoint on all **16,000 C1 dev cell-images**, reporting
exact-board and per-square scores per factor-cell × parity and pooled summaries.
Never read final-test images or compute final-test performance. Integrity and
disjointness checks may read the sealed metadata, as in C1.

Compare with the single hash-bound release baseline. Paired bootstrap uses 2,000
resamples, seed **20260913**, with each position group kept intact across the
whole matrix; do not split a group's geometry views into independent strata.
Report cell/parity intervals and exact gains/losses, every seed, and failures.

ChessReD test is the public retention check: 2,129 images through the authors'
loader, annotation hash and 10,800-image source inventory checked before/after.
Require **at least 99.12% exact**, within 0.5 points of the 99.62% reference.
This is a retention threshold, not an untouched external-test or superiority
claim. A receipt's `accepted` flag denotes this retention gate only; development
gains or losses must still be reported. No new minimum dev-gain threshold is
invented after seeing results, and there is no test-driven hyperparameter sweep.

Completed negative results receive receipts with `accepted=false`. Execution
failures remain incomplete, with progress and available adapter-only snapshots;
do not relabel them as successful experiments. Baseline retention failure blocks
the adaptation matrix pending a harness/data investigation. Store the final
adapter before long evaluation, never overwrite the base checkpoint.

## Runtime and stop condition

Run serially on otherwise idle render-host GPU 0, using script-backed named tmux sessions.
Check competing GPU processes during execution; never terminate other jobs.
Preserve all source/input hashes and artifact-hashed completion receipts, and
post one-line statuses when receipts land. Stop after the ten fixed cells; no
additional search, selector work, model publication or final-test opening.
`release_ready=false` pending the separate licence review. Timing comparisons
belong to C4, not these training/evaluation wall times.
