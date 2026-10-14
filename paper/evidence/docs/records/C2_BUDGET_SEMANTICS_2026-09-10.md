# Interpreting the fixed C2 data budgets

The registered budgets are **available position groups**, not a guarantee that
every group/view is presented to the optimizer. The two arms share the allowed
position pool, not necessarily the exact realized sequence of distinct groups.
Each run performs 1,500 × 6 = **9,000 image presentations**.

In particular, the 8,000-group both-parity pool has 16,000 available images;
9,000 presentations cannot cover all of them. Dark-only has 8,000 available
images. Shuffling, batch dropping, and the fixed step cap determine realized
coverage. The fit receipt's `distinct_training_images` is the **available dataset
size** (`len(selected)`), not a logged distinct-seen counter. Do not report it
as measured exposure, or describe all budgets as full training epochs.

No sampler, step count, arm, recipe, checkpoint, or gate is changed. This note
clarifies reporting only. Exact realized coverage would require a separately
verified sampler replay or a logged index trace; it is not invented here.
