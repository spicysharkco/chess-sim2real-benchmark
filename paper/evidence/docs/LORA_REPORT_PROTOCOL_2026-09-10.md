# C2 reporting protocol

Registered while the first fixed adaptation point is still running and before
reading any adapted-model evaluation. No change to the training protocol.

After the ten-point matrix receipt lands, `tools/paper/lora_report.py` verifies
every artifact-bound point and independently rescores its saved FEN strings.
It verifies complete 400-position-group × 40-view dev coverage, unchanged GT
and public-test membership, source/recipe identity, and the retention flags.
It reads predictions and receipts only, never images or the GPU.

Report all ten seeds/arms/budgets, both parity scores, every factor cell,
ChessReD retention failures, pooled paired changes and gained/lost cell-images.
Keep the preregistered whole-position bootstrap intervals from the bound run
metrics; do not describe this aggregation as an independent CI recomputation.
At 2,000 groups, show mean and sample SD across all three fixed seeds per arm.
No best-seed selection or unregistered significance threshold.

The report must distinguish matched positions/optimizer exposure from distinct
images (N versus 2N), 400 independent groups from 16,000 cell-images, development
scores from untouched final-test/general-real-world claims, and public retention
from overall superiority. No SOTA or zero-cost invariance claim is automatic.

Completion writes Markdown, machine-readable full metrics, and a hashed receipt.
This report gates the post-matrix order: **C2 report → C4 timing → C3 residual**.
`release_ready=false`; failures remain visible and final-test images remain sealed.
