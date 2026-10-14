# Fixed-budget adaptation: convention recovery with failed retention

**Completed:** all ten preregistered runs. **Qualified by the retention gate: 0/10.**
The [full table, every factor cell and paired intervals](lora_curve_2026-09-10/lora_report.md)
are backed by [machine-readable results](lora_curve_2026-09-10/lora_report.json) and the
[verified report receipt](lora_curve_2026-09-10/lora_report.complete.json).
Every saved prediction was independently rescored from its FEN; the report retains
rather than independently recomputes the runs' preregistered bootstrap intervals.

## What the evidence supports

- Dark-only adaptation raises dark-board exact recognition from **15.475%** to
  **64.700%** at 500 available groups, about **77.492%** across three 2,000-group
  seeds, and **79.700%** at 8,000. It produces **zero exact light-convention
  boards at every budget and seed**.
- Training on both rendered parities transfers to the light convention:
  **66.800% dark / 64.200% light** at 500 groups and **80.650% / 79.025%** at
  8,000. At 2,000 groups the three-seed means are **74.833% / 73.375%**.
- The reference checkpoint scores **2,121/2,129 = 99.624%** on ChessReD. Adapted
  checkpoints score **2,098–2,110/2,129 = 98.544–99.108%**. All are below the
  locked **99.12%** floor; no threshold or rounding exception was introduced.
- Low elevation remains difficult after adaptation. The 8,000-group both-parity
  checkpoint scores **14.50% dark / 14.75% light exact** at 8–15 degrees, versus
  **62.75% / 61.25%** at 15–25 degrees. Correcting convention is not a complete
  recognition remedy.

## Claims this does not support

**Not a complete remedy, not zero cost, not a retained-performance SOTA model.**
The arms use the same 1,500 updates and 9,000 image presentations, but both-parity
provides twice as many available rendered views. At 2,000 groups it also reduces
mean dark exact accuracy by about **2.66 points** relative to dark-only. Both arms
lose public-photo exact boards; the observed loss is not proven identical or
caused exclusively by one factor. There is no factorial intervention isolating
all causes of retention loss.

Budgets describe available position pools, not logged unique-seen counts. In
particular, 9,000 presentations cannot cover all 16,000 available images in the
largest both-parity pool. See the [budget semantics note](../../../docs/records/C2_BUDGET_SEMANTICS_2026-09-10.md).
The fixed recipe's failure does not establish that every adaptation method must
fail; no further recipe, seed selection, or remedy search was run.

There are **400 independent dev position groups**, not 16,000 independent
boards; cell-images share positions and some reference scenes. The 100 final-test
groups remain sealed. Synthetic dev gains and a reused public retention set do
not establish unseen-real-world robustness or final-test performance.

## Provenance

- Matrix completed 2026-09-10 17:25 UTC, exit 0, receipt SHA-256
  `[redacted: withheld digest]`.
- All-point report SHA-256
  `[redacted: withheld digest]`.
- Training source stayed frozen at `5ebd1a3`; reporting source at `027d0c7`.
- `release_ready=false`. Timing belongs to C4, not training/evaluation wall time.
