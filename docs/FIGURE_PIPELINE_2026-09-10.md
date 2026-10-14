# Receipt-driven paper figures — CPU-only protocol

Create five PDF/CSV pairs from existing report JSON/JSONL and receipts only:
adaptation curve, paired-cell heatmap, historical routed-pipeline/member table,
isolated timing table, and historical normalized corner-error/elevation figure.
No images, per-frame manifests, predictions, weights, inference or new scoring.

Inputs are explicitly hash-pinned in `tools/paper/figure_inputs_2026-09-10.json`.
The LoRA and timing summaries must match their original completion receipts.
The four historical routing summary JSONL files and v1 corner-summary JSON lack
run completion receipts: bind their current bytes, disclose that provenance gap,
and do not manufacture retrospective experiment receipts or confidence intervals.

- Adaptation: both arms in separate panels; dark/light dev exact accuracy, all
  training seeds as points and mean connectors. Twin axis shows positive ChessReD
  exact loss versus the release baseline and the actual locked retention floor.
  Available position groups are not realized distinct-seen counts; all ten
  adapted points failed retention. No seed selection or new performance gate.
- Heatmap: same 20 factor cells × two parities, release and 8k both-parity panels,
  common 0–100 exact-board scale. CSV also retains per-square accuracy and n.
- Routing table: all 18 historical splits, full/raw/warp-high/warp-low exact
  accuracy and full-minus-raw delta. This is a pipeline/member contrast, not an
  isolated causal estimate of routing; omit invalid shared-device v1 latency.
- Timing: all 15 verified systems, median/p90, fixed 500-image selection. Include
  batch/precision/power and timer-scope notes; timing validity is not retention.
- Corners: four elevation bands; median/p90 error as percent of mean projected
  board edge, conditional on non-divergent usable detections. Export usable,
  not-found and divergent counts; do not treat omitted failures as zero errors.

Use the existing plotting environment, vector PDF output, fixed typography,
colours, ordering and metadata without wall-clock timestamps. Record plotter
versions and font hash. No new project dependency is added. Rebuild in two fresh
output directories and require byte-identical PDF/CSV hashes, then render every
PDF locally for visual QA. Write the completion receipt last, binding input and
output hashes. Refuse overwrite and input drift. `release_ready=false`.
