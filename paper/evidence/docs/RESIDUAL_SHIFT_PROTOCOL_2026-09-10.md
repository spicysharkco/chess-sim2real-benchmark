# C3 preregistration: residual recognition shift and descriptive attention

This protocol is registered before C3 metrics are computed or attention is
captured. C3 is diagnostic, not a model-selection or mechanism-confirmation
stage. It uses no benchmark-v2 final-test image, score, inspection, timing, or
attention result.

## Frozen population and model output

- Use the completed historical confirmation cohort at
  `$HOME/bench_experiments/codex_transfer_20260910`: 288 position groups,
  six geometry families with 48 groups each, and paired parity renders.
- Analyze exactly the **288 canonical a1-dark** (`condition=parity`) images:
  deplorable, killgorack, lyricsz, omie, phuong and pjhanzlik. Do not analyze the
  byte-identical control here. The historical control renderer assigns the
  light material to its odd `(row + column)` branch; a1 is renderer row 7,
  column 0 and is therefore light. `transfer_cohort.py` creates the `parity`
  renderer by swapping that odd branch to the dark material, so `parity`, not
  `control`, is the a1-dark condition. Before using any rows, require every
  recorded parity-renderer source to equal `swap_parity(control_source)` and
  verify both checkerboard branches explicitly.
- Verify the artifact hashes and bindings in `prepare.complete.json`,
  `render_validation.complete.json` and `confirmation.complete.json`. Rehash
  every selected source image. The prepare manifest, validated confirmation
  inventory and saved predictions must agree by key, FEN and SHA-256.
- Use only rows where `model=default` from the already-saved canonical
  ChessQueries predictions. Selector, rule and adapted-model outputs are not
  C3 inputs. Save a new C3 prediction file containing only those default rows.
- The ground-truth and prediction label order is `.PNBRQKpnbrqk`; all 64
  squares are ordered a8, b8, ..., h8, a7, ..., h1. Pin the release checkpoint
  SHA-256 to
  `6151bdd98fbe25f32080c097eba7ae75808b5615160e4867240931085fc762e5`.

## Locked descriptive metrics

Report overall and separately for all six geometry families:

1. a 13 x 13 count matrix with ground-truth class on rows and predicted class
   on columns;
2. exact 13-class accuracy restricted to ground-truth occupied squares;
3. binary occupied-versus-empty accuracy over all squares, reported separately
   from class accuracy;
4. all-square exact-class accuracy and exact-board accuracy for context; and
5. a 65-bin histogram (0 through 64) of wrong-square counts per board.

These are a characterization of error type and severity. No post-hoc pass
threshold, new selector, corrected prediction, or significance claim is added.
A null, mixed or negative pattern is a valid completed C3 result.

## Outcome-independent attention sample

Before loading the checkpoint, independently select ten canonical images from
each of lyricsz, pjhanzlik and deplorable (30 total). Within a family, rank the
48 records by ascending SHA-256 of
`20260915:group_id:image_sha256`, with record key as a deterministic tiebreak,
and take the first ten. Prediction correctness, confidence and metric outcomes
do not enter selection.

Use the frozen minimal-reproduction model implementation that produced the
saved default predictions; do not substitute the newer public model class or
change the checkpoint architecture/label mapping. Its encoder, query module,
PyTorch decoder, norm and head expose the structural interface consumed by the
visualization helper. Rehash every model/source file recorded by the historical
confirmation receipt before model allocation and after attention capture. Use
the pinned ChessQueries source commit
`2a9f3c00619133541c65de4315149037b5dece80`. Load the same release model and
authors' `eval_transform(644)` as the saved default predictions. The original
1920 x 1080 frame is squashed to 644 x 644, with no crop or augmentation. Run
at most two boards per attention batch on render-host GPU 0 after C4 and after the LoRA
matrix has released it.

Call `chessqueries.viz.attention.extract_attention(model, batch)` and use
`maps.cross_grid(layer=-1, head=None)`. That upstream method averages heads,
drops CLS/register prefix tokens, and renormalizes patch mass. Preserve all 64
query maps in a compressed raw NPZ in fixed a8-to-h1 order. For every board,
create two 8 x 8 contact sheets containing all 64 fixed query overlays: one
interpolates the heat maps to the 644 x 644 model-input geometry, and one
stretches them to the original 1920 x 1080 frame before thumbnailing. Preserve
those geometries in the sheet tiles: 192 x 192 square tiles for the 644 x 644
view and 192 x 108 16:9 tiles for the source-frame view. Neither boards nor
queries may be chosen after viewing results.

Require finite, nonnegative maps whose patch mass sums to one per query. Set
PyTorch float32 matrix-multiplication precision to `high`, matching the frozen
confirmation run. Check that captured logits match an ordinary forward pass
within `atol=1e-4, rtol=1e-4` (the weight-returning attention path may select a
different numerical kernel), a subsequent ordinary
forward remains unchanged, temporary attention-forward overrides are restored,
and selected argmax predictions reproduce the saved default predictions.
Attention is reported only as descriptive localization evidence. It does not by
itself establish a causal model mechanism, and no forced explanation is a
stopping requirement.

## Outputs and stopping rule

The no-clobber C3 prefix contains default-only predictions, JSON metrics, the
pre-inference selection, raw maps/logits, contact-sheet hashes, and a completion
receipt. The receipt binds all input receipts/manifests/predictions, selected
IDs and images, source code, this protocol, pinned upstream attention source,
and the base checkpoint. Contact-sheet images are transitively bound through an
artifact-hashed manifest. Recheck frozen inputs after attention capture.

Also write a Markdown report with all six family metrics, occupancy-versus-class
error counts and every nonzero wrong-square histogram bin. It links the complete
confusion matrices and attention artifacts and states the mechanism remains
unresolved unless separately supported by an intervention. Display heat colours
are scaled to each query map's own maximum, not absolute cross-query saliency.
Require the upstream checkout to be clean at the pinned commit and its
`chessqueries/viz/attention.py` SHA-256 to be
`af091615c10157259ceddcdb0ce6e0e871c74b046b7ef5b87db0c84939ed9d15`
both before and after capture. Snapshot the C3 runner, protocol and imported
paper-tool dependencies before GPU allocation and require those hashes to remain
unchanged before completing the receipt. Refuse to allocate while another GPU 0
compute PID exists; recheck between batches while allowing only the current PID.

Complete once all 288 default predictions and all 30 x 64 attention maps pass
the locked checks, regardless of whether a coherent residual pattern appears.
An execution failure remains incomplete; use a new output prefix rather than
overwriting it. Do not run C3 until the LoRA report and C4 are complete.
`release_ready=false` remains in force pending the separate licence review.
