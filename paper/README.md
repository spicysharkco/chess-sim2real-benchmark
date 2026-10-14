# Receipt-backed paper

Compile from this directory:

```sh
SOURCE_DATE_EPOCH=1789084800 tectonic -X compile main.tex
```

The source is a neutral article. All figures are PDF or PNG; no shell escape or
custom class is required.

Regenerate tables with `python3 build_tables.py`. Regenerate print plots from the
repository root with Matplotlib 3.11.1, the version in `records/figures.complete.json`:
`uv run --no-project --with matplotlib==3.11.1 python paper/build_figures.py`. The five
print plots rebuild byte for byte. The script then builds the attention excerpt, which
reads `tools/reports/paper/attention_full_2026-09-10/` (61 files, 54 MB). That folder is
not in the public package, so the script stops there and `records/figures.complete.json`
is not rewritten. Compile the vector schematic with
`SOURCE_DATE_EPOCH=1789084800 tectonic -X compile generator.tex` from `figures/`. All
table cells carry source comments. The figure data, print layout and table outputs have
separate hash receipts in `records/`.

`evidence/index.json` records the hash of each sanitised copy. An embedded digest
stays only if it is the hash of a file in this package or a public digest: of an
image, model weights, a third-party file, a binary output of a run (an array archive
or a PDF), or a position or label the benchmark publishes. Every other digest reads
`[redacted: withheld digest]`, because such a digest would let a reader test guesses
for a redacted value. Evidence includes summaries and preregistrations, not
training/test images or model weights. The attention examples are receipt-bound
descriptive overlays, not causal explanations.
