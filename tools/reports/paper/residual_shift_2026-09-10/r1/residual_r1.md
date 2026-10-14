# Residual recognition shift on canonical a1-dark boards

Fixed historical confirmation cohort: 288 boards, 48 per geometry. Only the
unchanged release model's `parity` condition is analyzed; the historical
`control` condition is a1-light and excluded. No selectors or final-test data.

**Mechanism remains unresolved.** These metrics describe the errors; the
fixed-sample attention maps are descriptive routing evidence, not a causal
intervention or a proof that highlighted pixels caused the predictions.

| Family | Exact boards | All-square % | GT-occupied class % | Binary occupancy % |
|---|---:|---:|---:|---:|
| deplorable | 19/48 | 91.764 | 87.160 | 93.294 |
| killgorack | 16/48 | 92.318 | 87.072 | 94.824 |
| lyricsz | 0/48 | 83.105 | 69.489 | 88.704 |
| omie | 8/48 | 89.290 | 83.115 | 91.081 |
| phuong | 7/48 | 88.021 | 79.152 | 90.755 |
| pjhanzlik | 0/48 | 80.436 | 66.112 | 85.579 |

## Error types and wrong-square histograms

Confusion matrices use GT rows and prediction columns in `.PNBRQKpnbrqk` order.
Full 13×13 matrices and all 65 histogram bins are in `residual_r1.metrics.json`.
Below, `wrong:boards` lists every nonzero histogram bin, not selected examples.

### deplorable
Empty→piece: 88; piece→empty: 118; occupied wrong class: 47.
Histogram: 0:19, 1:6, 2:3, 3:2, 4:2, 5:1, 6:1, 7:1, 8:1, 9:1, 10:2, 16:1, 17:2, 18:1, 19:2, 21:1, 22:1, 23:1.

### killgorack
Empty→piece: 70; piece→empty: 89; occupied wrong class: 77.
Histogram: 0:16, 1:9, 2:6, 3:3, 4:2, 5:1, 6:1, 11:1, 12:1, 13:1, 15:1, 17:1, 22:2, 23:2, 29:1.

### lyricsz
Empty→piece: 155; piece→empty: 192; occupied wrong class: 172.
Histogram: 1:1, 2:4, 3:4, 4:2, 5:7, 6:1, 7:5, 8:3, 9:2, 10:2, 11:1, 13:2, 14:2, 16:1, 17:1, 18:2, 22:1, 24:1, 25:1, 26:1, 27:1, 29:1, 31:1, 34:1.

### omie
Empty→piece: 123; piece→empty: 151; occupied wrong class: 55.
Histogram: 0:8, 1:11, 2:7, 3:4, 4:1, 5:2, 6:1, 8:1, 10:1, 11:2, 12:1, 16:1, 20:1, 21:1, 22:1, 23:1, 25:1, 29:1, 31:1, 33:1.

### phuong
Empty→piece: 127; piece→empty: 157; occupied wrong class: 84.
Histogram: 0:7, 1:5, 2:4, 3:2, 4:2, 5:4, 6:1, 7:1, 9:5, 10:2, 11:1, 12:3, 13:1, 14:3, 15:1, 16:1, 21:2, 22:2, 24:1.

### pjhanzlik
Empty→piece: 194; piece→empty: 249; occupied wrong class: 158.
Histogram: 1:2, 2:6, 3:5, 4:1, 5:3, 6:2, 7:2, 8:5, 9:1, 10:1, 12:1, 13:2, 18:1, 19:3, 21:1, 22:1, 23:1, 25:1, 26:2, 27:2, 28:1, 30:1, 31:1, 34:1, 35:1.

## Fixed attention sample

Ten outcome-independent boards each from lyricsz, pjhanzlik and deplorable;
all 64 final-decoder-layer, head-mean query maps per board are retained.
Square model-input sheets preserve the 644×644 squash; 16:9 source sheets
stretch those same maps back to 1920×1080. Heat colours are normalized
to each query map's own maximum for display, not comparable absolute saliency.
Selection: `residual_r1.attention_selection.json`; raw maps/logits:
`residual_r1.attention_raw.npz`; all sheet hashes and forward-equivalence
checks: `residual_r1.contact_sheets.json`.

No model selection, corrective rule, or causal conclusion follows automatically.
`release_ready=false`.
