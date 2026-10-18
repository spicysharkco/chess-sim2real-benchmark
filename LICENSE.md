# Licensing

This package contains several classes of material under different licences. The table below
says which licence applies to which path. Full licence texts are in `LICENSES/`.

Copyright 2026 Spicy Shark, except where a third-party notice below says otherwise.

| material | paths | licence | text |
|---|---|---|---|
| Render generator and renderers (project code) | `rendering/*.py`, `rendering/*.js`, `rendering/*.mjs`, `rendering/*.html`, `rendering/holdout_palettes.json`, except the two Three.js files below | PolyForm Noncommercial 1.0.0 | `LICENSES/PolyForm-Noncommercial-1.0.0.md`, notice in `rendering/LICENSE.md` |
| Scoring, diagnostic, figure and export tools (project code) | `tools/paper/`, `src/board.py` | Apache License 2.0 | `LICENSES/Apache-2.0.txt` |
| Benchmark data: manifests, design and freeze records, hash lists, rendered images | `benchmark_v2/`, and rendered images distributed separately | Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0), with the evaluation permission below | `LICENSES/CC-BY-NC-4.0.txt` |
| Reports, receipts, protocols and documentation | `docs/`, `reports/`, `tools/reports/` | CC BY-NC 4.0 | `LICENSES/CC-BY-NC-4.0.txt` |
| Paper build and check scripts, and their tests (project code) | `paper/*.py`, `paper/tests/` | Apache License 2.0 | `LICENSES/Apache-2.0.txt` |
| Paper text, figures, generated tables and PDF | `paper/` except `paper/*.py`, `paper/tests/` and `paper/evidence/` | CC BY-NC 4.0 | `LICENSES/CC-BY-NC-4.0.txt` |
| Package overview, hash list and evidence receipts | `README.md`, `EXPORT_NOTICE.md`, `PACKAGE_INTEGRITY.json`, `paper/evidence/` | CC BY-NC 4.0 | `LICENSES/CC-BY-NC-4.0.txt` |
| Three.js r128 and GLTFLoader (third party) | `rendering/three.min.js`, `rendering/GLTFLoader.js` | MIT, copyright 2010-2021 three.js authors | `LICENSES/MIT-three.js.txt` |
| Piece geometry derived from third-party assets | `rendering/assets/`, `rendering/*_geometry.json` | the source asset's own terms: CC0 1.0 Universal for seven families; phuong is CC-BY as declared by its creator (no version displayed), attributed under CC BY 4.0 terms | `LICENSES/CC0-1.0.txt`, `LICENSES/CC-BY-4.0.txt`, `docs/records/BENCHMARK_ATTRIBUTIONS_2026-09-10.md` |
| Licence texts and notices | `LICENSE.md`, `LICENSES/`, `rendering/LICENSE.md`, `src/LICENSE.md` | licence texts (verbatim copies) and notices, each under its own terms | the files themselves |

Anything in this package not covered by a row above is all rights reserved.

## Material that is not in this package

- **ChessQueries code and release checkpoint.** These are third-party works under the
  PolyForm Noncommercial License 1.0.0. They are not redistributed here. The tools in
  `tools/paper/` expect the user to obtain them from the upstream repository and to accept its
  terms.
- **Adapter weights** trained in this project. These are derived from the ChessQueries release
  checkpoint and are withheld. If they are released, they will carry the PolyForm Noncommercial
  License 1.0.0 together with the upstream ChessQueries required notice, and they cannot be used
  commercially under any licence from this project.

## Attribution

For the benchmark data, cite the paper and credit:
"Chess-position recognition benchmark v2 by Spicy Shark, licensed under CC BY-NC 4.0."
Images of the phuong piece family must also credit the asset's creator as listed in
`docs/records/BENCHMARK_ATTRIBUTIONS_2026-09-10.md`: "Chess set by Phuong2647
(https://blendswap.com/blend/29244), CC-BY". The source page declares CC-BY without a version;
this package applies the CC BY 4.0 attribution terms to that family (author decision,
2026-09-14, `docs/records/PHUONG_DECISION_2026-09-14.md`).

## Evaluation permission

In addition to the rights that CC BY-NC 4.0 grants, the copyright holder permits anyone,
commercial organisations included, to:

1. use the benchmark data and the rendered images distributed by the copyright holder to
   evaluate a model's performance on the benchmark; and
2. publish the resulting scores, tables and figures in any medium, commercial ones included,
   with the credit line in "Attribution" above. Figures under this permission must not contain
   benchmark images.

This permission does not cover training or fine-tuning a model on the benchmark data or images,
redistributing them, or including them in a product or service. It does not extend the licence of
the render generator. Noncommercial uses remain under CC BY-NC 4.0, and the generator under its
own licence. For commercial uses beyond this permission, contact spicysharkco@gmail.com.

## Commercial use

The benchmark data, the rendered images, the documentation and the paper are licensed for
noncommercial use only, except for the evaluation permission above. The render generator is
licensed for noncommercial use only. For a commercial licence or custom rendered datasets,
contact the copyright holder at spicysharkco@gmail.com. The Apache-2.0 tools may be used
commercially under their own terms.
