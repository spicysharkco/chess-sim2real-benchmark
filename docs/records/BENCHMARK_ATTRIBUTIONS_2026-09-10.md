# Benchmark asset attribution and transformation record

Current attribution record for the eight-family benchmark-data lineage.
The Phuong render-attribution gate is closed by the author decision below;
publication and model distribution remain separate.
The paired source-file hashes and output-family mapping are in
`BENCHMARK_ASSET_INVENTORY_2026-09-10.json`. This package ships third-party piece meshes and
textures under `rendering/assets/` (original files or converted geometry) and the vendored
Three.js r128 files `rendering/three.min.js` and `rendering/GLTFLoader.js`; each keeps the terms
recorded here and in `LICENSE.md`. No third-party model weights are bundled.

| Family | Creator / source | Stated terms | Changes in this project |
|---|---|---|---|
| polyhaven | [Riley Queen](https://polyhaven.com/a/chess_set) | CC0 | Selected piece meshes, shared grid-fit scaling; procedural replacement board and parameterized palette/camera. |
| deplorable | [Deplorable Mountaineer](https://deplorablemountaineer.itch.io/chess-board-and-pieces) | CC0 | original CC0 Deplorable Mountaineer Staunton meshes; native Blender triangles and source smooth normals; shared max-width .82 scale; base aligned at y=0; no silhouette jitter |
| killgorack | [KillGorack](https://opengameart.org/content/chess-board-pieces) | CC0 | six original white meshes reused both neutral sides; native Blender triangles/corner normals; shared max-width .82 scale |
| lyricsz | [Lyricsz](https://opengameart.org/content/stylized-chess-pieces) | CC0 | original black and white GLBs; one shared source scale; upright Y-up bases retained |
| omie | [Omie’s Assets](https://omies-assets.itch.io/3d-chesscheckers-set) | CC0 | white artist meshes for both neutral sides; preserve all source corners and normals; shared scale and base alignment; no training use |
| phuong | [Phuong2647](https://blendswap.com/blend/29244) | Source: CC-BY (version not displayed); package attribution: CC BY 4.0 terms | original six white glTF meshes and node transforms; native normals; shared max-width .82 scale; no silhouette jitter |
| pjhanzlik | [pjhanzlik](https://opengameart.org/content/chess) | CC0 | original white meshes for both sides; translate base minY to zero; preserve artist proportions |
| nailfighter | [Nailfighter](https://nailfighter.itch.io/lowpoly-chess-set) | CC0 | artist FBX meshes; King identity and other meshes baked model transforms; shared maximum-width .82 normalization; no silhouette jitter |

All families are instantiated at generated positions, under project-defined cameras,
lighting and board colours; the two board parities are a project transformation.
Source geometry is normalized as stated above and may receive replacement materials.
These changes and use of an asset do not imply endorsement by its creator.

**Phuong2647:** Chess set by Phuong2647
([original source](https://blendswap.com/blend/29244)), **CC-BY** as declared by
its creator; geometry converted to glTF and rescaled for rendering. The original
six white meshes and node transforms are retained, with source normals,
shared grid-fit scaling, procedural board, camera, lighting and material changes.
The package applies [CC BY 4.0 attribution terms](https://creativecommons.org/licenses/by/4.0/)
(credit, licence link and indication of changes) under the author decision in
[`PHUONG_DECISION_2026-09-14.md`](PHUONG_DECISION_2026-09-14.md).
This closes the render-attribution gate; it is not a claim that the creator's
unspecified upstream version was recovered. If the creator specifies a version,
update this attribution as directed by that decision; no image changes are required.

For CC0 sources, attribution is kept as provenance even where it is not a licence
condition. The [CC0 legal tool](https://creativecommons.org/publicdomain/zero/1.0/)
does not remove unrelated third-party rights or imply endorsement.

A source-code bundle containing Three.js r128 or GLTFLoader must retain the
upstream copyright and [MIT licence](https://github.com/mrdoob/three.js/blob/r128/LICENSE).
This record is not a substitute for including the actual required notices in a bundle.
An image/label-only bundle need not contain the renderer runtime.

`release_ready=true` for the benchmark-data asset-licensing scope, as recorded in
`PHUONG_RELEASE_STATUS_2026-10-12.complete.json`. No family remains asset-gated
under the author's decision. The exported licence map and eight required
notices were verified by the packaging run; they must remain hash-bound in each
new export. A clean-package reproduction smoke test and the author's publishing
go-ahead remain outside this scope. Base-model and adapter weights remain withheld.
