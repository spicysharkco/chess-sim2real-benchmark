# Benchmark asset attribution and transformation record

Prepared for the eight-family C1/data lineage; **not a release licence or authorization**.
The paired source-file hashes and output-family mapping are in
`BENCHMARK_ASSET_INVENTORY_2026-09-10.json`. No third-party model/data bytes are bundled here.

| Family | Creator / source | Stated terms | Changes in this project |
|---|---|---|---|
| polyhaven | [Riley Queen](https://polyhaven.com/a/chess_set) | CC0 | Selected piece meshes, shared grid-fit scaling; procedural replacement board and parameterized palette/camera. |
| deplorable | [Deplorable Mountaineer](https://deplorablemountaineer.itch.io/chess-board-and-pieces) | CC0 | original CC0 Deplorable Mountaineer Staunton meshes; native Blender triangles and source smooth normals; shared max-width .82 scale; base aligned at y=0; no silhouette jitter |
| killgorack | [KillGorack](https://opengameart.org/content/chess-board-pieces) | CC0 | six original white meshes reused both neutral sides; native Blender triangles/corner normals; shared max-width .82 scale |
| lyricsz | [Lyricsz](https://opengameart.org/content/stylized-chess-pieces) | CC0 | original black and white GLBs; one shared source scale; upright Y-up bases retained |
| omie | [Omie’s Assets](https://omies-assets.itch.io/3d-chesscheckers-set) | CC0 | white artist meshes for both neutral sides; preserve all source corners and normals; shared scale and base alignment; no training use |
| phuong | [Phuong2647](https://blendswap.com/blend/29244) | CC-BY version unresolved | original six white glTF meshes and node transforms; native normals; shared max-width .82 scale; no silhouette jitter |
| pjhanzlik | [pjhanzlik](https://opengameart.org/content/chess) | CC0 | original white meshes for both sides; translate base minY to zero; preserve artist proportions |
| nailfighter | [Nailfighter](https://nailfighter.itch.io/lowpoly-chess-set) | CC0 | artist FBX meshes; King identity and other meshes baked model transforms; shared maximum-width .82 normalization; no silhouette jitter |

All families are instantiated at generated positions, under project-defined cameras,
lighting and board colours; the two board parities are a project transformation.
Source geometry is normalized as stated above and may receive replacement materials.
These changes and use of an asset do not imply endorsement by its creator.

**Phuong2647:** Chess set, [original source](https://blendswap.com/blend/29244),
displayed licence **CC-BY**. The exact version/legal-code link is not yet resolved;
retain the title/creator/source and changes above, and **do not release this family**
until the applicable version/grant is pinned. A generic version choice is not a fix.

For CC0 sources, attribution is kept as provenance even where it is not a licence
condition. The [CC0 legal tool](https://creativecommons.org/publicdomain/zero/1.0/)
does not remove unrelated third-party rights or imply endorsement.

A source-code bundle containing Three.js r128 or GLTFLoader must retain the
upstream copyright and [MIT licence](https://github.com/mrdoob/three.js/blob/r128/LICENSE).
This record is not a substitute for including the actual required notices in a bundle.
An image/label-only bundle need not contain the renderer runtime.

`release_ready=false`: Phuong exact terms, export membership review, and actual
package notices/project-code licensing remain open. Model-weight rights are a separate
gate and do not automatically block a clean benchmark-data-only package.
