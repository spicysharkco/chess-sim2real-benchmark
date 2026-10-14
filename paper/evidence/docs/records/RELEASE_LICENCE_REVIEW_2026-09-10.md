# Release licence review — 2026-09-10

**Decision: `release_ready=false`.** This is an evidence inventory and release
gate, not legal advice or a blanket approval. It covers the frozen C1/C2 visual
lineage plus named data/model inputs. No receipt, dataset, asset, or upstream
licence was changed by this review.

## Scope and evidence boundary

The C1 input custody is now independently checked, not inferred from image
hashes: `benchmark.complete.json` binds `design.complete.json`, which binds
`design.plan.json` and its `source_sha256` map. On render-host, all **80 source files,
including 39 asset-package files**, matched the clean frozen source clone at
`01c52fb9a69a5b36759b310ae18d26dd8ce662c1`. The eight-family mapping covers
20,000 C1 and 16,000 adaptation cell-images; metadata-only inspection touched
no final-test pixels. See `BENCHMARK_ASSET_INVENTORY_2026-09-10.json` for the
receipt chain, exact per-file hashes, descriptors, and output-family counts.
Original archive hashes quoted below remain descriptor provenance unless the
archive itself was separately verified; they are distinct from the freshly
rehashed converted/runtime files. The frozen C2 result also binds every model
input it consumed, including C1 manifests and image/label hashes.

"Rendered derivative" means a rendered PNG/FEN/metadata output using a source
asset, not a distribution of its mesh/archive. Whether a trained weight is an
adapted copyright work is not settled by these records; it is never silently
treated as licence-free.

### Actual renderer inputs outside the eight geometry families

| Input | Evidence / status | Release condition |
|---|---|---|
| Board, colours, camera, direct/hemisphere lighting | Renderer source constructs these procedurally; no standalone board texture or HDRI is loaded. The original Poly Haven board node is deliberately excluded. | Project-origin material only, subject to confirming authorship of the renderer code. |
| Poly Haven glTF textures | The glTF package contains board/piece texture files. The C1 renderer replaces selected piece mesh materials and builds the board procedurally, but the loader package remains the same CC0 asset. | Covered by the Poly Haven CC0 record below; do not claim that no third-party texture bytes were present unless a runtime dependency inventory proves it. |
| HDRI | No C1 renderer source reviewed here loads an HDRI/environment texture. The KillGorack source page attributes an HDRI, but that source HDRI is not itself a C1 render input. | No HDRI may be added to an export without a separate source/rights record. |
| Three.js r128 (`three.min.js`) and `GLTFLoader.js` | Vendored `three.min.js` identifies MIT; local SHA-256 `9274bbcec8d96168626c732b5d31c775aa8cfb7eaa0599bec0c175908a2c1ce2`. The upstream r128 licence is MIT. `GLTFLoader.js` local SHA-256 is `5c15967ba830918a9caea6338712c994c354bccd4edc4569bde411c3ec06a3e6`. | If either file is shipped, retain its copyright/MIT notice and include the MIT licence. A rendered-image-only export need not bundle either file. [Three.js r128 licence](https://github.com/mrdoob/three.js/blob/r128/LICENSE) |
| glTF/FBX/Collada conversion | These are file formats/conversions, not a licence grant. Converted JSON/GLB geometry remains tied to the source asset's licence. | Keep source URL, author, source/converted hash, and transformation note. |

## C1 geometry-asset register

All eight families occur in C1; `kings_gambit` and `shiva` are excluded. The
converted/runtime hashes are now verified through the nested C1 custody chain
above; source-archive hashes retain their explicitly separate provenance.

| Family / actual source | Author and primary source | Asset identity and digest available now | Licence and source grant | Rendered-derivative / notice requirement | Release state |
|---|---|---|---|---|---|
| `polyhaven` — Chess Set | Riley Queen; [Poly Haven asset page](https://polyhaven.com/a/chess_set) | `chess_set_1k.gltf`: local SHA-256 `3e21ecd0c2ffb703dfdb3f3ea7a379c1dbd4ec513ba2cb8e6cb1e5f276d7eb91`; official download manifest MD5 `ceac1016348f797f3db112e3e70a96a4`. The converted/runtime file is bound through the nested C1 source inventory. | **CC0 1.0 Universal**. The asset page identifies the author and CC0. CC0 permits copying, modification, distribution, and commercial use, subject to third-party rights/no-endorsement caveats. | Source meshes/textures and rendered derivatives may be redistributed under CC0 evidence. Attribution is not a CC0 condition, but retain author/source/provenance in the release record. [Asset](https://polyhaven.com/a/chess_set), [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | Conditional allow after export provenance is complete. |
| `deplorable` — Chess Board and Pieces | Deplorable Mountaineer; [author asset page](https://deplorablemountaineer.itch.io/chess-board-and-pieces) | Converted `chess_pieces.json` SHA-256 `489e258aeb49b0d7e5dafd3ba56b257e709d5a6eefa38b7686ed84c823972430` (locally verified); source archive SHA-256 `585a53483ced76b3627941e5594c10a66eadc33d44826024c0b38d9f28164422`. | **CC0 1.0 Universal**; the page separately calls its code MIT and its asset CC0, so asset terms—not code terms—govern the mesh. | Same CC0 result as above. Record conversion from the source Collada scene and material/board omission; do not imply author endorsement. | Conditional allow after export provenance is complete. |
| `killgorack` — Chess board & pieces | KillGorack; [OpenGameArt source page](https://opengameart.org/content/chess-board-pieces) | `killgorack_chess_20260906/chess_pieces.json` SHA-256 `6ed3b3ee918939b9a5845045dad73274ead13c86369f5455fe7e033a41c3257c`; source archive SHA-256 `21f538bfc320064aceb66a4ff63c81afbcdc2f95f2563f32a0e95763819a065e`. | **CC0** on the author upload. The source page says its HDRI is Poly Haven and its other textures are author-created; that does not license an HDRI not actually used in C1. | CC0 permits source/derivative redistribution; preserve the source attribution record despite no mandatory CC0 attribution. Any later use of the cited HDRI must be separately pinned/checked. | Conditional allow after export provenance is complete. |
| `lyricsz` — Stylized Chess Pieces | Lyricsz; [OpenGameArt source page](https://opengameart.org/content/stylized-chess-pieces) | Descriptor aggregate SHA-256 for sorted GLB filename/checksum lines: `d70b99ed7711fcf65cfa0d09c03685a245befa3cf1d70ffbdd79df55668a4ffa`; individual GLB hashes are in `rendering/lyricsz_geometry.json`. | **CC0**; source page identifies Lyricsz and says attribution is not mandatory. | CC0 permits the meshes and rendered derivatives; retain the author/source link and list the GLB conversion/recolouring as provenance. | Conditional allow after export provenance is complete. |
| `omie` — Low Poly 3D Chess/Checkers Set | Omie's Assets; [author asset page](https://omies-assets.itch.io/3d-chesscheckers-set) | `omie_chess_cc0_20260906/chess_pieces.json` SHA-256 `8137e1b787ad72fb53120314cf9c776a720e1fd19c7d7786044dc9b028d2b44e`; source archive SHA-256 `c6737d2f0d8ce5e8011f01bf801346250d61ce060013982722aff3d3f029ff7b`. | **CC0 1.0 Universal**; page states personal/commercial use and no attribution required. | CC0 permits source/derivative redistribution. Retain provenance and note that C1 uses the identified converted mesh rather than treating the commercial-use statement as a licence substitute. | Conditional allow after export provenance is complete. |
| `phuong` — Chess set | Phuong2647; [BlendSwap source page](https://blendswap.com/blend/29244) | `phuong_chess_20260907/chess_pieces.json` SHA-256 `452c1bfde52cdb54db850353fda91b60b3f9f29c06610a9b142d2c77e960fc91`; source archive SHA-256 `1b7c9f03e0bfc9836e9c775bb04dda390cd1d860d5aeb31e7cef935cfcd2a9c4`. | Source page labels **CC-BY**, but its displayed licence field does not specify a version and no exact asset `COPYING` file was verified. The Castle Engine zip is a distribution path, not independent proof that downstream exports lose original duties. | Preserve creator credit, title/source URL, the displayed `CC-BY` label, and the conversion/normalisation/recolour/render change record. **Do not state CC BY 4.0 or select a version-specific legal-code URL** until exact terms are recovered. | **Conditional only; version/grant gate open.** |
| `pjhanzlik` — Chess | pjhanzlik; [OpenGameArt source page](https://opengameart.org/content/chess) | `pjhanzlik_chess_cc0_20260906/Chess.glb` SHA-256 `1678dd307f2d944d39786739448bca631cfdf0a13bdbb439954fb749b4a35a9a`. | **CC0** on the author upload. | CC0 permits source/derivative redistribution. Retain source/author record and transformation/recolour note. | Conditional allow after export provenance is complete. |
| `nailfighter` — LowPoly Chess Set | Nailfighter; [author asset page](https://nailfighter.itch.io/lowpoly-chess-set) | `nailfighter_chess_cc0_20260906/chess_pieces.json` SHA-256 `0d96781041e87b0330f7646cb0720f63acbb42a8f6806d935d81dace20a6951a`; source archive SHA-256 `822ea937524c420fc60a54ff97ecd7aff498a9993dbc071f92e363e1bc0c5247`. | **CC0 1.0 Universal**; page also says personal, educational, and commercial work. | CC0 permits source/derivative redistribution. Preserve asset/source/provenance record. | Conditional allow after export provenance is complete. |

## Dataset and position-source register

FEN-only pools are not automatically visual-data licences. They are listed for
training-lineage and exclusion transparency; no raw pool is approved for export
solely because its positions are used to reject overlap.

### Live-source verification ledger

**Checked 2026-09-10:** Poly Haven
[`chess_set`](https://polyhaven.com/a/chess_set) (CC0, Riley Queen);
[Deplorable Mountaineer](https://deplorablemountaineer.itch.io/chess-board-and-pieces)
(asset CC0); [KillGorack](https://opengameart.org/content/chess-board-pieces)
(CC0); [Lyricsz](https://opengameart.org/content/stylized-chess-pieces) (CC0);
[Omie's Assets](https://omies-assets.itch.io/3d-chesscheckers-set) (CC0);
[Phuong2647](https://blendswap.com/blend/29244) (**displays `CC-BY`, version
not displayed**); [pjhanzlik](https://opengameart.org/content/chess) (CC0); and
[Nailfighter](https://nailfighter.itch.io/lowpoly-chess-set) (asset CC0).

**Checked 2026-09-10:**
[ChessReD v2](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2)
(CC BY-NC-SA 4.0);
[ChessCog OSF API](https://api.osf.io/v2/nodes/xf3ka/) (`license: null`);
[Lichess puzzle database](https://database.lichess.org/#puzzles) (CC0);
[ChessQueries pinned README](https://github.com/JSeytre/chessqueries/blob/2a9f3c00619133541c65de4315149037b5dece80/README.md)
(code/weights PolyForm NC; annotations CC BY-NC; source frames separately
reserved); and [CVChess repository](https://github.com/Luthiraa/CVChess)
(no declared repository licence/asset grant found).

**Historical / not live-complete:** the ChessRender360 source is recorded as
[Zenodo 13356818](https://zenodo.org/records/13356818), version 1.0, CC BY 4.0
in the local loader provenance, but the record's licence metadata did not return
in this review's live API attempt. The exact Zenodo version/licence field must
be rechecked before a CR360 export. Provided-private and own-render sources have
no public source page to live-check; their status follows the custody record,
not a web assertion.

| Source | What it supplies / source evidence | Licence signal and distribution consequence | Current release state |
|---|---|---|---|
| Provided private train | Private images/labels; no public distribution authorization was presented to this review. | No release grant established. Do not export raw pixels/labels, derived caches, samples, or weights trained on it until the rights holder provides a written scope that covers the proposed export. | **Deny.** |
| Own v1/v2/adaptation renders | Project-created camera, board, palette, FEN and render metadata; geometry is inherited from the asset register above. | Original work does not erase third-party asset terms. C1/C2 outputs can only enter the conditional visual allowlist when the exact source-family record, hash, and applicable notices travel with the export. | **Conditional only.** |
| ChessReD v2 | Masouris, *Chess Recognition Dataset (ChessReD)*, [4TU version 2](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2). | **CC BY-NC-SA 4.0**: sharing/adaptation requires attribution, licence link, change indication, non-commercial use, and same-licence treatment for adaptations. The source's own licence does not resolve whether a particular trained model is an adapted work. [Terms](https://creativecommons.org/licenses/by-nc-sa/4.0/) | Raw/converted photos **not in a general release**; any licence-compatible non-commercial data export and model claim requires counsel/rights review plus attribution/SA plan. |
| ChessCog | 4,888 rendered PNG/JSON records from [OSF `xf3ka`](https://osf.io/xf3ka/). | The OSF API checked for this record reports `license: null`; the upstream code repository's MIT licence is not a data licence. No permission is inferred from public access. | **Deny** raw/converted data, samples, and dependent release artifacts pending an explicit dataset grant. |
| SLCC | ChessQueries `slcc-v1` annotations and locally reconstructed frames from named YouTube broadcasts. [ChessQueries README](https://github.com/JSeytre/chessqueries/blob/2a9f3c00619133541c65de4315149037b5dece80/README.md) | The upstream release distinguishes **CC BY-NC 4.0 annotations** from source broadcasts/reconstructed frames, which remain with their rights holders. Annotation terms do not grant broadcast-frame redistribution. | **Deny** reconstructed image pixels and image-derived export. Annotation-only reuse remains conditional on exact CC BY-NC compliance and scope review. |
| CVChess | CVChess repository/image collection; [upstream repository](https://github.com/Luthiraa/CVChess). | Repository metadata shows no declared licence and no licence file; no separate image-data grant was found. | **Deny** raw/converted images and dependent public release artifacts pending explicit author/rights-holder permission. |
| ChessRender360 / CR360 | Kojić, version 1.0, [Zenodo record 13356818](https://zenodo.org/records/13356818); local loader records official archive MD5 `aa8cc34a38b32d5eee6c9129c65023ab`. | **CC BY 4.0** per the recorded versioned source. Sharing/adaptation needs creator credit, licence link, and changes indication; the record must be rechecked against the exact exported version. [CC BY terms](https://creativecommons.org/licenses/by/4.0/) | Conditional data/export allow after a versioned record, attribution, change log, and model-lineage review. |
| Lichess puzzles / 6,100,960-row FEN snapshot | Position strings used as an exclusion and historical-prior source; [Lichess open database — puzzles](https://database.lichess.org/#puzzles). | The open-database page expressly releases its exports under **CC0** and says they may be downloaded, modified, and redistributed. The source snapshot must still be identified by its exact download/retrieval date and file hash; the live page's current count is not the historical snapshot identity. | Raw snapshot export is **conditionally permitted by the stated CC0 grant** once the exact file/hash is identified; it is not required for a C1 benchmark-data export. Exclusion hashes/counts may be published only after confirming they disclose no rows or account-linked data. |

## Code, checkpoint, and learned-artifact register

| Artifact | Primary evidence | Release consequence |
|---|---|---|
| ChessQueries code at `2a9f3c00619133541c65de4315149037b5dece80` | [Pinned README](https://github.com/JSeytre/chessqueries/blob/2a9f3c00619133541c65de4315149037b5dece80/README.md) and [licence](https://github.com/JSeytre/chessqueries/blob/2a9f3c00619133541c65de4315149037b5dece80/LICENSE) identify **PolyForm Noncommercial 1.0.0** and a required copyright notice. | Do not label as MIT/Apache or commercially redistributable. Any copied source needs the PolyForm terms/required notice and a non-commercial scope review. |
| ChessQueries release checkpoint SHA-256 `6151…fc762e5` | Upstream [development guide](https://github.com/JSeytre/chessqueries/blob/2a9f3c00619133541c65de4315149037b5dece80/DEVELOPMENT.md) binds the release-checkpoint digest; README states weights are PolyForm Noncommercial. | Do not redistribute the base checkpoint, bundle it with an adapter, or present a release as unconstrained without PolyForm non-commercial compatibility and required-notice review. |
| DINOv2 base encoder | The upstream DINOv2 code repository is [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE); Meta's `facebook/dinov2-large` model card declares Apache-2.0. The exact `timm` retrieval artifact used inside the release checkpoint was not independently pinned here. | Code/model-card evidence is insufficient to certify the exact inherited checkpoint component. Record the exact upstream weight URL/revision/hash before any separate weight or model-bundle release. |
| C2 LoRA adapters | New tensors are trained against the ChessQueries base and C2 render pool; no independent licence declaration exists yet. | Adapter-only export is **conditional**, not automatically free of base-model or training-data obligations. It needs base/checkpoint compatibility, C1/C2 source-family receipts, and an explicit adapter licence/notice file. |
| Legacy trained ensemble / prior | It has an unresolved private-origin training lineage. | Treat weights, training data, and distilled artifacts as a **separate conservative export-policy hold** until their lineage is inventoried. This is not a determination that reporting independently reproducible aggregate benchmark results copies those inputs. |

## Prospective release allowlist and denylist

### A. Benchmark-data export: C1 images, labels, manifests

This package can be assessed independently of weights, legacy training data, or
the ChessQueries stack because it contains none of them. Its **conditional
allowlist** is the C1 PNGs/labels/non-sensitive manifests and the
project-authored generator, only for the eight tabled source families.

Open gates for this package only:

1. **Inventory prepared and verified:** the machine-readable C1 input register maps every
   family/output to source URL, author, original-descriptor and verified converted
   hashes, renderer revision, and licence status. Preserve this binding in the
   eventual export and check actual package membership against it.
2. **Attribution draft prepared:** `BENCHMARK_ATTRIBUTIONS_2026-09-10.md` carries Phuong2647's creator/title/
   source and conversion/recolour/render change record. Resolve its exact CC BY
   version/grant before exporting that family; do not invent a version. Include
   this record and the applicable notices in the actual package.
3. For generator source that ships Three.js/GLTFLoader, include MIT terms and
   notices; an image/label-only package need not bundle those files.
4. Recheck the eight live asset pages immediately before export and freeze the
   resulting source-page status/date beside the asset inventory.

The C1 data package does **not** depend on resolving unrelated ChessCog, SLCC,
CVChess, private-train, legacy-ensemble, base-checkpoint, or adapter questions.
Its current `release_ready=false` state is justified by the incomplete C1
exact Phuong terms, actual export membership/notices, and project-code licensing
gates above, not by those unrelated sources.

### B. Third-party data/source-asset repackaging

Conditional candidates are CC0 assets and Lichess database rows after exact
file/hash capture, and CR360 after its exact-version licence field is live
verified plus CC BY attribution/change documentation. This is not permission to
bundle any other source by association.

**Denylist for raw/converted third-party source packages:** `kings_gambit`,
`shiva`, ChessCog, SLCC reconstructed broadcast pixels, CVChess pixels, and
provided-private train data. ChessReD requires its own non-commercial,
attribution, and ShareAlike compatibility review before data export.

### C. Learned weights and upstream model/code bundles

This is a separate export class. ChessQueries code/checkpoint require PolyForm
Noncommercial terms/notice treatment; the exact DINOv2/timm source-weight
provenance and any LoRA-adapter terms still need a bounded review. The
private-origin legacy ensemble is held from weight/data/distillation export
until its lineage is known. None of those facts automatically prohibits an
independently reproducible, source-cited benchmark result claim that distributes
no protected bytes.

## Cross-cutting gates that remain open

1. Each proposed bundle must name its artifact class (A, B, or C) and prove it
   excludes inputs outside that class's allowlist.
2. Recheck all live source pages immediately before that bundle's release and
   record retrieval date/version, because web-hosted labels can change.
3. Do not turn a model-dependence uncertainty into a legal conclusion: obtain
   qualified review for copyright ownership, fair use, model-derivative status,
   and jurisdiction-specific rights where those affect the chosen artifact.

Until the relevant gates close, the applicable package remains deny-by-default;
that status is scoped per artifact class rather than a claim that all materials
must be bundled or cleared together.
