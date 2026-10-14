# Board-colour and label conventions — 2026-09-10

Status: source-sample audit; **not a whole-dataset guarantee**. `release_ready=false`.
No source image is embedded or redistributed. Paths below are private custody references.
No recognizer, detector, training, GPU work, or final-test image access was used.

## What this checks

One lexicographically first available image per populated source directory, paired with its
FEN and available native/camera annotation. A *named board corner* is not an image bounding-box
corner. `a1_light` means named corners a1/h8 light and a8/h1 dark; `a1_dark` means the reverse.
A camera rotation does not change this convention. Source parity is not inferred from a model score.

Pixel checks for the procedural/rendered sources use their recorded **projected named board corners**,
not the detector or the cyclic `order_corners` start index. After projective rectification, four
inset pixels per square give a median luminance for odd/even `(rank_index+file_index)` parity.
All 33 render-host own-render/holdout representatives have odd squares brighter by 52–65 intensity levels
(8-bit OpenCV grayscale), agreeing with a1-light; source contact sheets were also inspected.
These diagnostic warps were local analysis only, never written back into any dataset.

**Mirroring versus canonicalization:** `src/prep.py:42-72` copies FEN unchanged and may rotate
the source-corner sequence to put a8 first before a homography. `src/prep_raw.py:18-62` optionally
crops and resizes pixels, copying the existing FEN unchanged. Neither mirrors labels.
`src/train.py:113-129` can apply matched 90-degree image/grid rotations during legacy training;
that is augmentation, not an ingestion reflection. C2 has no such augmentation.
CR360 exposes `image_is_mirrored` metadata as an orientation diagnosis, not a request to mirror FEN.

## render-host source inventory

`Q` = the existing data root on render-host. The populated `Q/synth` inventory has **25 directories**,
not the historically quoted 26; `_incoming` and `_jobs` are staging and excluded.
All 25 plus the provided training source and ten holdout directories are listed.
Prepared raw/crop/warp/mix caches are transformations or unions of these, not new source datasets.

| Source under Q | Representative | Pixels | Named light corners | FEN mirrored at load? | Image SHA256 |
|---|---|---:|---|---|---|
| private train set | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/angles` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/colors_b` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/geometry` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/geometry_b` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/lowangle` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/lowangle_b` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/mixed` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_angles` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_colors` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_colors_glossy` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_geometry` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_geometry_c` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_geometry_d` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_lowangle` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_lowangle_b` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_lowangle_c` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_lowangle_d` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_occlusion_low` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_occlusion_low2` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_a` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_b` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_c` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_glossy` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_glossy_low` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_low` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `holdout/cr360_scheme` | [redacted: private artefact] | [redacted: private artefact] | a8, h1 | No | [redacted: private artefact] |
| `holdout/cr360_yaw` | [redacted: private artefact] | [redacted: private artefact] | a8, h1 | No | [redacted: private artefact] |
| `holdout/ph_colors_h` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `holdout/ph_colors_h_f` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `holdout/ph_geometry_h` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `holdout/ph_geometry_h2` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `holdout/ph_geometry_h2_f` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `holdout/ph_lowangle` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `holdout/ph_lowangle_h` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `holdout/ph_occlusion_h` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |

For the provided training image, [redacted: private artefact]; a1 at the lower left is light.
The source has no camera manifest, so this is direct visual/FEN evidence, not projection evidence.

The two CR360 holdout samples retain canonical FEN unchanged and native source IDs [redacted: private artefact].
Their source convention is also checked against the separately verified native-annotation pilot below.

## Public datasets

`ext` above means the existing external-data root on data-host. File hashes are computed from actual source bytes, not screenshots.

| Source | Representative | Pixels | Named light corners | FEN mirrored? | SHA256 |
|---|---|---:|---|---|---|
| ChessReD | data-host `ext/chessred/images_1024.zip::images/0/G000_IMG000.jpg` | 1024×1024 | a8, h1 | No | `8e38e53767ed0bb468b94c289bf60e21e37ff10aaeb00fb4f61fb3609daf88c4` |
| ChessCog | data-host `ext/chesscog/images/000000.png` | 1200×800 | a8, h1 | No | `71de1b82bdeea4ba4ccf8715474706055a6150dd5c15f24684e1f8c7bd608c52` |
| SLCC | data-host `ext/slcc_train-host/dataset/images/0cdIkt4Q2zI_100879.jpg` | 573×321 | Unresolved | No | `f2bf0330c8f845a589c87deb6bba92d3a3bbf7af33860d54e5390225d40761fa` |
| CVChess | data-host `ext/cqdata/cvchess/images/IMG_6648.jpg` | 4284×4284 | a8, h1 | No | `7ddf7e9247ea6f6b0daa2fefd2dcb988d063ba6050edc11e97152af6584ac611` |
| CR360 | render-host `Q/ext/chessrender360_verified_pilot/images/000000.png`, native `rgb/rgb_0.jpeg` | 1024×1024 | a8, h1 | No | `aeb4d65988646d9715049b724fa010dbd87db2a08c158fc61c5b8f99ab082166` |

- **ChessReD:** Printed coordinates; annotation image 0; no image/FEN reflection in loaders/chessred.py:57-81,170-221.
- **ChessCog:** Native FEN and camera-side mapping; loaders/chesscog.py:329-349,419-432 preserves source pixels/FEN.
- **SLCC:** No printed coordinates in this sample; no independently anchored named corner. loaders/slcc.py:29-51 copies gt_fen unchanged. Do not generalize that all public sources are a1-dark.
- **CVChess:** Printed coordinates; upstream minimal_reproduction/data.py:149-163 copies gt_fen unchanged.
- **CR360:** native semantic annotation scores the non-mirrored orientation 30/30 pieces, versus 24/30 under reflection; `image_is_mirrored=false`. Source corner names `[h1,a1,a8,h8]` correspond to the stored cyclic quad; visible h1/a8 squares are light. `loaders/chessrender360.py:267-299,373-400` preserves canonical FEN. Do not use the earlier unverified prototype as evidence.
- Pinned upstream `2a9f3c00619133541c65de4315149037b5dece80` `minimal_reproduction/data.py:69-86` expands FEN in a8-to-h1 order, without reflection. ChessReD/Cog/CVChess/SLCC readers do not undo a physical parity convention. Visualization-only flipped display is not dataset ingestion.

## Additional rendered-source inventory

Read-only data-host enumeration found these additional populated source directories. `_failed*`,
`_staging`, `_jobs`, and `_backups` are not counted as distinct released sources. All 15
representative pixels and FEN/metadata were fetched and hashed; none was rendered or modified.

| Source under data-host data root | Representative | Pixels | Named light corners | FEN mirrored at load? | Image SHA256 |
|---|---|---:|---|---|---|
| `synth/mixed_data-host` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_angles_data-host` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_colors_data-host` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_geometry_data-host` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_lowangle_b_data-host` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_lowangle_c_data-host` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_lowangle_data-host` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_blender` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_blender_low` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_blender_occl` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_blender_occl2` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_blender_occl3` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_blender_occl4` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_blender_occl4b` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |
| `synth/ph_staunton_blender_shape3` | [redacted: private artefact] | [redacted: private artefact] | a1, h8 | No | [redacted: private artefact] |

`ph_staunton_blender_occl4` and `occl4b` share the same representative image hash and FEN;
these are retry/copy lineages, not independent evidence. The `_data-host` families also overlap
historical lineages; counting directories is not counting independent distributions.
Blender and Three.js representatives agree on a1-light despite differing lighting.
Renderer evidence: `rendering/renderer*.html` board-square expressions and
`rendering/render_blender.py` board material assignment; no label reflection in rendering output.

## Boundaries and remaining uncertainty

This table does not certify that every image within a source uses one convention. A single
sample is deliberately the requested scope. In particular, unconstrained video/photographic
corpora may include mirrored footage or wrongly set-up physical boards. Such possibilities
cannot be ruled out by a loader that merely preserves a FEN.

The C1 v2 and C2 adaptation pools explicitly contain **both** declared parities without changing
FEN labels. Their complete paired metadata and image-hash integrity audits are in the C1/data
receipts; no v2 final-test pixels were viewed for this table. No historical image was recoloured.
