# pjhanzlik chess set — CC0 1.0

A genuine, artist-authored family chessboard model by **pjhanzlik**, staged as a
third geometry holdout source.  It is not Poly Haven's `chess_set`, not the
Deplorable Mountaineer Staunton set, and not procedural geometry.

## Provenance and license

- Source landing page: <https://opengameart.org/content/chess>
- Direct archive downloaded 2026-09-06: <https://opengameart.org/sites/default/files/chess_3.zip>
- Source author: pjhanzlik (published 2022-12-19)
- Source page classifies the work as **3D Art** under **CC0** and says the zip
  contains the glTF version for game engines.
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
  CC0 permits copying, modification, distribution, and performance, including
  commercial use, without permission; no attribution is legally required.

## Staged asset

`Chess.glb` is the sole member of the official archive, a glTF 2.0 binary
(GLB) exported by `Khronos glTF Blender I/O v3.4.50`. It requires **no glTF
extensions, Draco decoder, textures, or external buffers**.

The GLB contains complete black and white models, including named meshes:

| FEN type | white mesh | black mesh | unscaled local dimensions (X x Y x Z) |
| --- | --- | --- | --- |
| P | `White_Pawn` | `Black_Pawn` | 2.20 x 3.675 x 2.20 |
| R | `White_Rook` | `Black_Rook` | 2.00 x 4.05 x 2.00 |
| N | `White_Knight` | `Black_Knight` | 2.00 x 4.655 x 2.502 |
| B | `White_Bishop` | `Black_Bishop` | 1.70 x 5.00 x 1.70 |
| Q | `White_Queen` | `Black_Queen` | 2.30 x 5.45 x 2.30 |
| K | `White_King` | `Black_King` | 2.30 x 6.464 x 2.30 |

## Renderer integration notes (observed from the staged GLB)

- glTF is right-handed and Y-up. Piece meshes are already upright along +Y.
  Their horizontal axes are X/Z. The board's original squares are roughly
  3.0 source units wide (piece-center spacing); it should be omitted because
  the renderer supplies the board.
- Nodes use board-placement translations (piece bases land around Y=5.47–5.52)
  but the piece mesh coordinates are local. Clone a piece mesh rather than an
  original scene node; then translate each clone by `-minY` (0 for King/Queen,
  about +0.05 for Pawn/Rook/Knight/Bishop) to put its base at Y=0 before one
  shared renderer scale.
- White/black meshes are distinct material variants. If using one neutral mesh
  plus renderer materials, use the named white mesh for P/R/N/B/Q/K. The black
  knight is mirror-opposed on Z relative to the white knight; preserve a chosen
  color's local orientation or rotate intentionally.
- Meshes retain artist-authored smooth geometry and have no animation. The
  original material names include Black Felt, Brass, and Chrome; there are no
  embedded image textures.
