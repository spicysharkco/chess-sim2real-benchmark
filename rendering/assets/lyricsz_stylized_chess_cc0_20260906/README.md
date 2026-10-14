# Lyricsz stylized chess pieces — CC0 1.0

A fourth, genuine artist-authored geometry family for training renders. This
low-poly stylized set is distinct from Poly Haven's `chess_set`, the Deplorable
Mountaineer Staunton set, and pjhanzlik's family chessboard. It is source mesh
content, not a procedural substitute.

## Provenance and license

- Source page: <https://opengameart.org/content/stylized-chess-pieces>
- Author: Lyricsz; published 2026-03-07
- Official direct archive downloaded 2026-09-06:
  <https://opengameart.org/sites/default/files/chess_source.zip>
- The author page calls it 3D low-poly chess art, says it includes all chess
  pieces with black and white materials, and licenses it **CC0**.
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
  It permits copying, modification, distribution, and commercial use without
  permission; the source says attribution is not mandatory.

## Staged files

The official archive also contains Blender and FBX source. This directory stages
only the exact, unmodified game-ready GLB members (12 individual files) because
they are the smallest directly loadable Three.js assets:

| FEN | white GLB / root node | black GLB / root node |
| --- | --- | --- |
| P | `Pawn_White.glb` / `Pawn` | `Pawn_Black.glb` / `Pawn` |
| R | `Rook_White.glb` / `Rook` | `Rook_Black.glb` / `Rook` |
| N | `Knight_White.glb` / `Knight` | `Knight_Black.glb` / `Knight` |
| B | `Bishop_White.glb` / `Bishop` | `Bishop_Black.glb` / `Bishop` |
| Q | `Queen_White.glb` / `Queen` | `Queen_Black.glb` / `Queen` |
| K | `King_White.glb` / `King` | `King_Black.glb` / `King` |

Every staged file is glTF 2.0 binary (GLB), one node/one mesh, with no required
or used extensions, no Draco, no external buffers, no image textures, and no
animation. It therefore loads directly via Three.js `GLTFLoader`.

## Orientation and scale notes

- Standard right-handed glTF axes: **+Y is up**. All nodes have no transform;
  source geometry is already upright and its base is at local `Y=0` (queen's
  floating-point minimum is effectively zero). Clone the loaded root or mesh
  directly; no base recentering is required.
- White local dimensions (X x Y x Z): Pawn `0.2525 x 0.6007 x 0.2525`; Rook
  `0.2534 x 0.7039 x 0.2534`; Knight `0.2718 x 0.7182 x 0.3456`; Bishop
  `0.2535 x 0.7466 x 0.2535`; Queen `0.2496 x 0.7761 x 0.2496`; King
  `0.2479 x 0.8918 x 0.2479`.
- Apply one common renderer scale to match a board square. Keep a color's
  original GLB rather than mirroring it unless a deliberate orientation change
  is desired.
