# Phuong chess pieces

`chess_pieces.json` contains the six piece types from Phuong2647's chess set, **Chess Models for
Castle Game Engine Bad Chess**. The board and original materials are intentionally omitted.

- Author: Phuong2647
- Source: <https://blendswap.com/blend/29244>
- Public distribution: <https://github.com/castle-engine/bad-chess/releases/download/chess-models/chess-models.zip>
- Licence: CC-BY as declared by its creator (no version displayed), attributed under CC BY 4.0
  terms. See `LICENSE.md` and `docs/records/PHUONG_DECISION_2026-09-14.md`.
- Download archive SHA-256:
  `1b7c9f03e0bfc9836e9c775bb04dda390cd1d860d5aeb31e7cef935cfcd2a9c4`

Attribution line to use with images of this family:
"Chess set by Phuong2647 (https://blendswap.com/blend/29244), CC-BY; geometry converted to glTF
and rescaled for rendering (see docs/records/BENCHMARK_ATTRIBUTIONS_2026-09-10.md)."

The conversion uses the six white glTF meshes (`chess/white_pawn.gltf` and its five siblings,
each with its `.bin` file). Source node transforms are applied. Blender imports the glTF Y-up
data as Z-up, and the geometry is then mapped to Three.js Y-up coordinates with
`(x, y, z) -> (x, z, -y)`. Each piece is recentred on its horizontal bounding box and its base
is placed at `y=0`. One uniform scale is applied to all six pieces, making the largest
horizontal extent 0.82 board-square units.

Each piece object provides flat `positions`, `normals` and `indices` arrays. The original glTF
normals are preserved, transformed with the node matrices, coordinate-mapped and unit
normalised. The changes are conversion, normalisation and rescaling only.
