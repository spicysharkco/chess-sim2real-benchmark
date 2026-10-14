# Deplorable Mountaineer Staunton pieces

`chess_pieces.json` contains the six piece types from Deplorable Mountaineer's
artist-authored **Chess Board and Pieces** asset. The asset is released under
CC0 1.0 Universal. The board and original materials are intentionally omitted.

- Author: Deplorable Mountaineer
- Source: <https://deplorablemountaineer.itch.io/chess-board-and-pieces>
- Asset license: CC0 1.0 Universal
- Download archive SHA-256:
  `585a53483ced76b3627941e5594c10a66eadc33d44826024c0b38d9f28164422`
- Collada member: `Chess/Collada/Chess Scene.dae`
- Collada SHA-256:
  `4e33f1d97737bc2e74f803b64218c401107c27a7f16c110b79dfd95f0373794e`
- Generated JSON SHA-256:
  `489e258aeb49b0d7e5dafd3ba56b257e709d5a6eefa38b7686ed84c823972430`

The conversion selects one complete white set:

| FEN | Scene node | Collada geometry |
| --- | --- | --- |
| P | `Black Pawn 1.001` | `Cylinder_017-mesh` |
| N | `White Knight K` | `Knight_002-mesh` |
| B | `White Bishop K` | `Cylinder_019-mesh` |
| R | `White Rook K` | `Rook_002-mesh` |
| Q | `White Queen` | `Cylinder_021-mesh` |
| K | `White King` | `King_001-mesh` |

The pawn's source node retains an erroneous Blender display name, but it is a
member of the `White Pieces` scene group and uses the white pawn geometry.

The Collada file is metre-scale and Z-up. Geometry is converted to right-handed
Three.js Y-up coordinates with `(x, y, z) -> (x, z, -y)`. Board-placement node
translations are not applied. Each piece is recentered on its own horizontal
bounding box and its base is placed at `y=0`. One uniform scale is then applied
to all six pieces, making the largest horizontal extent 0.82 board-square units;
the king remains approximately 1.93 squares high.

Each piece object provides flat `positions`, `normals`, and `indices` arrays
that can be passed directly to Three.js r128 `BufferAttribute` objects. Source
smooth normals and all artist geometry are preserved; there is no procedural
replacement, jitter, or decimation.
