# KillGorack chess pieces

`chess_pieces.json` contains the six piece types from KillGorack's **Chess board + pieces**
asset. The asset is released under CC0 1.0 Universal. The board and original materials are intentionally
omitted.

- Author: KillGorack
- Source: <https://opengameart.org/content/chess-board-pieces>
- First-party page: <https://www.killgorack.com/PX4/index.php?ap=Assets&cn=det&id=52>
- Asset licence: CC0 1.0 Universal
- Download archive SHA-256:
  `21f538bfc320064aceb66a4ff63c81afbcdc2f95f2563f32a0e95763819a065e`
- Blender member: `wood chess board.blend`
- Blender member SHA-256:
  `1a04bffc1fa2734110b597929654b9acd65cf1232ced1e9a6bbdae77c5900206`

The conversion uses the six white meshes (`White_Pawn`, `White_Knight`, `White_Bishop`,
`White_Rook`, `White_Queen`, `White_King`). The renderer reuses them for both neutral sides.

The Blender scene is Z-up. Geometry is converted to right-handed Three.js Y-up coordinates with
`(x, y, z) -> (x, z, -y)`. Scene placement transforms are not applied. Each piece is recentred
on its own horizontal bounding box and its base is placed at `y=0`. One uniform scale is then
applied to all six pieces, making the largest horizontal extent 0.82 board-square units.

Each piece object provides flat `positions`, `normals` and `indices` arrays. Triangles are the
native Blender triangles, and the corner normals are preserved, coordinate-mapped and unit
normalised.
