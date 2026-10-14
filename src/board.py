"""Board localisation + perspective normalisation.

The renders have a flat background and a board with two known square colours, so the
board outline is found by colour, not by learning. Steps:
  1. mask pixels that look like board wood (near either square colour, warm tint)
  2. largest connected blob -> convex hull -> 4 corners (fit 4 lines when a corner is
     occluded by a piece, so approxPolyDP alone is not enough)
  3. order corners tl,tr,br,bl in IMAGE frame, warp to a square canvas with a margin so
     pieces that rise above the far edge stay inside the crop
Output canvas: SIZE x SIZE, board occupying the centre BOARD x BOARD. Cell = BOARD/8 px.

Only numpy + opencv here so the same code runs inside the evaluator container.
"""
import cv2
import numpy as np

SIZE = 768          # canvas
BOARD = 512         # board side inside canvas
MARGIN = (SIZE - BOARD) // 2
CELL = BOARD // 8

LIGHT = np.array([215, 206, 190], dtype=np.int16)
DARK = np.array([170, 148, 126], dtype=np.int16)


def board_mask(img_rgb: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Pixels near either square colour with a warm tint. All-OpenCV: ~6 ms at 1920x1080
    instead of ~120 ms with numpy int16 math (the evaluator has a 15-minute cap)."""
    tol = 40
    m_light = cv2.inRange(img_rgb, np.clip(LIGHT - tol, 0, 255).astype(np.uint8), np.clip(LIGHT + tol, 0, 255).astype(np.uint8))
    m_dark = cv2.inRange(img_rgb, np.clip(DARK - tol, 0, 255).astype(np.uint8), np.clip(DARK + tol, 0, 255).astype(np.uint8))
    r, _, bch = cv2.split(img_rgb)
    warm = cv2.compare(cv2.subtract(r, bch), 10, cv2.CMP_GT)      # wood is warm; white pieces are grey
    m = cv2.bitwise_and(cv2.bitwise_or(m_light, m_dark), warm)
    k = np.ones((ksize, ksize), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == i).astype(np.uint8) * 255


def _line(p, q):
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def _intersect(l1, l2):
    x = np.cross(l1, l2)
    if abs(x[2]) < 1e-9:
        return None
    return x[:2] / x[2]


def _quad_from_hull(hull: np.ndarray):
    """hull: (N,2). Take the 4 longest hull edges, sorted by angle around the centroid,
    intersect consecutive ones. Robust when one corner is clipped by a piece."""
    pts = hull.astype(np.float64)
    n = len(pts)
    if n < 4:
        return None
    edges = []
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        edges.append((np.linalg.norm(q - p), i))
    edges.sort(reverse=True)
    top = sorted(i for _, i in edges[:4])          # keep hull order (= angular order)
    lines = [_line(pts[i], pts[(i + 1) % n]) for i in top]
    corners = []
    for k in range(4):
        c = _intersect(lines[k], lines[(k + 1) % 4])
        if c is None:
            return None
        corners.append(c)
    return np.array(corners, dtype=np.float32)


def order_corners(c: np.ndarray) -> np.ndarray:
    """tl, tr, br, bl in image coordinates: clockwise around the centroid, starting at the
    corner with the smallest x+y. Never maps two corners to the same point."""
    c = np.asarray(c, dtype=np.float32)
    ctr = c.mean(0)
    ang = np.arctan2(c[:, 1] - ctr[1], c[:, 0] - ctr[0])
    idx = np.argsort(ang)                      # increasing angle = clockwise with y down
    c = c[idx]
    start = int(np.argmin(c.sum(1)))
    return np.roll(c, -start, axis=0)


def _ransac_line(pts: np.ndarray, iters=300, tol=1.5, rng=None):
    """pts (N,2) -> homogeneous line through the largest collinear subset. Pieces standing
    on the board edge push contour points inward; the true edge is the long collinear run."""
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    if n < 2:
        return None
    # vectorised: all candidate pairs at once (iters x N distance matrix), no Python loop
    i = rng.integers(0, n, iters); j = rng.integers(0, n, iters)
    ok = i != j
    i, j = i[ok], j[ok]
    p, q = pts[i], pts[j]                                   # (M,2)
    a = q[:, 1] - p[:, 1]; bb = p[:, 0] - q[:, 0]; c = -(a * p[:, 0] + bb * p[:, 1])
    nrm = np.hypot(a, bb)
    keep = nrm > 1e-9
    a, bb, c, nrm = a[keep], bb[keep], c[keep], nrm[keep]
    if len(a) == 0:
        return None
    d = np.abs(np.outer(a, pts[:, 0]) + np.outer(bb, pts[:, 1]) + c[:, None]) / nrm[:, None]   # (M,N)
    inl_count = (d < tol).sum(1)
    k = int(inl_count.argmax())
    best_in = d[k] < tol
    if best_in.sum() < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts[best_in].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return _line((x0, y0), (x0 + vx, y0 + vy))


def refine_quad(cnt: np.ndarray, quad: np.ndarray):
    """Assign contour points to the nearest side of the initial quad, RANSAC-fit each side,
    intersect. Falls back to the initial quad if any side fails."""
    pts = cnt.reshape(-1, 2).astype(np.float64)
    lines = []
    for k in range(4):
        p, q = quad[k].astype(np.float64), quad[(k + 1) % 4].astype(np.float64)
        v = q - p
        L = np.linalg.norm(v)
        if L < 1:
            return quad
        u = v / L
        rel = pts - p
        t = rel @ u                                  # position along the side
        d = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])   # distance from the side line
        sel = (d < 30) & (t > 0.03 * L) & (t < 0.97 * L)
        if sel.sum() < 20:
            return quad
        l = _ransac_line(pts[sel])
        if l is None:
            return quad
        lines.append(l)
    out = []
    for k in range(4):
        c = _intersect(lines[(k - 1) % 4], lines[k])
        if c is None:
            return quad
        out.append(c)
    out = np.array(out, dtype=np.float32)
    # sanity: refinement may only nudge corners. Reject if any corner moved more than 15% of
    # the shortest side, or the quad lost convexity/area (thin low-angle trapezoids do this).
    side = min(np.linalg.norm(quad[k] - quad[(k + 1) % 4]) for k in range(4))
    if np.linalg.norm(out - quad, axis=1).max() > 0.15 * side + 4:
        return quad
    if not cv2.isContourConvex(out.reshape(-1, 1, 2)):
        return quad
    if cv2.contourArea(out) < 0.85 * cv2.contourArea(quad):
        return quad
    return out


def find_corners(img_rgb: np.ndarray, scale: float = None):
    """Detect at half resolution when the image is large (1920x1080 -> 960x540; corner error
    under 2 px, inside the +-12 px jitter the models trained with); small images run at full
    resolution. The evaluator caps a run at 15 minutes, so this path must stay fast."""
    if scale is None:
        scale = 0.5 if min(img_rgb.shape[:2]) >= 900 else 1.0
    if scale != 1.0:
        small = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        c = find_corners(small, 1.0)
        return None if c is None else c / scale
    m = _largest_blob(board_mask(img_rgb))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 5000 * 0.25:
        return None
    hull = cv2.convexHull(cnt).reshape(-1, 2)
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    quad = approx.astype(np.float32) if len(approx) == 4 else _quad_from_hull(hull)
    if quad is None:
        return None
    quad = order_corners(quad)
    quad = refine_quad(cnt, quad)
    return order_corners(quad)


def warp(img_rgb: np.ndarray, corners: np.ndarray, size: int = SIZE, board: int = BOARD) -> np.ndarray:
    m = (size - board) // 2
    dst = np.array([[m, m], [m + board, m], [m + board, m + board], [m, m + board]], np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img_rgb, H, (size, size), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(38, 38, 46))


def checker_score(warped: np.ndarray, size: int = SIZE, board: int = BOARD) -> float:
    """Self-check: how strongly the 64 cell means alternate. ~1 = clean board, ~0 = junk.
    Uses the cell centres' lower half to dodge piece bodies."""
    g = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
    MARGIN, CELL = (size - board) // 2, board // 8
    # sample a thin strip just inside each cell's bottom edge: least likely to hold a piece
    ok = 0
    for r in range(8):
        for c in range(8):
            y0 = MARGIN + r * CELL + CELL - CELL // 5
            x0 = MARGIN + c * CELL + CELL // 4
            cell = g[y0:y0 + CELL // 6, x0:x0 + CELL // 2].mean()
            # neighbour to the right (same strip); colours must alternate
            if c < 7:
                nb = g[y0:y0 + CELL // 6, x0 + CELL:x0 + CELL + CELL // 2].mean()
                light_here = (r + c) % 2 == 1   # this renderer has a1 LIGHT (measured), so a8 is dark
                ok += float((cell > nb) == light_here)
    return ok / 56.0


def normalise(img_rgb: np.ndarray, size: int = SIZE, board: int = BOARD):
    """Full pipeline. Returns (warped, corners, score). Falls back to a centre crop that
    scales the whole image into the canvas when detection fails."""
    corners = find_corners(img_rgb)
    if corners is None:
        h, w = img_rgb.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = img_rgb[y0:y0 + s, x0:x0 + s]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), None, 0.0
    out = warp(img_rgb, corners, size, board)
    return out, corners, checker_score(out, size, board)


def crop_box(corners, img_w, img_h, side=0.2, top=0.8, bottom=0.25):
    """Loose pixel box around a detected/annotated board quad, for the raw member's crop-then-resize input.
    corners: (4,2) in IMAGE frame, any order. Expands the quad's bbox by `side` x width left/right, `top` x height
    upward (tall pieces rise above the far edge) and `bottom` x height downward, clamped to the image.
    Returns x0, y0, x1, y1 (ints, x1/y1 exclusive). Same rule at prep time and in predict.py."""
    c = np.asarray(corners, dtype=np.float64)
    x0, y0 = c.min(0); x1, y1 = c.max(0)
    w, h = x1 - x0, y1 - y0
    X0 = int(max(0, np.floor(x0 - side * w))); X1 = int(min(img_w, np.ceil(x1 + side * w)))
    Y0 = int(max(0, np.floor(y0 - top * h))); Y1 = int(min(img_h, np.ceil(y1 + bottom * h)))
    return X0, Y0, max(X1, X0 + 8), max(Y1, Y0 + 8)
