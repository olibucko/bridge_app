"""
geometry.py
-----------
Defines the Howe Truss bridge geometry:
  - 32 nodes across two side trusses + floor + top bracing
  - Members grouped by structural role
  - 600mm span | 50mm wide | 100mm tall | 6 interior cells

Coordinate system:
  X = along bridge length
  Y = vertical (up)
  Z = across bridge width
"""

import numpy as np

# ── Configurable parameters ─────────────────────────────────────────
SPAN   = 600.0   # mm  total bridge length
WIDTH  = 50.0    # mm  side-to-side
HEIGHT = 100.0   # mm  truss height
HALF_END = 75.0  # mm  half-triangle end panel length
PANEL    = 75.0  # mm  interior panel length (6 panels)

# Derived X positions along the bridge (9 stations)
X = [
    0.0,                            # node 1  — left pin
    HALF_END,                       # node 2  — end of left half-triangle
    HALF_END + PANEL,               # node 3
    HALF_END + 2*PANEL,             # node 4
    HALF_END + 3*PANEL,             # node 5  — CENTRE
    HALF_END + 4*PANEL,             # node 6
    HALF_END + 5*PANEL,             # node 7
    HALF_END + 6*PANEL,             # node 8  — end of right half-triangle
    SPAN,                           # node 9  — right roller
]

# ── Key node IDs (for solver & visualiser) ──────────────────────────
SUPPORT_NODES = [1, 9, 17, 25]     # four ground-pin corners
LOAD_NODES    = [5, 21]            # centre bottom chord, both sides

# ── Diagonal definitions ────────────────────────────────────────────
# Each entry: (bottom_node, top_node) — full-length diagonals
_LEFT_DIAGS  = [(2,11), (3,12), (4,13), (6,13), (7,14), (8,15)]
_RIGHT_DIAGS = [(18,27), (19,28), (20,29), (22,29), (23,30), (24,31)]


def build_nodes(width=WIDTH):
    """
    Returns dict  {node_id (int): (x, y, z)}
    Nodes  1-9  : left side bottom chord  Z=0
    Nodes 10-16 : left side top chord     Z=0
    Nodes 17-25 : right side bottom chord Z=width
    Nodes 26-32 : right side top chord    Z=width
    """
    n = {}
    # Left bottom chord
    for i, x in enumerate(X, start=1):
        n[i] = (x, 0.0, 0.0)
    # Left top chord (no nodes at the two ground-pins)
    for i, x in enumerate(X[1:-1], start=10):
        n[i] = (x, HEIGHT, 0.0)
    # Right bottom chord
    for i, x in enumerate(X, start=17):
        n[i] = (x, 0.0, width)
    # Right top chord
    for i, x in enumerate(X[1:-1], start=26):
        n[i] = (x, HEIGHT, width)

    return n


def build_members():
    """
    Returns list of  (n1, n2, group_label)
    Groups:
      bottom    — bottom chord
      top       — top chord
      endpost   — inclined posts at each end
      vertical  — Howe verticals (compression members)
      diagonal  — Howe diagonals (full length, bottom → top)
      floor     — floor cross members
      floordiag — floor diagonal bracing
      topbrace  — top lateral cross + diagonal members
    """
    m = []

    # ── LEFT SIDE TRUSS ──────────────────────────────────────────────
    # Bottom chord: 1-2, 2-3, ..., 8-9
    for i in range(1, 9):
        m.append((i, i+1, "bottom"))
    # Top chord: 10-11, 11-12, ..., 15-16
    for i in range(10, 16):
        m.append((i, i+1, "top"))
    # End inclined posts (ground → top corner)
    m.append((1,  10, "endpost"))
    m.append((9,  16, "endpost"))
    # Verticals (bottom → top, same X position)
    for bot, top in [(2,10),(3,11),(4,12),(5,13),(6,14),(7,15),(8,16)]:
        m.append((bot, top, "vertical"))
    # Diagonals (full length, bottom → top)
    for bot, top in _LEFT_DIAGS:
        m.append((bot, top, "diagonal"))

    # ── RIGHT SIDE TRUSS (mirror, nodes +16) ─────────────────────────
    for i in range(17, 25):
        m.append((i, i+1, "bottom"))
    for i in range(26, 32):
        m.append((i, i+1, "top"))
    m.append((17, 26, "endpost"))
    m.append((25, 32, "endpost"))
    for bot, top in [(18,26),(19,27),(20,28),(21,29),(22,30),(23,31),(24,32)]:
        m.append((bot, top, "vertical"))
    # Diagonals (full length)
    for bot, top in _RIGHT_DIAGS:
        m.append((bot, top, "diagonal"))

    # ── FLOOR TRUSS ──────────────────────────────────────────────────
    # Cross members (straight across at each X station)
    left_bot  = list(range(1, 10))
    right_bot = list(range(17, 26))
    for l, r in zip(left_bot, right_bot):
        if l == 5:  # centre span — load application point
            m.append((l, r, "floorcentre"))
        else:
            m.append((l, r, "floor"))
    # Symmetric floor diagonals — chevron pattern converging at centre
    # Left half: "/" sloping right toward centre
    floor_diags_left  = [(1,18), (2,19), (3,20), (4,21)]
    # Right half: "\" sloping left toward centre
    floor_diags_right = [(21,6), (22,7), (23,8), (24,9)]
    for a, b in floor_diags_left + floor_diags_right:
        m.append((a, b, "floordiag"))

    # ── TOP LATERAL BRACING ──────────────────────────────────────────
    # Cross members at each top-chord station
    left_top  = list(range(10, 17))
    right_top = list(range(26, 33))
    for l, r in zip(left_top, right_top):
        m.append((l, r, "topbrace"))
    # Diagonal bracing in the top XZ plane (provides lateral stability)
    for l, r in zip(left_top, right_top[1:]):
        m.append((l, r, "topbrace"))

    return m


# Colour map for each member group (used by visualiser)
GROUP_COLORS = {
    "bottom":      "#4fc3f7",   # light blue
    "top":         "#ba68c8",   # purple
    "endpost":     "#ffb74d",   # amber
    "vertical":    "#81c784",   # green
    "diagonal":    "#e57373",   # red
    "floor":       "#4dd0e1",   # cyan
    "floorcentre": "#00e5ff",   # bright cyan — centre load beam
    "floordiag":   "#f06292",   # pink
    "topbrace":    "#aed581",   # lime
}

GROUP_LABELS = {
    "bottom":      "Bottom Chord",
    "top":         "Top Chord",
    "endpost":     "End Post",
    "vertical":    "Vertical",
    "diagonal":    "Diagonal",
    "floor":       "Floor Cross",
    "floorcentre": "Centre Beam",
    "floordiag":   "Floor Diagonal",
    "topbrace":    "Top Brace",
}
