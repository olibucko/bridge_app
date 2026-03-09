"""
geometry.py
-----------
Defines the Howe Truss bridge geometry:
  - 24 nodes across two side trusses + floor + top bracing
  - 63 members grouped by structural role
  - 500mm span | 150mm wide | 150mm tall

Coordinate system:
  X = along bridge length
  Y = vertical (up)
  Z = across bridge width
"""

import numpy as np

# ── Configurable parameters ─────────────────────────────────────────
SPAN   = 500.0   # mm  total bridge length
WIDTH  = 150.0   # mm  side-to-side
HEIGHT = 150.0   # mm  truss height
HALF_END = 50.0  # mm  half-triangle end panel length
PANEL    = 100.0 # mm  interior panel length (4 panels)

# Derived X positions along the bridge
X = [
    0.0,                            # node 1  — left pin
    HALF_END,                       # node 2  — end of left half-triangle
    HALF_END + PANEL,               # node 3
    HALF_END + 2*PANEL,             # node 4  — CENTRE
    HALF_END + 3*PANEL,             # node 5
    HALF_END + 4*PANEL,             # node 6  — end of right half-triangle
    SPAN,                           # node 7  — right roller
]


def build_nodes(width=WIDTH):
    """
    Returns dict  {node_id (int): (x, y, z)}
    Nodes  1-7  : left side bottom chord  Z=0
    Nodes  8-12 : left side top chord     Z=0
    Nodes 13-19 : right side bottom chord Z=width
    Nodes 20-24 : right side top chord    Z=width
    """
    n = {}
    # Left bottom chord
    for i, x in enumerate(X, start=1):
        n[i] = (x, 0.0, 0.0)
    # Left top chord (no nodes at the two ground-pins)
    for i, x in enumerate(X[1:-1], start=8):
        n[i] = (x, HEIGHT, 0.0)
    # Right bottom chord
    for i, x in enumerate(X, start=13):
        n[i] = (x, 0.0, width)
    # Right top chord
    for i, x in enumerate(X[1:-1], start=20):
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
      diagonal  — Howe diagonals (tension, rising toward centre)
      floor     — floor cross members
      floordiag — floor diagonal bracing
      topbrace  — top lateral cross + diagonal members
    """
    m = []

    # ── LEFT SIDE TRUSS ──────────────────────────────────────────────
    # Bottom chord
    for i in range(1, 7):
        m.append((i, i+1, "bottom"))
    # Top chord
    for i in range(8, 12):
        m.append((i, i+1, "top"))
    # End inclined posts (ground → top corner)
    m.append((1,  8,  "endpost"))
    m.append((7,  12, "endpost"))
    # Verticals  (bottom → top, same X position)
    for bot, top in [(2,8),(3,9),(4,10),(5,11),(6,12)]:
        m.append((bot, top, "vertical"))
    # Howe diagonals (rising toward centre from each end)
    m.append((2,  9,  "diagonal"))   # panel 1
    m.append((3,  10, "diagonal"))   # panel 2
    m.append((5,  10, "diagonal"))   # panel 3 (mirrored)
    m.append((6,  11, "diagonal"))   # panel 4 (mirrored)

    # ── RIGHT SIDE TRUSS (mirror, nodes +12 / +12) ───────────────────
    for i in range(13, 19):
        m.append((i, i+1, "bottom"))
    for i in range(20, 24):
        m.append((i, i+1, "top"))
    m.append((13, 20, "endpost"))
    m.append((19, 24, "endpost"))
    for bot, top in [(14,20),(15,21),(16,22),(17,23),(18,24)]:
        m.append((bot, top, "vertical"))
    m.append((14, 21, "diagonal"))
    m.append((15, 22, "diagonal"))
    m.append((17, 22, "diagonal"))
    m.append((18, 23, "diagonal"))

    # ── FLOOR TRUSS ──────────────────────────────────────────────────
    # Cross members (straight across at each X station)
    # Centre cross beam (4-16) uses heavier section for load transfer
    left_bot  = list(range(1, 8))
    right_bot = list(range(13, 20))
    for l, r in zip(left_bot, right_bot):
        if l == 4:  # centre span — load application point
            m.append((l, r, "floorcentre"))
        else:
            m.append((l, r, "floor"))
    # Symmetric floor diagonals — chevron pattern converging at centre
    # Left half: "/" sloping right toward centre
    floor_diags_left  = [(1,14), (2,15), (3,16)]
    # Right half: "\" sloping left toward centre
    floor_diags_right = [(16,5), (17,6), (18,7)]
    for a, b in floor_diags_left + floor_diags_right:
        m.append((a, b, "floordiag"))

    # ── TOP LATERAL BRACING ──────────────────────────────────────────
    # Cross members
    left_top  = list(range(8,  13))
    right_top = list(range(20, 25))
    for l, r in zip(left_top, right_top):
        m.append((l, r, "topbrace"))
    # Diagonal bracing
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
