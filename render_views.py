"""Render LHS / Top / Bottom views of the current bridge to a PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bridge.geometry import build_nodes, build_members
from bridge.solver import solve, SectionDef
from bridge.visualiser import BridgeVisualiser, BG, BORDER

SECTIONS = {
    "bottom":      SectionDef(12.0, 12.0),
    "top":         SectionDef(12.0, 12.0),
    "endpost":     SectionDef(5.0, 12.0),
    "vertical":    SectionDef(3.0, 12.0),
    "diagonal":    SectionDef(3.0, 12.0),
    "floor":       SectionDef(1.5, 12.0),
    "floorcentre": SectionDef(5.0, 12.0),
    "floordiag":   SectionDef(1.5, 12.0),
    "topbrace":    SectionDef(1.5, 12.0),
}

nodes = build_nodes(width=50.0)
members = build_members()
result = solve(nodes, members, sections=SECTIONS)

vis = BridgeVisualiser.__new__(BridgeVisualiser)
import matplotlib.colors as mcolors
from bridge.visualiser import FORCE_CMAP
vis.result = result
vis.nodes = result.nodes
vis.members = result.members
vis.forces = result.forces
vis.U = result.U
vis._show_labels = False
vis._show_data = False
vis._show_dims = False
vis._max_z = max(z for (_, _, z) in nodes.values())
vis._height = max(y for (_, y, _) in nodes.values())
abs_max = max(abs(result.forces.max()), abs(result.forces.min()), 1.0)
vis._norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
vis._cmap = FORCE_CMAP
vis._visible_members = []
vis._status_lookup = {r['idx']: r for r in result.member_status()}

fig = plt.figure(figsize=(14, 12), facecolor=BG)
fig.suptitle("Bridge — LHS / Top / Bottom Views",
             color="#f0f0f4", fontsize=16, fontweight='bold', y=0.98)

ax_side = fig.add_subplot(3, 1, 1)
vis._draw_2d_side(ax_side, 0.0, "LHS Truss  (Z = 0)")

ax_top = fig.add_subplot(3, 1, 2)
vis._draw_2d_top(ax_top)

ax_floor = fig.add_subplot(3, 1, 3)
vis._draw_2d_floor(ax_floor)

fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.05, hspace=0.55)

out = "D:/Project Files/bridge_app/views.png"
fig.savefig(out, dpi=160, facecolor=BG)
print(f"Saved {out}")
