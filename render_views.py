"""Render LHS / Top / Bottom views of the current bridge to a PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bridge.geometry import build_nodes, build_members
from bridge.solver import solve, SectionDef
from bridge.visualiser import BridgeVisualiser, BG, FORCE_CMAP

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


def render(draw_fn, filename, figsize, force_equal=False, xlim=None, ylim=None):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    ax = fig.add_subplot(1, 1, 1)
    draw_fn(ax)
    if force_equal:
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.20)
    fig.savefig(filename, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"Saved {filename}")


# LHS — equal aspect, centred xlim, tight ylim (deck 0..100)
render(lambda ax: vis._draw_2d_side(ax, 0.0, "LHS Truss  (Z = 0)"),
       "D:/Project Files/bridge_app/view_lhs.png",
       figsize=(14, 3.0), force_equal=True,
       xlim=(-30, 630), ylim=(-30, 130))

# Top — force equal aspect so 600×50 deck reads proportionally
render(vis._draw_2d_top,
       "D:/Project Files/bridge_app/view_top.png",
       figsize=(14, 2.4), force_equal=True,
       xlim=(-30, 630), ylim=(-25, 75))

# Bottom (floor) — same treatment, full span 0..600
render(vis._draw_2d_floor,
       "D:/Project Files/bridge_app/view_bottom.png",
       figsize=(14, 2.4), force_equal=True,
       xlim=(-30, 630), ylim=(-25, 75))

# 3D — clean, no UI overlays, no axes, tight crop
fig3d = plt.figure(figsize=(20, 8), facecolor=BG)
ax3d = fig3d.add_subplot(1, 1, 1, projection='3d')
ax3d.set_facecolor(BG)
vis._show_data = False
vis._draw_3d(ax3d)
# Remove overlay text, axis labels, ticks, panes — keep only the bridge
for txt in list(ax3d.texts):
    txt.remove()
ax3d.set_xlabel(""); ax3d.set_ylabel(""); ax3d.set_zlabel("")
ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
ax3d.grid(False)
for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
    axis.line.set_color((0, 0, 0, 0))
    axis.pane.set_visible(False)

xs = [p[0] for p in nodes.values()]
ys = [p[1] for p in nodes.values()]
zs = [p[2] for p in nodes.values()]
ax3d.set_box_aspect((max(xs)-min(xs), max(zs)-min(zs), max(ys)-min(ys)))
ax3d.view_init(elev=22, azim=-55)
ax3d.set_position([-0.10, -0.15, 1.20, 1.30])
fig3d.savefig("D:/Project Files/bridge_app/view_3d.png", dpi=180, facecolor=BG,
              bbox_inches='tight', pad_inches=0.15)
plt.close(fig3d)
print("Saved D:/Project Files/bridge_app/view_3d.png")

# Combined poster with proper height ratios
fig = plt.figure(figsize=(14, 11), facecolor=BG)
fig.suptitle("Bridge — LHS / Top / Bottom Views",
             color="#f0f0f4", fontsize=16, fontweight='bold', y=0.985)
gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.7, 1.7],
                      left=0.06, right=0.98, top=0.95, bottom=0.05, hspace=0.55)
vis._draw_2d_side(fig.add_subplot(gs[0]), 0.0, "LHS Truss  (Z = 0)")
vis._draw_2d_top(fig.add_subplot(gs[1]))
vis._draw_2d_floor(fig.add_subplot(gs[2]))
fig.savefig("D:/Project Files/bridge_app/views.png", dpi=160, facecolor=BG)
plt.close(fig)
print("Saved D:/Project Files/bridge_app/views.png")
