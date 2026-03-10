"""
visualiser.py
-------------
Interactive matplotlib application for bridge analysis results.
Professional CAD-style dark UI with top toolbar, main viewport, and data panel.

Views:  3D | Side L | Side R | Floor | Top
Toggles: Labels | Data
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from bridge.solver import SolverResult, BALSA_TENSILE, BALSA_COMPRESSIVE
from bridge.geometry import GROUP_COLORS, GROUP_LABELS, SUPPORT_NODES, LOAD_NODES
from bridge.budget import compute_budget

# ── High-quality rendering defaults ──────────────────────────────────
matplotlib.rcParams.update({
    'lines.antialiased': True,
    'text.antialiased': True,
    'figure.dpi': 110,
    'savefig.dpi': 250,
    'axes.linewidth': 0.6,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
})

# ── Theme ────────────────────────────────────────────────────────────
BG         = "#1b1b1f"
TOOLBAR_BG = "#28282e"
PANEL_BG   = "#222228"
GRID_COL   = "#2e2e36"
BORDER     = "#3a3a44"
ACCENT     = "#2b7de9"
ACCENT_HI  = "#4a9af5"
ACCENT_DIM = "#1e3a5f"
TEXT_PRI   = "#d4d4d8"
TEXT_SEC   = "#8b8b96"
TEXT_HEAD  = "#f0f0f4"
TENSION_C  = "#34c759"
COMPRESS_C = "#ff453a"
NEUTRAL_C  = "#b0b0b8"
WARN_C     = "#ffd60a"
NODE_FILL  = "#f0f0f4"
NODE_EDGE  = "#2b7de9"
NODE_GLOW  = "#2b7de940"
SUPPORT_C  = "#ff9f0a"
DIM_C      = "#80cbc4"   # teal for dimension lines
SECTION_C  = "#ce93d8"   # lilac for section callouts


# ── Colormap: red -> white -> green (smooth, wide ramp) ─────────────
_cdict = {
    'red':   [(0.0, 0.92, 0.92), (0.25, 0.75, 0.75),
              (0.5, 0.96, 0.96),
              (0.75, 0.45, 0.45), (1.0, 0.20, 0.20)],
    'green': [(0.0, 0.22, 0.22), (0.25, 0.38, 0.38),
              (0.5, 0.96, 0.96),
              (0.75, 0.72, 0.72), (1.0, 0.78, 0.78)],
    'blue':  [(0.0, 0.22, 0.22), (0.25, 0.38, 0.38),
              (0.5, 0.96, 0.96),
              (0.75, 0.38, 0.38), (1.0, 0.22, 0.22)],
}
FORCE_CMAP = mcolors.LinearSegmentedColormap('force', _cdict, N=1024)


class BridgeVisualiser:

    VIEWS = ["3D", "Side L", "Side R", "Floor", "Top"]

    def __init__(self, result: SolverResult):
        self.result  = result
        self.nodes   = result.nodes
        self.members = result.members
        self.forces  = result.forces
        self.U       = result.U

        self._current_view = "3D"
        self._show_labels  = False
        self._show_data    = True
        self._show_dims    = False

        self._max_z  = max(z for (_, _, z) in self.nodes.values())
        self._height = max(y for (_, y, _) in self.nodes.values())

        abs_max = max(abs(self.forces.max()), abs(self.forces.min()), 1.0)
        self._norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        self._cmap = FORCE_CMAP

        self._main_ax = None
        self._data_ax = None
        self._main_ax_type = None
        self._visible_members = []
        self._cb_labels = []

        self._status_lookup = {r['idx']: r for r in result.member_status()}

        self._build_figure()
        self._connect_mouse()
        self._draw()

    # ══════════════════════════════════════════════════════════════════
    #  MOUSE INTERACTION — scroll-to-zoom + middle-click pan
    # ══════════════════════════════════════════════════════════════════

    def _connect_mouse(self):
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _on_scroll(self, event):
        """Zoom toward cursor on scroll in 2D views."""
        ax = self._main_ax
        if ax is None or event.inaxes is not ax:
            return
        if self._main_ax_type == "3d":
            return  # 3D has built-in scroll zoom

        scale = 0.85 if event.button == 'up' else 1.18
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Zoom centered on cursor position
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        rx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        ry = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        ax.set_xlim(xdata - new_w * rx, xdata + new_w * (1 - rx))
        ax.set_ylim(ydata - new_h * ry, ydata + new_h * (1 - ry))
        self.fig.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════
    #  FIGURE LAYOUT
    # ══════════════════════════════════════════════════════════════════

    # ── Layout constants ────────────────────────────────────────────
    # All y-positions in figure fraction (0=bottom, 1=top)
    _TB_Y     = 0.958       # toolbar button row y
    _TB_H     = 0.030       # toolbar button height
    _SUMM_Y   = 0.940       # status summary y (just below toolbar)
    _PLOT_TOP = 0.910       # plot area top (clear of toolbar + summary)
    _PLOT_BOT = 0.100       # plot area bottom (clear of colorbar + x-axis label)
    _CB_Y     = 0.025       # colorbar y
    _CB_H     = 0.012       # colorbar height
    _MARGIN   = 0.05        # left/right figure margin

    def _build_figure(self):
        self.fig = plt.figure(figsize=(19, 10.5), facecolor=TOOLBAR_BG)
        self.fig.canvas.manager.set_window_title("Bridge Analyser v2")

        self._btn_axes = []
        self._btn_objs = {}
        self._make_toolbar()
        self._make_colorbar_ax()

    def _make_toolbar(self):
        """Horizontal toolbar across top with status summary below."""
        bw  = 0.054          # button width
        bh  = self._TB_H
        y   = self._TB_Y
        gap = 0.005
        x   = 0.015

        # ── App title ──
        self.fig.text(x + 0.003, y + bh/2, "BRIDGE ANALYSER",
                      color=ACCENT_HI, fontsize=11, fontweight='bold',
                      va='center', transform=self.fig.transFigure)
        x += 0.120

        self._toolbar_sep(x, y, bh); x += 0.010

        # ── View buttons ──
        for name in self.VIEWS:
            w = bw if len(name) <= 4 else bw + 0.010
            self._add_btn(x, y, w, bh, name)
            x += w + gap

        self._toolbar_sep(x + 0.003, y, bh); x += 0.016

        # ── Toggles ──
        for name in ["Labels", "Dims", "Data"]:
            w = bw + 0.006
            self._add_btn(x, y, w, bh, name)
            x += w + gap

        self._toolbar_sep(x + 0.003, y, bh); x += 0.016

        # ── Export ──
        self._add_btn(x, y, bw + 0.012, bh, "Save PNG")

        # ── Status summary — separate line below toolbar buttons ──
        r = self.result
        score, fail_N, fail_mode, _ = r.compute_score()
        weight = r.bridge_weight_grams()
        worst = r.worst_safety_factor()
        ok = worst > 1
        v_col = TENSION_C if ok else COMPRESS_C
        tag = "PASS" if ok else "FAIL"
        summary = (f"{tag}  |  Score: {score:.1f} N/g  |  "
                   f"Failure: {fail_N:.0f} N ({fail_mode})  |  "
                   f"Weight: {weight:.0f} g  |  SF: {worst:.1f}x")
        self.fig.text(0.50, self._SUMM_Y, summary,
                      color=v_col, fontsize=9.5, fontweight='bold',
                      va='center', ha='center',
                      transform=self.fig.transFigure)

        self._update_button_highlights()

    def _toolbar_sep(self, x, y, bh):
        self.fig.text(x, y + bh/2, "|", color=BORDER, fontsize=14,
                      va='center', transform=self.fig.transFigure)

    def _add_btn(self, x, y, w, h, name):
        ax = self.fig.add_axes([x, y, w, h])
        ax.set_facecolor(TOOLBAR_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.6)
        btn = Button(ax, name, color=TOOLBAR_BG, hovercolor=ACCENT_DIM)
        btn.label.set_color(TEXT_PRI)
        btn.label.set_fontsize(9)
        btn.on_clicked(self._make_handler(name))
        self._btn_axes.append(ax)
        self._btn_objs[name] = (btn, ax)

    def _make_colorbar_ax(self):
        # Centred colorbar with room for COMP./TENS. labels on each side
        cb_w = 0.40
        cb_x = (1.0 - cb_w) / 2.0   # centred horizontally
        self._cbar_ax = self.fig.add_axes([cb_x, self._CB_Y, cb_w, self._CB_H])

    def _make_handler(self, name):
        def handler(event):
            if name in self.VIEWS:
                self._current_view = name
            elif name == "Labels":
                self._show_labels = not self._show_labels
            elif name == "Dims":
                self._show_dims = not self._show_dims
            elif name == "Data":
                self._show_data = not self._show_data
            elif name == "Save PNG":
                self._save_png()
                return
            self._update_button_highlights()
            self._draw()
        return handler

    def _update_button_highlights(self):
        toggles = {"Labels": self._show_labels, "Dims": self._show_dims,
                   "Data": self._show_data}
        for name, (btn, ax) in self._btn_objs.items():
            active = (name == self._current_view) or toggles.get(name, False)
            bg = ACCENT if active else TOOLBAR_BG
            ax.set_facecolor(bg)
            btn.color = bg
            btn.hovercolor = ACCENT_HI if active else ACCENT_DIM
            btn.label.set_color(TEXT_HEAD if active else TEXT_PRI)
            btn.label.set_fontweight('bold' if active else 'normal')
        self.fig.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════
    #  AXES MANAGEMENT
    # ══════════════════════════════════════════════════════════════════

    def _ensure_main_ax(self, rect):
        needed = "3d" if self._current_view == "3D" else "2d"
        if self._main_ax is not None and self._main_ax_type == needed:
            self._main_ax.cla()
            self._main_ax.set_position(rect)
            return self._main_ax
        if self._main_ax is not None:
            self.fig.delaxes(self._main_ax)
            self._main_ax = None
        proj = {'projection': '3d'} if needed == "3d" else {}
        self._main_ax = self.fig.add_axes(rect, **proj)
        self._main_ax_type = needed
        return self._main_ax

    def _ensure_data_ax(self, rect):
        if self._data_ax is not None:
            self._data_ax.cla()
            self._data_ax.set_position(rect)
            return self._data_ax
        self._data_ax = self.fig.add_axes(rect)
        return self._data_ax

    def _hide_data_ax(self):
        if self._data_ax is not None:
            self.fig.delaxes(self._data_ax)
            self._data_ax = None

    # ══════════════════════════════════════════════════════════════════
    #  DRAW DISPATCHER
    # ══════════════════════════════════════════════════════════════════

    def _draw(self):
        L = self._MARGIN          # left margin
        B = self._PLOT_BOT        # bottom
        T = self._PLOT_TOP        # top
        H = T - B                 # available height
        R = 1.0 - self._MARGIN   # right edge

        panel = self._show_data and self._current_view != "3D"
        if panel:
            plot_w = 0.48
            gap    = 0.025
            data_x = L + plot_w + gap
            data_w = R - data_x
            plot_rect = [L, B, plot_w, H]
            data_rect = [data_x, B, data_w, H]
        else:
            plot_rect = [L, B, R - L, H]
            data_rect = None

        ax = self._ensure_main_ax(plot_rect)

        if self._current_view == "3D":
            self._draw_3d(ax)
        elif self._current_view == "Side L":
            self._draw_2d_side(ax, 0.0, "Left Side Truss  (Z = 0)")
        elif self._current_view == "Side R":
            self._draw_2d_side(ax, self._max_z,
                               f"Right Side Truss  (Z = {self._max_z:.0f})")
        elif self._current_view == "Floor":
            self._draw_2d_floor(ax)
        elif self._current_view == "Top":
            self._draw_2d_top(ax)

        if data_rect:
            self._draw_data_panel(self._ensure_data_ax(data_rect))
        else:
            self._hide_data_ax()

        self._draw_colorbar()
        self.fig.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════
    #  RENDERING HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _member_lw(self, idx, base):
        gov = self._status_lookup[idx]['governing_SF']
        if gov < 1.5:   return base * 2.2
        if gov < 3.0:   return base * 1.4
        return base

    def _draw_member_label(self, ax, x1, y1, x2, y2, idx, side):
        mx, my = (x1+x2)/2, (y1+y2)/2
        dx, dy = x2-x1, y2-y1
        L = max(np.hypot(dx, dy), 1e-6)
        px, py = -dy/L, dx/L
        off = 16 * side
        lx, ly = mx + px*off, my + py*off

        f = self.forces[idx]
        n1, n2, _ = self.members[idx]
        if f > 0.01:    col, tag = TENSION_C, "T"
        elif f < -0.01: col, tag = COMPRESS_C, "C"
        else:           col, tag = NEUTRAL_C, "-"

        ax.text(lx, ly, f"{n1}-{n2}  {f:+.1f} N  {tag}",
                color=col, fontsize=8.5, ha='center', va='center',
                fontweight='bold',
                bbox=dict(facecolor='#111115ee', edgecolor=col,
                          linewidth=0.7, pad=3, boxstyle='round,pad=0.35'))

    # ── Dimensions mode helpers ─────────────────────────────────────

    _DIM_SCALE = 2.0   # display scale for section thicknesses (2× real)

    def _section_for(self, grp):
        """Get the SectionDef for a member group."""
        from bridge.solver import DEFAULT_SECTION
        return self.result.sections.get(grp, DEFAULT_SECTION)

    def _draw_member_thick(self, ax, x1, y1, x2, y2, idx, grp):
        """Draw a member as a filled rectangle with true proportional width."""
        sec = self._section_for(grp)
        # Use the smaller dimension (visible thickness in elevation/plan)
        thickness = min(sec.b, sec.h) * self._DIM_SCALE

        dx, dy = x2 - x1, y2 - y1
        L = max(np.hypot(dx, dy), 1e-6)
        # Perpendicular unit vector
        px, py = -dy / L, dx / L
        half = thickness / 2.0

        # Four corners of the rectangle
        corners = [
            (x1 + px * half, y1 + py * half),
            (x2 + px * half, y2 + py * half),
            (x2 - px * half, y2 - py * half),
            (x1 - px * half, y1 - py * half),
        ]

        col = self._cmap(self._norm(self.forces[idx]))
        poly = mpatches.Polygon(corners, closed=True,
                                facecolor=col, edgecolor=TEXT_HEAD,
                                linewidth=0.5, alpha=0.85, zorder=3)
        ax.add_patch(poly)

    def _draw_callout_at(self, ax, label_x, label_y, target_x, target_y, text):
        """Draw a section callout at a fixed position with a leader line to target."""
        ax.plot([label_x, target_x], [label_y, target_y],
                color=SECTION_C, linewidth=0.6, alpha=0.5, zorder=12,
                linestyle=(0, (3, 3)))
        ax.scatter(target_x, target_y, color=SECTION_C, s=12, zorder=13, alpha=0.7)
        ax.text(label_x, label_y, text, color=SECTION_C, fontsize=8,
                ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='#111115ee', edgecolor=SECTION_C,
                          linewidth=0.6, pad=3, boxstyle='round,pad=0.3'),
                zorder=15)

    def _draw_side_callouts(self, ax, z_filter):
        """Place section callouts outside the truss for side elevation."""
        from bridge.geometry import HEIGHT, SPAN

        # Collect one representative midpoint per group
        reps = {}
        for idx, (n1, n2, grp) in enumerate(self.members):
            if grp in reps:
                continue
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            if abs(z1 - z_filter) > 0.1 or abs(z2 - z_filter) > 0.1:
                continue
            reps[grp] = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Fixed label positions OUTSIDE the truss envelope
        # Format: group -> (label_x, label_y)
        positions = {
            "top":      (SPAN + 60,  HEIGHT + 30),
            "bottom":   (SPAN + 60,  -30),
            "endpost":  (-60,        HEIGHT / 2 + 30),
            "vertical": (SPAN + 60,  HEIGHT / 2 + 30),
            "diagonal": (SPAN + 60,  HEIGHT / 2 - 20),
        }

        for grp, (lx, ly) in positions.items():
            if grp not in reps:
                continue
            sec = self._section_for(grp)
            label = f"{sec.b:g}x{sec.h:g}"
            tx, ty = reps[grp]
            self._draw_callout_at(ax, lx, ly, tx, ty, label)

    def _draw_floor_callouts(self, ax):
        """Place section callouts outside the truss for floor plan."""
        from bridge.geometry import SPAN
        width = self._max_z

        reps = {}
        for idx, (n1, n2, grp) in enumerate(self.members):
            if grp in reps:
                continue
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            if y1 != 0 or y2 != 0:
                continue
            if grp not in ("bottom", "floor", "floorcentre", "floordiag"):
                continue
            reps[grp] = ((x1 + x2) / 2, (z1 + z2) / 2)

        positions = {
            "bottom":      (SPAN + 55, -15),
            "floor":       (SPAN + 55, width / 2 - 10),
            "floorcentre": (SPAN + 55, width / 2 + 10),
            "floordiag":   (SPAN + 55, width + 15),
        }

        for grp, (lx, ly) in positions.items():
            if grp not in reps:
                continue
            sec = self._section_for(grp)
            label = f"{sec.b:g}x{sec.h:g}"
            tx, ty = reps[grp]
            self._draw_callout_at(ax, lx, ly, tx, ty, label)

    def _draw_top_callouts(self, ax):
        """Place section callouts outside the truss for top plan."""
        from bridge.geometry import SPAN
        width = self._max_z

        reps = {}
        for idx, (n1, n2, grp) in enumerate(self.members):
            if grp in reps:
                continue
            if grp not in ("top", "topbrace"):
                continue
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            reps[grp] = ((x1 + x2) / 2, (z1 + z2) / 2)

        positions = {
            "top":      (SPAN + 55, -15),
            "topbrace": (SPAN + 55, width + 15),
        }

        for grp, (lx, ly) in positions.items():
            if grp not in reps:
                continue
            sec = self._section_for(grp)
            label = f"{sec.b:g}x{sec.h:g}"
            tx, ty = reps[grp]
            self._draw_callout_at(ax, lx, ly, tx, ty, label)

    def _draw_dim_line(self, ax, x1, y1, x2, y2, label, offset=25, side=1):
        """Draw an engineering dimension line with arrows and label."""
        dx, dy = x2 - x1, y2 - y1
        L = max(np.hypot(dx, dy), 1e-6)
        px, py = -dy / L, dx / L
        off = offset * side

        # Offset points (dimension line position)
        ox1, oy1 = x1 + px * off, y1 + py * off
        ox2, oy2 = x2 + px * off, y2 + py * off

        # Extension lines — from near the member out past the dimension line
        for x, y in [(x1, y1), (x2, y2)]:
            s_off = off * 0.15   # start slightly away from member
            e_off = off + 8 * side  # extend past the dim line
            sx, sy = x + px * s_off, y + py * s_off
            ex, ey = x + px * e_off, y + py * e_off
            ax.plot([sx, ex], [sy, ey], color=DIM_C, linewidth=0.5,
                    alpha=0.6, zorder=12)

        # Dimension line with arrows
        ax.annotate('', xy=(ox1, oy1), xytext=(ox2, oy2),
                    arrowprops=dict(arrowstyle='<->', color=DIM_C,
                                    lw=0.9, mutation_scale=8),
                    zorder=12)

        # Label at midpoint
        mx, my = (ox1 + ox2) / 2, (oy1 + oy2) / 2
        ax.text(mx, my, label, color=DIM_C, fontsize=8,
                ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='#111115ee', edgecolor=DIM_C,
                          linewidth=0.5, pad=2, boxstyle='round,pad=0.25'),
                zorder=15)

    def _draw_side_dims(self, ax, z_filter):
        """Draw key dimension lines on a side elevation."""
        from bridge.geometry import HEIGHT, SPAN, HALF_END, PANEL

        # Panel dims — closer row (below bottom chord)
        self._draw_dim_line(ax, 0, 0, HALF_END, 0,
                            f"{HALF_END:.0f}", offset=45, side=-1)
        self._draw_dim_line(ax, HALF_END, 0, HALF_END + PANEL, 0,
                            f"{PANEL:.0f}", offset=45, side=-1)
        # Overall span — further out row
        self._draw_dim_line(ax, 0, 0, SPAN, 0,
                            f"{SPAN:.0f} mm", offset=80, side=-1)
        # Height — well clear to the left
        self._draw_dim_line(ax, 0, 0, 0, HEIGHT,
                            f"{HEIGHT:.0f} mm", offset=55, side=-1)

    def _draw_floor_dims(self, ax):
        """Draw key dimension lines on a floor plan."""
        from bridge.geometry import SPAN
        width = self._max_z

        # Span — below
        self._draw_dim_line(ax, 0, 0, SPAN, 0,
                            f"{SPAN:.0f} mm", offset=30, side=-1)
        # Width — to the left
        self._draw_dim_line(ax, 0, 0, 0, width,
                            f"{width:.0f} mm", offset=45, side=-1)

    def _draw_nodes_2d(self, ax, filter_fn, label_below=True):
        """Draw nodes with glow effect and clear labels."""
        for n, (x, y, z) in self.nodes.items():
            if not filter_fn(x, y, z):
                continue
            px, py = (x, y) if label_below else (x, z)
            pval = y if label_below else z

            # Glow ring
            ax.scatter(px, pval, color=NODE_GLOW, s=180, zorder=9,
                       edgecolors='none', alpha=0.35)
            # Main dot
            ax.scatter(px, pval, color=NODE_FILL, s=55, zorder=10,
                       edgecolors=NODE_EDGE, linewidths=1.2)
            # Label — offset based on position
            if label_below:
                # Side elevation: bottom nodes below, top nodes above
                oy = -13 if y < self._height / 2 else 11
                va = 'top' if y < self._height / 2 else 'bottom'
            else:
                # Plan views: label below for bottom row, above for top row
                mid_z = self._max_z / 2
                if z < mid_z:
                    oy = -6
                    va = 'top'
                else:
                    oy = 6
                    va = 'bottom'
            ax.text(px, pval + oy, str(n), color=TEXT_HEAD,
                    fontsize=9, ha='center', va=va, fontweight='bold',
                    bbox=dict(facecolor='#111115dd', edgecolor=NODE_EDGE,
                              linewidth=0.5, pad=1.5, boxstyle='round,pad=0.2'))

    def _draw_supports_2d(self, ax, z_filter, coord='xy'):
        """Draw fixed support markers."""
        for n in SUPPORT_NODES:
            nx, ny, nz = self.nodes[n]
            if abs(nz - z_filter) > 0.1:
                continue
            px = nx
            py = ny if coord == 'xy' else nz
            # Triangle marker
            ax.scatter(px, py, color=SUPPORT_C, s=260, marker='^',
                       zorder=11, edgecolors='#000000', linewidths=1.2)
            # Ground line
            gw = 18
            gy = py - 16
            ax.plot([px - gw, px + gw], [gy, gy], color=SUPPORT_C,
                    linewidth=2.0, solid_capstyle='butt')
            for tick_x in np.linspace(px - gw, px + gw, 7):
                ax.plot([tick_x, tick_x - 5], [gy, gy - 6],
                        color=SUPPORT_C, linewidth=0.9, alpha=0.7)

    def _draw_loads_2d(self, ax, z_filter):
        half = self.result.load_N / 2.0
        for n in LOAD_NODES:
            nx, ny, nz = self.nodes[n]
            if abs(nz - z_filter) > 0.1:
                continue
            ax.annotate('', xy=(nx, ny + 8), xytext=(nx, ny + 55),
                        arrowprops=dict(arrowstyle='->', color=COMPRESS_C,
                                        lw=2.8, mutation_scale=18))
            ax.text(nx, ny + 60, f'{half:.1f} N', color=COMPRESS_C,
                    fontsize=10, ha='center', fontweight='bold',
                    bbox=dict(facecolor='#111115dd', edgecolor=COMPRESS_C,
                              linewidth=0.5, pad=2, boxstyle='round,pad=0.25'))

    # ══════════════════════════════════════════════════════════════════
    #  3D VIEW
    # ══════════════════════════════════════════════════════════════════

    def _draw_3d(self, ax):
        ax.set_facecolor(BG)
        self._visible_members = list(range(len(self.members)))

        for idx, (n1, n2, grp) in enumerate(self.members):
            x1,y1,z1 = self.nodes[n1]; x2,y2,z2 = self.nodes[n2]
            col = self._cmap(self._norm(self.forces[idx]))
            base = 3.0 if grp in ("bottom","top","endpost") else 1.8
            lw = self._member_lw(idx, base)
            ax.plot([x1,x2],[z1,z2],[y1,y2],
                    color=col, linewidth=lw, solid_capstyle='round', alpha=0.95)

        for n,(x,y,z) in self.nodes.items():
            ax.scatter(x,z,y, color=NODE_FILL, s=55, zorder=5,
                       edgecolors=NODE_EDGE, linewidths=1.2)

        for n in SUPPORT_NODES:
            x,y,z = self.nodes[n]
            ax.scatter(x,z,y, color=SUPPORT_C, s=160, marker='^',
                       zorder=7, edgecolors='#000000', linewidths=1.0)

        for n in LOAD_NODES:
            x,y,z = self.nodes[n]
            ax.quiver(x,z,y+40, 0,0,-35, color=COMPRESS_C,
                      linewidth=2.5, arrow_length_ratio=0.3)

        # Style
        for attr, label in [(ax.set_xlabel, 'X  Length (mm)'),
                            (ax.set_ylabel, 'Z  Width (mm)'),
                            (ax.set_zlabel, 'Y  Height (mm)')]:
            attr(label, color=TEXT_SEC, labelpad=10, fontsize=10)
        ax.tick_params(colors=TEXT_SEC, labelsize=9)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            p.set_edgecolor(GRID_COL)
        # Title inside axes to avoid toolbar/summary overlap
        ax.text2D(0.99, 0.99, '3D View', transform=ax.transAxes,
                  fontsize=12, color=TEXT_HEAD, fontweight='bold',
                  ha='right', va='top')
        ax.view_init(elev=25, azim=-50)

        if self._show_data:
            r = self.result
            d4, _ = r.centre_deflection()
            worst = r.worst_safety_factor()
            w = r.bridge_weight_grams()
            score, fail_N, fail_mode, _ = r.compute_score()
            ok = worst > 1
            c = TENSION_C if ok else COMPRESS_C
            v = "HOLDS" if ok else "FAILS"
            txt = (f"SCORE  {score:.1f} N/g\n"
                   f"Failure {fail_N:.0f} N ({fail_mode})\n"
                   f"Weight {w:.1f} g    SF {worst:.2f}x    {v}")
            ax.text2D(0.01, 0.97, txt, transform=ax.transAxes,
                      fontsize=11, color=c, va='top', fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.6', facecolor='#0a0a0eee',
                                edgecolor=ACCENT, linewidth=1.2))

    # ══════════════════════════════════════════════════════════════════
    #  2D SIDE VIEW
    # ══════════════════════════════════════════════════════════════════

    def _draw_2d_side(self, ax, z_filter, title):
        ax.set_facecolor(BG)
        groups = {"bottom","top","endpost","vertical","diagonal","midtie"}
        visible = []
        side = 1

        for idx, (n1, n2, grp) in enumerate(self.members):
            if grp not in groups:
                continue
            x1,y1,z1 = self.nodes[n1]; x2,y2,z2 = self.nodes[n2]
            if abs(z1 - z_filter) > 0.1 or abs(z2 - z_filter) > 0.1:
                continue
            visible.append(idx)

            if self._show_dims:
                self._draw_member_thick(ax, x1, y1, x2, y2, idx, grp)
            else:
                col = self._cmap(self._norm(self.forces[idx]))
                base = 4.0 if grp in ("bottom","top","endpost") else 2.8
                lw = self._member_lw(idx, base)
                ax.plot([x1,x2],[y1,y2], color=col, linewidth=lw,
                        solid_capstyle='round')

            if self._show_labels and not self._show_dims:
                self._draw_member_label(ax, x1, y1, x2, y2, idx, side)
                side *= -1

        self._visible_members = visible
        self._draw_nodes_2d(ax, lambda x,y,z: abs(z - z_filter) < 0.1,
                            label_below=True)
        self._draw_supports_2d(ax, z_filter)
        self._draw_loads_2d(ax, z_filter)

        if self._show_dims:
            self._draw_side_dims(ax, z_filter)
            self._draw_side_callouts(ax, z_filter)

        self._style_2d_ax(ax, title, 'X  Length (mm)', 'Y  Height (mm)',
                          (-100, 620), (-130, 300) if self._show_dims else (-80, 260))

    # ══════════════════════════════════════════════════════════════════
    #  2D FLOOR VIEW
    # ══════════════════════════════════════════════════════════════════

    def _draw_2d_floor(self, ax):
        ax.set_facecolor(BG)
        groups = {"bottom","floor","floorcentre","floordiag"}
        visible = []
        side = 1

        for idx, (n1, n2, grp) in enumerate(self.members):
            if grp not in groups:
                continue
            x1,y1,z1 = self.nodes[n1]; x2,y2,z2 = self.nodes[n2]
            if y1 != 0 or y2 != 0:
                continue
            visible.append(idx)

            if self._show_dims:
                self._draw_member_thick(ax, x1, z1, x2, z2, idx, grp)
            else:
                col = self._cmap(self._norm(self.forces[idx]))
                base = 3.5 if grp in ("bottom", "floorcentre") else 2.2
                lw = self._member_lw(idx, base)
                ls = '--' if grp == "floordiag" else '-'
                ax.plot([x1,x2],[z1,z2], color=col, linewidth=lw,
                        linestyle=ls, solid_capstyle='round')

            if self._show_labels and not self._show_dims:
                self._draw_member_label(ax, x1, z1, x2, z2, idx, side)
                side *= -1

        self._visible_members = visible
        self._draw_nodes_2d(ax, lambda x,y,z: y == 0, label_below=False)

        w = self._max_z
        if self._show_dims:
            self._draw_floor_dims(ax)
            self._draw_floor_callouts(ax)
            self._style_2d_ax(ax, 'Floor Plan  (Y = 0)',
                              'X  Length (mm)', 'Z  Width (mm)',
                              (-90, 620), (-w * 0.8, w * 1.8),
                              aspect='auto')
        else:
            self._style_2d_ax(ax, 'Floor Plan  (Y = 0)',
                              'X  Length (mm)', 'Z  Width (mm)',
                              (-50, 550), (-w * 0.6, w * 1.6),
                              aspect='auto')

    # ══════════════════════════════════════════════════════════════════
    #  2D TOP VIEW
    # ══════════════════════════════════════════════════════════════════

    def _draw_2d_top(self, ax):
        ax.set_facecolor(BG)
        groups = {"top","topbrace"}
        visible = []
        side = 1

        for idx, (n1, n2, grp) in enumerate(self.members):
            if grp not in groups:
                continue
            x1,y1,z1 = self.nodes[n1]; x2,y2,z2 = self.nodes[n2]
            visible.append(idx)

            if self._show_dims:
                self._draw_member_thick(ax, x1, z1, x2, z2, idx, grp)
            else:
                col = self._cmap(self._norm(self.forces[idx]))
                base = 3.5 if grp == "top" else 2.2
                lw = self._member_lw(idx, base)
                ls = '--' if grp == "topbrace" and x1 != x2 and z1 != z2 else '-'
                ax.plot([x1,x2],[z1,z2], color=col, linewidth=lw,
                        linestyle=ls, solid_capstyle='round')

            if self._show_labels and not self._show_dims:
                self._draw_member_label(ax, x1, z1, x2, z2, idx, side)
                side *= -1

        self._visible_members = visible
        self._draw_nodes_2d(ax, lambda x,y,z: y == self._height, label_below=False)

        w = self._max_z
        if self._show_dims:
            self._draw_top_callouts(ax)
            self._style_2d_ax(ax, f'Top Plan  (Y = {self._height:.0f})',
                              'X  Length (mm)', 'Z  Width (mm)',
                              (-50, 620), (-w * 0.6, w * 1.6),
                              aspect='auto')
        else:
            self._style_2d_ax(ax, f'Top Plan  (Y = {self._height:.0f})',
                              'X  Length (mm)', 'Z  Width (mm)',
                              (-50, 550), (-w * 0.6, w * 1.6),
                              aspect='auto')

    # ══════════════════════════════════════════════════════════════════
    #  DATA PANEL
    # ══════════════════════════════════════════════════════════════════

    def _draw_data_panel(self, ax):
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(1.0)
            sp.set_visible(True)

        r = self.result
        worst = r.worst_safety_factor()
        weight = r.bridge_weight_grams()
        d4, _  = r.centre_deflection()
        score, fail_N, fail_mode, crit_idx = r.compute_score()
        fail_kg = fail_N / 9.81

        P = {'transform': ax.transAxes, 'va': 'top'}

        def t(x, y, s, color=TEXT_PRI, fs=9, fw='normal', **kw):
            ax.text(x, y, s, fontsize=fs, color=color, fontweight=fw, **P, **kw)

        y = 0.970

        # ── Score hero ──
        ok = worst > 1.0
        vc = TENSION_C if ok else COMPRESS_C
        t(0.06, y, f"SCORE  {score:.1f} N/g", color=WARN_C, fs=18, fw='bold')
        y -= 0.050
        t(0.06, y, f"Failure: {fail_N:.0f} N ({fail_kg:.1f} kg) — {fail_mode}",
          color=vc, fs=10, fw='bold')
        if crit_idx >= 0:
            n1, n2, grp = r.members[crit_idx]
            y -= 0.030
            t(0.06, y, f"Critical: {n1}-{n2} ({grp})", color=vc, fs=9)
        y -= 0.038

        # ── Metrics grid ──
        metrics = [
            ("Weight",      f"{weight:.1f} g",
             TENSION_C if r.within_weight_limit else COMPRESS_C),
            ("Deflection",  f"{abs(d4):.4f} mm", TEXT_PRI),
            ("Worst SF",    f"{worst:.2f}x",
             TENSION_C if worst >= 2 else (WARN_C if worst >= 1 else COMPRESS_C)),
        ]
        for label, val, vc in metrics:
            t(0.06, y, label, color=TEXT_SEC, fs=9)
            t(0.52, y, val, color=vc, fs=11, fw='bold')
            y -= 0.036

        # ── Divider ──
        y -= 0.010
        ax.plot([0.06, 0.94], [y, y], color=BORDER, lw=1,
                transform=ax.transAxes, clip_on=False)
        y -= 0.026

        # ── Table title ──
        view_name = self._current_view
        n_vis = len(self._visible_members)
        t(0.06, y, f"{view_name} MEMBERS  ({n_vis})", color=ACCENT_HI, fs=11, fw='bold')
        y -= 0.035

        # ── Column headers ──
        cx = [0.05, 0.17, 0.32, 0.50, 0.70, 0.86]
        ch = ["Member", "Group", "Type", "Force (N)", "SF", "Status"]
        for xi, hi in zip(cx, ch):
            t(xi, y, hi, color=TEXT_SEC, fs=8, fw='bold')
        y -= 0.010
        ax.plot([0.05, 0.95], [y, y], color=BORDER, lw=0.5,
                transform=ax.transAxes, clip_on=False)
        y -= 0.020

        # ── Rows ──
        avail = y - 0.015
        rh = min(0.030, avail / max(n_vis, 1))
        rfs = 9.0 if rh >= 0.025 else max(7.0, rh * 340)

        for i, idx in enumerate(self._visible_members):
            if y < 0.015:
                break
            n1, n2, grp = self.members[idx]
            f = self.forces[idx]
            row = self._status_lookup[idx]

            # Zebra
            if i % 2 == 0:
                ax.axhspan(y - rh * 0.8, y + 0.006, xmin=0.04, xmax=0.96,
                           facecolor='#ffffff08', transform=ax.transAxes)

            if f > 0.01:    tc, tcc = "Tension", TENSION_C
            elif f < -0.01: tc, tcc = "Compr.", COMPRESS_C
            else:           tc, tcc = "Neutral", NEUTRAL_C

            st = row['status']
            sc = COMPRESS_C if st == "FAIL" else (WARN_C if st == "WARN" else TENSION_C)

            gov = row['governing_SF']
            gs = f"{gov:.1f}" if gov < 1e6 else "-"

            t(cx[0], y, f"{n1}-{n2}", color=TEXT_HEAD, fs=rfs, fw='bold')
            t(cx[1], y, grp[:7], color=TEXT_SEC, fs=rfs)
            t(cx[2], y, tc, color=tcc, fs=rfs, fw='bold')
            t(cx[3], y, f"{f:+.2f}", color=tcc, fs=rfs)
            t(cx[4], y, gs, color=sc if gov < 3 else TEXT_SEC, fs=rfs)
            t(cx[5], y, st, color=sc, fs=rfs, fw='bold')
            y -= rh

    # ══════════════════════════════════════════════════════════════════
    #  COLOURBAR
    # ══════════════════════════════════════════════════════════════════

    def _draw_colorbar(self):
        # Remove previous COMP./TENS. figure-level labels
        for txt in self._cb_labels:
            txt.remove()
        self._cb_labels.clear()

        self._cbar_ax.cla()
        sm = cm.ScalarMappable(cmap=self._cmap, norm=self._norm)
        sm.set_array([])
        cb = self.fig.colorbar(sm, cax=self._cbar_ax, orientation='horizontal')
        cb.set_label('Axial Force (N)', color=TEXT_SEC, fontsize=7, labelpad=1)
        cb.ax.xaxis.set_tick_params(color=TEXT_SEC, labelsize=7, pad=1)
        plt.setp(cb.ax.xaxis.get_ticklabels(), color=TEXT_SEC)
        cb.outline.set_edgecolor(BORDER)
        cb.outline.set_linewidth(0.6)
        self._cbar_ax.set_facecolor(TOOLBAR_BG)
        # Labels placed via figure coords so they never clip off-screen
        pos = self._cbar_ax.get_position()
        label_y = pos.y0 + pos.height / 2.0
        t1 = self.fig.text(pos.x0 - 0.015, label_y, 'COMP.',
                           fontsize=8.5, fontweight='bold', color=COMPRESS_C,
                           ha='right', va='center', transform=self.fig.transFigure)
        t2 = self.fig.text(pos.x1 + 0.015, label_y, 'TENS.',
                           fontsize=8.5, fontweight='bold', color=TENSION_C,
                           ha='left', va='center', transform=self.fig.transFigure)
        self._cb_labels = [t1, t2]

    # ══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _style_2d_ax(self, ax, title, xlabel, ylabel, xlim, ylim,
                     aspect='equal'):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_SEC, labelsize=9, length=4, width=0.6,
                       pad=3)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)
        ax.set_xlabel(xlabel, color=TEXT_SEC, fontsize=10, labelpad=8)
        ax.set_ylabel(ylabel, color=TEXT_SEC, fontsize=10, labelpad=8)
        ax.set_title(title, color=TEXT_HEAD, fontsize=13, pad=10, fontweight='bold')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect(aspect)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)

    def _save_png(self):
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"bridge_{self._current_view.replace(' ','_').lower()}.png")
        self.fig.savefig(out, dpi=250, facecolor=TOOLBAR_BG,
                         edgecolor='none', pad_inches=0.1)
        print(f"  Saved -> {out}")

    def show(self):
        plt.show()
