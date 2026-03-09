"""
main.py
-------
Entry point for Bridge Analyser v2.

Usage:
    python main.py

Controls:
    3D view  —  left-click drag to rotate, scroll to zoom, right-click drag to pan
    Buttons  —  left sidebar: switch views, toggle labels/deformed/data, save PNG
"""

import sys
from bridge.geometry import build_nodes, build_members
from bridge.solver   import solve, SectionDef
from bridge.budget   import compute_budget
from bridge.visualiser import BridgeVisualiser


# ── Section assignments per member group ─────────────────────────────
# Matched to Balsa Central 450 Economy Pack inventory:
#   4× 12×12×450mm sticks  |  2× 5×75×450mm sheets
#   2× 3×100×450mm sheets  |  10× 1.5×75×450mm sheets
SECTIONS = {
    "bottom":      SectionDef(12.0, 12.0),   # primary tension — 12×12 stick
    "top":         SectionDef(12.0, 12.0),   # primary compression — 12×12 stick
    "endpost":     SectionDef(5.0, 12.0),    # high compression — strip from 5×75 sheet
    "vertical":    SectionDef(3.0, 12.0),    # compression — strip from 3×100 sheet
    "diagonal":    SectionDef(3.0, 12.0),    # tension — strip from 3×100 sheet
    "floor":       SectionDef(1.5, 12.0),    # floor cross — strip from 1.5×75 sheet
    "floorcentre": SectionDef(5.0, 12.0),    # centre load beam — strip from 5×75 sheet
    "floordiag":   SectionDef(1.5, 12.0),    # floor diagonal — strip from 1.5×75 sheet
    "topbrace":    SectionDef(1.5, 12.0),    # top bracing — strip from 1.5×75 sheet
}

BRIDGE_WIDTH = 50.0   # mm — competition minimum (narrower = lighter = better score)


def main():
    print("=" * 60)
    print("  Bridge Analyser v2 — Competition Mode")
    print("  Howe Truss — Balsawood | 500mm span")
    print("  METRIC: score = failure_load / bridge_weight")
    print("=" * 60)

    # Build geometry
    nodes   = build_nodes(width=BRIDGE_WIDTH)
    members = build_members()
    print(f"\n  Nodes: {len(nodes)}   Members: {len(members)}")
    print(f"  Width: {BRIDGE_WIDTH} mm")

    # Sections summary
    print("\n  SECTIONS:")
    seen = set()
    for grp, sec in SECTIONS.items():
        key = (sec.b, sec.h)
        if key not in seen:
            print(f"    {sec.b}×{sec.h} mm  A={sec.A:.1f} mm²  I_min={sec.I_min:.2f} mm⁴")
            seen.add(key)

    # Solve
    result = solve(nodes, members, sections=SECTIONS)

    # ── COMPETITION SCORE (hero metric) ──
    score, failure_N, failure_mode, crit_idx = result.compute_score()
    failure_kg = failure_N / 9.81
    weight = result.bridge_weight_grams()

    print("\n" + "─" * 60)
    print(f"  ★  COMPETITION SCORE:  {score:.1f} N/g")
    print(f"  ★  Predicted failure:  {failure_N:.0f} N  ({failure_kg:.1f} kg)")
    print(f"  ★  Failure mode:       {failure_mode}")
    if crit_idx >= 0:
        n1, n2, grp = members[crit_idx]
        print(f"  ★  Critical member:    {n1}-{n2} ({grp})")
    print("─" * 60)

    # Weight breakdown
    mat_w = result.material_weight_grams()
    glue_w = result.glue_weight_grams()
    print(f"\n  WEIGHT:")
    print(f"    Material:   {mat_w:.1f} g")
    print(f"    Glue est:   {glue_w:.1f} g  ({result.n_joints} joints × 0.3g)")
    print(f"    Total:      {weight:.1f} g  {'(OK)' if result.within_weight_limit else '(OVER 200g LIMIT!)'}")

    # Safety factors
    d4, d16 = result.centre_deflection()
    sf_t = result.safety_factor_tension()
    sf_c = result.safety_factor_compression()
    sf_b = result.safety_factor_buckling()

    print(f"\n  ANALYSIS (at {result.load_N:.1f} N = 10 kg):")
    print(f"    Deflection:     {abs(d4):.5f} mm  (node 4)")
    print(f"    SF tension:     {sf_t:.2f}×")
    print(f"    SF compression: {sf_c:.2f}×")
    print(f"    SF buckling:    {sf_b:.2f}×")

    worst = result.worst_safety_factor()
    if worst >= 1.0:
        print(f"\n  BRIDGE HOLDS at 10 kg  (worst SF = {worst:.2f}×)")
    else:
        print(f"\n  BRIDGE FAILS at 10 kg  (worst SF = {worst:.2f}×)")

    # Material budget
    budget, budget_ok = compute_budget(members, result.lengths, SECTIONS)
    print(f"\n  MATERIAL BUDGET:")
    print(f"  {'Stock':<16} {'Used':>8} {'Avail':>8} {'Pieces':>8} {'Status'}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for su in budget:
        status = "OVER!" if su.over_budget else "OK"
        print(f"  {su.name:<16} {su.total_used_mm:>7.0f}mm {su.total_available_mm:>7.0f}mm "
              f"{su.pieces_used:>7.1f}/{su.qty_available:<2} {status}")
    if not budget_ok:
        print("  ⚠  MATERIAL BUDGET EXCEEDED — redesign needed!")

    # Top 5 critical members
    status_list = result.member_status()
    print(f"\n  CRITICAL MEMBERS (top 5):")
    print(f"  {'#':>3}  {'Group':<10} {'Force':>8} {'MatSF':>7} {'BuckSF':>7} {'Status'}")
    print(f"  {'---':>3}  {'----------':<10} {'--------':>8} {'-------':>7} {'-------':>7} {'------'}")
    for row in status_list[:5]:
        i = row['idx']
        bsf = f"{row['buckling_SF']:.1f}" if row['buckling_SF'] != float('inf') else "  -"
        msf = f"{row['material_SF']:.1f}" if row['material_SF'] != float('inf') else "  -"
        n1, n2 = row['n1'], row['n2']
        print(f"  {n1:>2}-{n2:<2} {row['group']:<10} {row['force']:>+8.2f} {msf:>7} {bsf:>7} {row['status']}")

    # Build warnings
    print(f"\n  ⚠  BUILD WARNINGS:")
    print(f"    1. Hook/deck attachment is a known failure point — reinforce with extra glue area")
    print(f"    2. Bearing surfaces must be flat & smooth — uneven contact causes torsional failure")
    print(f"    3. Joint quality is critical — apply glue evenly, clamp until set")

    print(f"\n  Launching interactive visualiser...")
    print("=" * 60)

    # Launch GUI
    viz = BridgeVisualiser(result)
    viz.show()


if __name__ == "__main__":
    main()
