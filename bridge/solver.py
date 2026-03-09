"""
solver.py
---------
3D truss solver using the direct stiffness method.
Assumptions:
  - Pin joints (no moment transfer) — pure bar elements
  - 3 DOF per node: Ux, Uy, Uz
  - Linear elastic, small displacements

Returns a SolverResult dataclass with:
  U           — full nodal displacement vector  (n_nodes*3,)
  forces      — axial force in each member (N), +ve = tension
  stresses    — axial stress in each member (N/mm²)
  reactions   — reaction forces at supported DOF
  lengths     — member lengths (mm)
  sections    — per-group SectionDef mapping
  areas       — per-member cross-sectional area (mm²)
  buckling_Pcr — Euler critical load per member (N)
  buckling_SF  — buckling safety factor per compression member
"""

from dataclasses import dataclass, field
import math
import numpy as np


# ── Material defaults — Balsawood ────────────────────────────────────
DEFAULT_E          = 3700.0    # N/mm²  Balsawood MOE (along grain)
BALSA_TENSILE      = 20.0     # N/mm²  tensile strength (conservative)
BALSA_COMPRESSIVE  = 12.0     # N/mm²  compressive strength (conservative)
BALSA_SHEAR        = 3.0      # N/mm²  shear strength
BALSA_DENSITY      = 160.0    # kg/m³  (medium-density balsa)
LOAD_TOTAL         = 98.1     # N      10 kg × 9.81
WEIGHT_LIMIT_G     = 200.0    # grams — competition max
GLUE_PER_JOINT_G   = 0.3      # grams per joint (balsa cement estimate)


@dataclass
class SectionDef:
    """Cross-section definition for a rectangular member."""
    b: float   # width  (mm)
    h: float   # height (mm)

    @property
    def A(self) -> float:
        return self.b * self.h

    @property
    def I_strong(self) -> float:
        """Second moment about strong axis (b × h³/12)."""
        return self.b * self.h**3 / 12.0

    @property
    def I_weak(self) -> float:
        """Second moment about weak axis (h × b³/12)."""
        return self.h * self.b**3 / 12.0

    @property
    def I_min(self) -> float:
        return min(self.I_strong, self.I_weak)

    def __repr__(self):
        return f"SectionDef({self.b}×{self.h} mm, A={self.A:.1f} mm²)"


# Default section: 5×5 mm square
DEFAULT_SECTION = SectionDef(5.0, 5.0)


@dataclass
class SolverResult:
    nodes:      dict                    # {id: (x,y,z)}
    members:    list                    # [(n1,n2,group), ...]
    U:          np.ndarray              # nodal displacements, shape (n_dof,)
    forces:     np.ndarray              # member axial forces, shape (n_members,)
    stresses:   np.ndarray              # member stresses, shape (n_members,)
    reactions:  np.ndarray              # reaction at each fixed DOF
    lengths:    np.ndarray              # member lengths
    E:          float
    load_N:     float
    sections:   dict                    # {group: SectionDef}
    areas:      np.ndarray              # per-member area (mm²)
    density:    float = BALSA_DENSITY   # kg/m³
    tensile_str:    float = BALSA_TENSILE
    compressive_str: float = BALSA_COMPRESSIVE
    shear_str:      float = BALSA_SHEAR
    buckling_Pcr: np.ndarray = field(default_factory=lambda: np.array([]))
    buckling_SF:  np.ndarray = field(default_factory=lambda: np.array([]))
    n_joints:   int = 0               # number of glued joints (for weight estimate)

    def safety_factor_tension(self):
        """Minimum tension SF across all members (stress-based)."""
        tension_mask = self.stresses > 0
        if not tension_mask.any():
            return float('inf')
        return float(self.tensile_str / self.stresses[tension_mask].max())

    def safety_factor_compression(self):
        """Minimum compression SF across all members (stress-based)."""
        comp_mask = self.stresses < 0
        if not comp_mask.any():
            return float('inf')
        return float(self.compressive_str / np.abs(self.stresses[comp_mask]).max())

    def safety_factor_buckling(self):
        """Minimum buckling SF across compression members."""
        if self.buckling_SF.size == 0:
            return float('inf')
        comp_mask = self.forces < -1e-6
        if not comp_mask.any():
            return float('inf')
        return float(self.buckling_SF[comp_mask].min())

    def worst_safety_factor(self):
        """Overall governing safety factor (min of tension, compression, buckling)."""
        return min(self.safety_factor_tension(),
                   self.safety_factor_compression(),
                   self.safety_factor_buckling())

    def bridge_weight_grams(self):
        """Estimated bridge weight in grams from member volumes, density, and glue."""
        volumes_mm3 = self.areas * self.lengths   # mm³ per member
        total_mm3 = volumes_mm3.sum()
        total_m3 = total_mm3 * 1e-9              # mm³ → m³
        mass_kg = total_m3 * self.density
        material_g = mass_kg * 1000.0
        glue_g = self.n_joints * GLUE_PER_JOINT_G
        return material_g + glue_g

    def material_weight_grams(self):
        """Material weight only (no glue)."""
        volumes_mm3 = self.areas * self.lengths
        return volumes_mm3.sum() * 1e-9 * self.density * 1000.0

    def glue_weight_grams(self):
        """Estimated glue weight."""
        return self.n_joints * GLUE_PER_JOINT_G

    def load_weight_ratio(self):
        """Load:Weight ratio at applied load (higher = better)."""
        w = self.bridge_weight_grams()
        if w <= 0:
            return float('inf')
        return (self.load_N / 9.81 * 1000.0) / w   # load_g / weight_g

    def predicted_failure_load(self):
        """Predicted failure load (N) — load at which worst SF reaches 1.0.

        Since forces scale linearly with applied load, failure load =
        applied_load × worst_SF.

        Returns (failure_load_N, failure_mode, critical_member_idx).
        """
        worst_sf = float('inf')
        worst_mode = "none"
        worst_idx = -1

        for i, (n1, n2, grp) in enumerate(self.members):
            f = self.forces[i]
            s = self.stresses[i]

            # Material failure
            if f > 1e-6:
                sf = self.tensile_str / s
                if sf < worst_sf:
                    worst_sf, worst_mode, worst_idx = sf, "tension", i
            elif f < -1e-6:
                sf = self.compressive_str / abs(s)
                if sf < worst_sf:
                    worst_sf, worst_mode, worst_idx = sf, "compression", i

            # Buckling failure
            if self.buckling_SF.size > 0 and f < -1e-6:
                bsf = self.buckling_SF[i]
                if bsf < worst_sf:
                    worst_sf, worst_mode, worst_idx = bsf, "buckling", i

        if worst_sf == float('inf'):
            return float('inf'), "none", -1

        failure_load_N = self.load_N * worst_sf
        return failure_load_N, worst_mode, worst_idx

    def compute_score(self):
        """Competition score = failure_load_N / bridge_weight_g.

        This is THE metric that determines the winner.
        Returns (score, failure_load_N, failure_mode, critical_idx).
        """
        failure_N, mode, idx = self.predicted_failure_load()
        w = self.bridge_weight_grams()
        if w <= 0:
            return float('inf'), failure_N, mode, idx
        score = failure_N / w
        return score, failure_N, mode, idx

    @property
    def within_weight_limit(self):
        return self.bridge_weight_grams() <= WEIGHT_LIMIT_G

    def max_deflection(self):
        """Maximum downward (–Y) displacement across all nodes."""
        y_disps = [self.U[_dofs(n)[1]] for n in self.nodes]
        return min(y_disps)

    def centre_deflection(self):
        """Deflection at load nodes (4 and 16)."""
        return self.U[_dofs(4)[1]], self.U[_dofs(16)[1]]

    def member_status(self):
        """Return sorted list of member data dicts, most critical first.

        Each dict: idx, group, force, stress, material_SF, buckling_SF, governing_SF, status
        """
        rows = []
        for i, (n1, n2, grp) in enumerate(self.members):
            f = self.forces[i]
            s = self.stresses[i]
            if f > 1e-6:
                mat_sf = self.tensile_str / s
            elif f < -1e-6:
                mat_sf = self.compressive_str / abs(s)
            else:
                mat_sf = float('inf')
            buck_sf = self.buckling_SF[i] if self.buckling_SF.size > 0 else float('inf')
            gov_sf = min(mat_sf, buck_sf)
            if gov_sf == float('inf'):
                status = "OK"
            elif gov_sf >= 2.0:
                status = "OK"
            elif gov_sf >= 1.0:
                status = "WARN"
            else:
                status = "FAIL"
            rows.append({
                'idx': i,
                'n1': n1, 'n2': n2,
                'group': grp,
                'force': f,
                'stress': s,
                'material_SF': mat_sf,
                'buckling_SF': buck_sf,
                'governing_SF': gov_sf,
                'status': status,
            })
        rows.sort(key=lambda r: r['governing_SF'])
        return rows


def _dofs(n):
    b = (n - 1) * 3
    return [b, b + 1, b + 2]


def solve(nodes: dict, members: list, E: float = DEFAULT_E,
          sections: dict | None = None,
          load_N: float = LOAD_TOTAL) -> SolverResult:
    """
    Run the direct stiffness analysis.

    Parameters
    ----------
    nodes    : {id: (x,y,z)}
    members  : [(n1, n2, group), ...]
    E        : Young's modulus (N/mm²)
    sections : {group_name: SectionDef} — per-group cross-sections.
               If None, all members use DEFAULT_SECTION.
    load_N   : Total central point load (N), split equally across nodes 4 & 16
    """
    if sections is None:
        sections = {}

    # Build per-member area array
    n_members = len(members)
    areas = np.zeros(n_members)
    I_mins = np.zeros(n_members)
    for i, (n1, n2, grp) in enumerate(members):
        sec = sections.get(grp, DEFAULT_SECTION)
        areas[i] = sec.A
        I_mins[i] = sec.I_min

    n_dof = len(nodes) * 3
    K = np.zeros((n_dof, n_dof))
    lengths = []

    for idx, (n1, n2, _) in enumerate(members):
        x1, y1, z1 = nodes[n1]
        x2, y2, z2 = nodes[n2]
        dx, dy, dz = x2-x1, y2-y1, z2-z1
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        lengths.append(L)
        T  = np.array([dx/L, dy/L, dz/L])
        kl = (E * areas[idx] / L) * np.outer(
            np.concatenate([-T, T]),
            np.concatenate([-T, T])
        )
        ds = _dofs(n1) + _dofs(n2)
        for i, gi in enumerate(ds):
            for j, gj in enumerate(ds):
                K[gi, gj] += kl[i, j]

    lengths = np.array(lengths)

    # Boundary conditions
    # Pin (fixed) at all four support nodes: 1, 7, 13, 19 → fix X, Y, Z
    fixed = []
    for n in [1, 7, 13, 19]:
        fixed += _dofs(n)
    fixed = sorted(set(fixed))
    free  = [i for i in range(n_dof) if i not in fixed]

    # Load vector
    F = np.zeros(n_dof)
    half = load_N / 2.0
    F[_dofs(4)[1]]  = -half
    F[_dofs(16)[1]] = -half

    # Solve
    K_ff = K[np.ix_(free, free)]
    F_f  = F[free]
    U_f  = np.linalg.solve(K_ff, F_f)

    U = np.zeros(n_dof)
    for i, d in enumerate(free):
        U[d] = U_f[i]

    # Reactions
    reactions = K[np.ix_(fixed, list(range(n_dof)))] @ U

    # Member forces & stresses (using per-member area)
    forces = np.zeros(n_members)
    for idx, (n1, n2, _) in enumerate(members):
        x1,y1,z1 = nodes[n1]; x2,y2,z2 = nodes[n2]
        dx,dy,dz  = x2-x1,y2-y1,z2-z1
        L = lengths[idx]
        T = np.array([dx/L, dy/L, dz/L])
        forces[idx] = (E * areas[idx] / L) * np.dot(T, U[_dofs(n2)] - U[_dofs(n1)])

    stresses = forces / areas

    # Euler buckling: P_cr = π²EI_min / L²  (for all members)
    buckling_Pcr = (math.pi**2 * E * I_mins) / (lengths**2)
    # Buckling SF: only meaningful for compression members (force < 0)
    buckling_SF = np.full(n_members, float('inf'))
    comp_mask = forces < -1e-6
    buckling_SF[comp_mask] = buckling_Pcr[comp_mask] / np.abs(forces[comp_mask])

    # Estimate number of joints (each member end is a joint, shared at nodes)
    # Approximate: each node is one glued joint
    n_joints = len(nodes)

    return SolverResult(
        nodes=nodes,
        members=members,
        U=U,
        forces=forces,
        stresses=stresses,
        reactions=reactions,
        lengths=lengths,
        E=E,
        load_N=load_N,
        sections=sections,
        areas=areas,
        buckling_Pcr=buckling_Pcr,
        buckling_SF=buckling_SF,
        n_joints=n_joints,
    )
