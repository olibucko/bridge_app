"""
budget.py
---------
Material inventory tracker for the bridge competition.

Compares material usage against the fixed inventory from
Balsa Central 450 Economy Pack.

All stock pieces are 450 mm long.
"""

from dataclasses import dataclass

# ── Fixed inventory ──────────────────────────────────────────────────
STOCK_LENGTH = 450.0  # mm — all pieces are 450mm long

INVENTORY = [
    {"name": "12x12 stick",  "b": 12.0, "h": 12.0,  "qty": 4,  "length": STOCK_LENGTH},
    {"name": "5x75 sheet",   "b": 5.0,  "h": 75.0,  "qty": 2,  "length": STOCK_LENGTH},
    {"name": "3x100 sheet",  "b": 3.0,  "h": 100.0, "qty": 2,  "length": STOCK_LENGTH},
    {"name": "1.5x75 sheet", "b": 1.5,  "h": 75.0,  "qty": 10, "length": STOCK_LENGTH},
]


@dataclass
class StockUsage:
    """Usage summary for one stock type."""
    name: str
    b: float
    h: float
    qty_available: int
    total_available_mm: float   # total length available
    total_used_mm: float        # total length consumed by members
    members_using: int          # number of members using this stock

    @property
    def pieces_used(self) -> float:
        """Number of 450mm pieces consumed (fractional)."""
        if STOCK_LENGTH <= 0:
            return 0.0
        return self.total_used_mm / STOCK_LENGTH

    @property
    def pieces_remaining(self) -> float:
        return self.qty_available - self.pieces_used

    @property
    def over_budget(self) -> bool:
        return self.pieces_used > self.qty_available + 0.01

    @property
    def utilisation_pct(self) -> float:
        if self.total_available_mm <= 0:
            return 0.0
        return (self.total_used_mm / self.total_available_mm) * 100.0


def compute_budget(members, lengths, sections, nodes=None):
    """Compute material budget usage.

    Parameters
    ----------
    members  : list of (n1, n2, group)
    lengths  : array of member lengths (mm)
    sections : dict {group: SectionDef}
    nodes    : optional dict for joint count

    Returns
    -------
    list of StockUsage, one per inventory item
    bool — True if all within budget
    """
    # Map each (b, h) to a stock item
    # Members are matched to stock by their section dimensions.
    # Sheets can be cut to width, so we match by the sheet thickness (smaller dim).
    usage_mm = {}   # key: (b, h) of stock item -> total length used
    usage_count = {}

    for inv in INVENTORY:
        key = (inv["b"], inv["h"])
        usage_mm[key] = 0.0
        usage_count[key] = 0

    from bridge.solver import DEFAULT_SECTION

    for i, (n1, n2, grp) in enumerate(members):
        sec = sections.get(grp, DEFAULT_SECTION)
        # Find matching stock: exact match on b×h or h×b
        matched = False
        for inv in INVENTORY:
            ib, ih = inv["b"], inv["h"]
            # Exact section match
            if (abs(sec.b - ib) < 0.01 and abs(sec.h - ih) < 0.01) or \
               (abs(sec.b - ih) < 0.01 and abs(sec.h - ib) < 0.01):
                key = (ib, ih)
                usage_mm[key] += lengths[i]
                usage_count[key] += 1
                matched = True
                break
        if not matched:
            # Try matching by smaller dimension (strip cut from sheet)
            sec_min = min(sec.b, sec.h)
            sec_max = max(sec.b, sec.h)
            for inv in INVENTORY:
                ib, ih = inv["b"], inv["h"]
                inv_min = min(ib, ih)
                inv_max = max(ib, ih)
                # Strip cut: thickness matches, width <= sheet width
                if abs(sec_min - inv_min) < 0.01 and sec_max <= inv_max + 0.01:
                    key = (ib, ih)
                    # Account for material waste: actual sheet width consumed
                    waste_factor = sec_max / inv_max if inv_max > 0 else 1.0
                    usage_mm[key] += lengths[i] * waste_factor
                    usage_count[key] += 1
                    matched = True
                    break

        if not matched:
            # Unmatched section — flag it
            pass

    results = []
    all_ok = True
    for inv in INVENTORY:
        key = (inv["b"], inv["h"])
        total_avail = inv["qty"] * inv["length"]
        su = StockUsage(
            name=inv["name"],
            b=inv["b"],
            h=inv["h"],
            qty_available=inv["qty"],
            total_available_mm=total_avail,
            total_used_mm=usage_mm.get(key, 0.0),
            members_using=usage_count.get(key, 0),
        )
        results.append(su)
        if su.over_budget:
            all_ok = False

    return results, all_ok
