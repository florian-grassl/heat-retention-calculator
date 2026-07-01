"""Small material database for the thermal storage mass.

Every material stores heat proportionally to its mass and its specific heat
capacity c_p. The stored amount of heat is:

    Q = m · c_p · ΔT       [J]

with
    m    = mass                      [kg]   = density · volume
    c_p  = specific heat capacity    [J/(kg·K)]
    ΔT   = temperature change        [K]

A heavy concrete floor therefore stores far more heat than an equally thick
wooden layer -- it buffers temperature swings more strongly, but also keeps
releasing that heat well into the evening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Material:
    """Physical properties of a building material.

    Attributes:
        name:            Display name.
        density:         Density in kg/m³.
        specific_heat:   Specific heat capacity c_p in J/(kg·K).
    """

    name: str
    density: float          # kg/m³
    specific_heat: float    # J/(kg·K)

    def volumetric_heat_capacity(self) -> float:
        """Volumetric heat capacity in J/(m³·K) = density · c_p."""
        return self.density * self.specific_heat


# Typical values (reference figures from building physics).
MATERIALS: Dict[str, Material] = {
    "wood":         Material("Wood (spruce)",   density=500.0,  specific_heat=1600.0),
    "concrete":     Material("Concrete",        density=2400.0, specific_heat=1000.0),
    "plaster":      Material("Gypsum plaster",  density=1400.0, specific_heat=1000.0),
    "brick":        Material("Brick masonry",   density=1800.0, specific_heat=840.0),
    "screed":       Material("Cement screed",   density=2000.0, specific_heat=1000.0),
    "plasterboard": Material("Plasterboard",    density=900.0,  specific_heat=1000.0),
}

# Air as a special case (fills the room volume).
AIR = Material("Air", density=1.2, specific_heat=1005.0)


def get_material(key: str) -> Material:
    """Look up a material by its key (case-insensitive)."""
    k = key.strip().lower()
    if k not in MATERIALS:
        raise KeyError(
            f"Unknown material '{key}'. "
            f"Available: {', '.join(sorted(MATERIALS))}"
        )
    return MATERIALS[k]
