"""Data model of the apartment.

From the building configuration the two central quantities of the RC model
are derived:

    UA  -- heat-transfer conductance of the envelope   [W/K]   (the "1/R")
    C   -- thermal storage capacity                     [J/K]

Electrical analogy:
    temperature  <->  voltage
    heat flow    <->  current
    UA = 1/R     <->  electrical conductance
    C            <->  capacitor

The time constant τ = R·C = C/UA describes how sluggishly the apartment reacts
to temperature changes -- i.e. "how long the heat keeps reverberating".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .materials import AIR, get_material

# Compass direction -> facade azimuth in degrees (0 = north, clockwise).
ORIENTATION_AZIMUTH: Dict[str, float] = {
    "north": 0.0,
    "east": 90.0,
    "south": 180.0,
    "west": 270.0,
}


@dataclass
class Window:
    """A window element of a particular orientation.

    Attributes:
        orientation:  'north' | 'east' | 'south' | 'west'.
        area:         Window area in m².
        u_value:      U-value of the glazing in W/(m²·K) (heat loss).
        g_value:      Total solar energy transmittance (0..1). Fraction of the
                      incident solar radiation that enters the room as heat.
        shading:      Shading factor (0..1). 1 = fully exposed to the sun,
                      0.2 = shutter lets only 20 % through.
    """

    orientation: str
    area: float
    u_value: float = 1.1
    g_value: float = 0.6
    shading: float = 1.0

    @property
    def azimuth(self) -> float:
        key = self.orientation.strip().lower()
        if key not in ORIENTATION_AZIMUTH:
            raise ValueError(
                f"Unknown orientation '{self.orientation}'. "
                f"Allowed: north, east, south, west"
            )
        return ORIENTATION_AZIMUTH[key]

    @property
    def effective_g(self) -> float:
        """Effective g-value including shading."""
        return self.g_value * self.shading


@dataclass
class Wall:
    """Opaque exterior wall / roof surface (no direct solar gain through glass).

    Attributes:
        area:     Area in m².
        u_value:  U-value in W/(m²·K).
    """

    area: float
    u_value: float = 0.3


@dataclass
class StorageMass:
    """A heat-storing building layer (floor, wall, furniture ...).

    The effective heat capacity is:

        C_i = area · thickness · density · c_p        [J/K]

    Only a limited penetration depth of the heat is thermally active; via
    ``active_fraction`` this can be accounted for roughly (default 1.0).

    Attributes:
        material:         Key from the material database (e.g. 'wood').
        area:             Area in m².
        thickness:        Effective thickness in m.
        active_fraction:  Fraction of the mass that participates in storage.
    """

    material: str
    area: float
    thickness: float
    active_fraction: float = 1.0

    def heat_capacity(self) -> float:
        """Heat capacity of this layer in J/K."""
        mat = get_material(self.material)
        volume = self.area * self.thickness
        return volume * mat.volumetric_heat_capacity() * self.active_fraction


@dataclass
class Building:
    """Complete apartment configuration.

    Attributes:
        name:            Scenario label.
        floor_area:      Floor area in m².
        ceiling_height:  Ceiling height in m.
        windows:         List of window elements.
        walls:           List of opaque exterior surfaces.
        storage:         List of storage masses.
        ventilation_ach: Air change rate in 1/h (ventilation/infiltration).
    """

    name: str = "Apartment"
    floor_area: float = 60.0
    ceiling_height: float = 2.7
    windows: List[Window] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)
    storage: List[StorageMass] = field(default_factory=list)
    ventilation_ach: float = 0.5

    # ---- derived quantities -------------------------------------------------

    @property
    def air_volume(self) -> float:
        """Air volume of the apartment in m³."""
        return self.floor_area * self.ceiling_height

    def air_heat_capacity(self) -> float:
        """Heat capacity of the room air in J/K."""
        return self.air_volume * AIR.volumetric_heat_capacity()

    def thermal_capacitance(self) -> float:
        """Total thermal capacity C in J/K (air + storage masses)."""
        c = self.air_heat_capacity()
        for s in self.storage:
            c += s.heat_capacity()
        return c

    def window_ua(self) -> float:
        """Conductance of the windows only in W/K = Σ U·A (fast path to air)."""
        return sum(w.u_value * w.area for w in self.windows)

    def wall_ua(self) -> float:
        """Conductance of the opaque walls/roof only in W/K (path to the mass)."""
        return sum(wall.u_value * wall.area for wall in self.walls)

    def transmission_ua(self) -> float:
        """Transmission conductance of the envelope in W/K = Σ U·A (windows + walls)."""
        return self.window_ua() + self.wall_ua()

    def ventilation_ua_for_ach(self, ach: float) -> float:
        """Ventilation conductance in W/K for a given air change rate.

        H_V = air changes · volume · density · c_p / 3600
        (divided by 3600 because the air change rate is given per hour).
        """
        return ach * self.air_volume * AIR.volumetric_heat_capacity() / 3600.0

    def ventilation_ua(self) -> float:
        """Ventilation conductance in W/K at the configured default air change rate."""
        return self.ventilation_ua_for_ach(self.ventilation_ach)

    def total_ua(self) -> float:
        """Total conductance UA in W/K (transmission + ventilation)."""
        return self.transmission_ua() + self.ventilation_ua()

    # ---- quantities for the 2-node model -----------------------------------

    def storage_capacitance(self) -> float:
        """Heat capacity of the storage masses only in J/K (node 'mass')."""
        return sum(s.heat_capacity() for s in self.storage)

    def internal_surface_area(self) -> float:
        """Interior surface area of the storage masses in m² (for air-mass coupling)."""
        return sum(s.area for s in self.storage)

    def internal_coupling(self, h_surface: float = 7.7) -> float:
        """Conductance air <-> storage mass in W/K.

        H_am = h · A_interior, with h ≈ 7.7 W/(m²·K) as the combined
        convective-radiative heat transfer coefficient at interior surfaces.
        This path is typically well conducting -- the room air therefore
        "feels" the mass quickly, but the mass itself reacts sluggishly because
        of its large capacity. This separation is exactly what the single-node
        model lacks.
        """
        return h_surface * self.internal_surface_area()

    def time_constant_hours(self) -> float:
        """Time constant τ = C / UA in hours.

        Rough interpretation: after ~τ hours about 63 % of a step change in
        temperature has "seeped through"; after 3·τ the apartment has
        practically followed the new outdoor state.
        """
        return self.thermal_capacitance() / self.total_ua() / 3600.0

    def summary(self) -> Dict[str, float]:
        """Compact key figures for output/report."""
        return {
            "air_volume_m3": round(self.air_volume, 1),
            "C_kJ_per_K": round(self.thermal_capacitance() / 1000.0, 1),
            "UA_transmission_W_per_K": round(self.transmission_ua(), 2),
            "UA_ventilation_W_per_K": round(self.ventilation_ua(), 2),
            "UA_total_W_per_K": round(self.total_ua(), 2),
            "tau_hours": round(self.time_constant_hours(), 1),
        }
