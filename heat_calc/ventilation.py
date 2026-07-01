"""Ventilation strategies: time-dependent air change rate.

The most effective lever against summer overheating is *night ventilation*:
during the day the windows stay closed (keep the warm air out), at night you
ventilate strongly to "cool down" the storage mass.

A strategy returns, for every point in time, the air change rate (ACH, air
changes per hour) in 1/h. From the ACH the model derives the ventilation
conductance:

    H_V = ACH · volume · ρ_air · c_air / 3600      [W/K]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class VentilationStrategy:
    """Time- (and optionally temperature-) dependent air change rate.

    Attributes:
        day_ach:      Air change rate during the day in 1/h (windows mostly closed).
        night_ach:    Air change rate at night in 1/h (cross ventilation, windows open).
        night_start:  Start of night ventilation (local hour, 0-23).
        night_end:    End of night ventilation (local hour, 0-23).
        smart:        If True, extra ventilation at night only happens when it is
                      actually cooler outside than inside (otherwise an open
                      window would provide no cooling).
        utc_offset:   Offset of the weather timestamps (UTC) to local time in
                      hours (e.g. +2 for Central Europe in summer).
    """

    day_ach: float = 0.4
    night_ach: float = 2.5
    night_start: int = 21
    night_end: int = 7
    smart: bool = True
    utc_offset: float = 0.0

    def is_night(self, when: datetime) -> bool:
        """Does the (local-time-converted) instant fall within the night window?"""
        local = (when.hour + when.minute / 60.0 + self.utc_offset) % 24
        if self.night_start <= self.night_end:
            return self.night_start <= local < self.night_end
        # Window across midnight (e.g. 21:00 -> 07:00)
        return local >= self.night_start or local < self.night_end

    def ach(self, when: datetime, t_out: float, t_in: float) -> float:
        """Air change rate [1/h] for a given time and current temperatures."""
        if not self.is_night(when):
            return self.day_ach
        if self.smart and t_out >= t_in:
            # Not cooler outside -> night ventilation would achieve nothing.
            return self.day_ach
        return self.night_ach


def constant_strategy(ach: float) -> VentilationStrategy:
    """Convenience constructor for a constant air change rate (no night ventilation)."""
    return VentilationStrategy(day_ach=ach, night_ach=ach, smart=False)
