"""Physical core calculation: the heat balance of the apartment.

The indoor temperature T_indoor is modelled as *one* thermal node (first-order
RC model). The heat balance reads:

    C · dT/dt = Q_solar(t) + Q_convective(t) − Q_loss(t)

We combine the convective gain and the envelope loss into a single conductance
term, because both are the same physics -- only with opposite sign:

    Q_envelope(t) = UA · (T_out(t) − T_indoor(t))

    * T_out > T_indoor  ->  Q_envelope > 0  (heat flows in  = "convective")
    * T_out < T_indoor  ->  Q_envelope < 0  (heat flows out = "loss")

The solar gain through the windows is always a gain:

    Q_solar(t) = Σ_windows  area · g_effective · irradiance_on_facade(t)

Integration is done explicitly with the Euler method (default) or optionally
with scipy.integrate.solve_ivp. The time step is freely selectable (e.g. 15 min).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

from .building import Building
from .ventilation import VentilationStrategy
from .weather import WeatherSeries, irradiance_on_surface, solar_position


@dataclass
class SimulationResult:
    """Result of a simulation.

    All time series have the same length as ``times``.

    ``t_in`` is the perceived indoor (air) temperature. For the 2-node model
    ``t_mass`` (temperature of the storage mass) is additionally filled.
    """

    times: List[datetime]
    t_in: np.ndarray          # indoor/air temperature [°C]
    t_out: np.ndarray         # outdoor temperature [°C]
    q_solar: np.ndarray       # solar gain [W]
    q_envelope: np.ndarray    # envelope flow [W] (+ = gain, − = loss)
    q_total_solar_irr: np.ndarray  # weighted irradiance sum on the windows [W]
    building: Building
    dt_seconds: float
    # optional fields (2-node model / ventilation)
    t_mass: Optional[np.ndarray] = None          # temperature of the storage mass [°C]
    ach_series: Optional[np.ndarray] = None       # actual air change rate [1/h]
    tau_fast_h: Optional[float] = None            # fast time constant [h]
    tau_slow_h: Optional[float] = None            # slow ("reverberating") one [h]

    @property
    def model_name(self) -> str:
        return "2-node" if self.t_mass is not None else "1-node"

    # ---- evaluation ---------------------------------------------------------

    def peak_indoor(self) -> tuple:
        """(time, temperature) of the indoor-temperature maximum."""
        i = int(np.argmax(self.t_in))
        return self.times[i], float(self.t_in[i])

    def peak_outdoor(self) -> tuple:
        """(time, temperature) of the outdoor-temperature maximum."""
        i = int(np.argmax(self.t_out))
        return self.times[i], float(self.t_out[i])

    def phase_shift_hours(self) -> float:
        """Time shift of the indoor vs. the outdoor maximum (hours).

        Positive = the indoor maximum arrives *later* -- exactly the effect of
        why the apartment is still hot in the evening.
        """
        t_in_peak, _ = self.peak_indoor()
        t_out_peak, _ = self.peak_outdoor()
        return (t_in_peak - t_out_peak).total_seconds() / 3600.0

    def overheating_hours(self, threshold: float = 26.0) -> float:
        """Number of hours the indoor temperature stays above ``threshold`` °C.

        A simple thermal-comfort metric: 26 °C is a common daytime comfort
        limit, 28 °C is often used for bedrooms/night. Computed by summing the
        time steps whose indoor temperature exceeds the threshold.
        """
        exceed = np.count_nonzero(self.t_in > threshold)
        return exceed * self.dt_seconds / 3600.0

    def energy_shares(self) -> dict:
        """Share of solar vs. convective heat gains (positive parts only).

        Heat losses (negative envelope flow) are not counted here -- the
        question is how the *gain paths* split up.
        """
        dt = self.dt_seconds
        solar_energy = float(np.sum(np.clip(self.q_solar, 0, None)) * dt)
        conv_gain = float(np.sum(np.clip(self.q_envelope, 0, None)) * dt)
        total = solar_energy + conv_gain
        if total <= 0:
            return {"solar_share": 0.0, "convective_share": 0.0,
                    "solar_kWh": 0.0, "convective_kWh": 0.0}
        return {
            "solar_share": solar_energy / total,
            "convective_share": conv_gain / total,
            "solar_kWh": solar_energy / 3.6e6,
            "convective_kWh": conv_gain / 3.6e6,
        }

    def report(self) -> str:
        """Human-readable summary."""
        t_in_time, t_in_val = self.peak_indoor()
        t_out_time, t_out_val = self.peak_outdoor()
        shares = self.energy_shares()
        b = self.building
        lines = [
            f"Scenario: {b.name}   [{self.model_name} model]",
        ]
        if self.tau_slow_h is not None:
            if self.tau_fast_h < 1.0:
                fast = f"{self.tau_fast_h * 60:.0f} min"
            else:
                fast = f"{self.tau_fast_h:.1f} h"
            lines.append(
                f"  Time constants tau     : fast {fast} "
                f"(air) / slow {self.tau_slow_h:.1f} h (storage mass)"
            )
        else:
            lines.append(
                f"  Time constant tau      : {b.time_constant_hours():.1f} h "
                f"(how long heat keeps reverberating)"
            )
        lines += [
            f"  Storage capacity C     : {b.thermal_capacitance() / 1000:.0f} kJ/K",
            f"  Total conductance UA   : {b.total_ua():.1f} W/K",
            f"  Max. outdoor temp.     : {t_out_val:.1f} °C at "
            f"{t_out_time.strftime('%d %b %H:%M')} UTC",
            f"  Max. indoor temp.      : {t_in_val:.1f} °C at "
            f"{t_in_time.strftime('%d %b %H:%M')} UTC",
            f"  Phase shift            : {self.phase_shift_hours():+.1f} h "
            f"(indoor maximum later than outdoor)",
            f"  Overheating (>26 °C)   : {self.overheating_hours(26.0):.0f} h "
            f"(>28 °C: {self.overheating_hours(28.0):.0f} h)",
            f"  Heat gain solar        : {shares['solar_share'] * 100:.0f} % "
            f"({shares['solar_kWh']:.1f} kWh)",
            f"  Heat gain convective   : {shares['convective_share'] * 100:.0f} % "
            f"({shares['convective_kWh']:.1f} kWh)",
        ]
        if self.ach_series is not None:
            lines.append(
                f"  Air change avg/max     : {float(np.mean(self.ach_series)):.1f} / "
                f"{float(np.max(self.ach_series)):.1f} 1/h"
            )
        if self.t_mass is not None:
            lines.append(
                f"  Storage mass max.      : {float(np.max(self.t_mass)):.1f} °C "
                f"(more sluggish than the air)"
            )
        return "\n".join(lines)


def _solar_gain_watts(
    building: Building, weather: WeatherSeries, when: datetime
) -> tuple:
    """Solar heat gain [W] and the weighted window irradiance [W].

    The irradiance is projected onto each window's facade individually and
    multiplied by area · effective g-value.
    """
    elev, azi = solar_position(weather.latitude, weather.longitude, when)
    dni = _interp(weather, weather.dni, when)
    dhi = _interp(weather, weather.dhi, when)
    ghi = _interp(weather, weather.ghi, when)

    q = 0.0
    irr_weighted = 0.0
    for win in building.windows:
        irr = irradiance_on_surface(
            win.azimuth, dni, dhi, ghi, elev, azi
        )
        q += win.area * win.effective_g * irr
        irr_weighted += win.area * irr
    return q, irr_weighted


def _interp(weather: WeatherSeries, series: np.ndarray, when: datetime) -> float:
    """Linear interpolation of an hourly quantity onto the instant ``when``."""
    t0 = weather.times[0]
    x = (when - t0).total_seconds()
    xs = np.array([(t - t0).total_seconds() for t in weather.times])
    return float(np.interp(x, xs, series))


def simulate(
    building: Building,
    weather: WeatherSeries,
    dt_minutes: float = 15.0,
    t_start: Optional[float] = None,
    method: str = "euler",
    ventilation: Optional[VentilationStrategy] = None,
) -> SimulationResult:
    """Simulate the indoor temperature over time (1-node model).

    Args:
        building:    Building configuration (provides C and UA).
        weather:     Hourly weather time series.
        dt_minutes:  Integration time step in minutes.
        t_start:     Initial indoor temperature in °C. Default: outdoor
                     temperature at the start time (settled night-time state).
        method:      'euler' (explicit) or 'scipy' (solve_ivp). A time-dependent
                     ventilation strategy is only honoured by 'euler'.
        ventilation: Optional ventilation strategy (e.g. night ventilation).
                     Without it, the constant ``building.ventilation_ach`` applies.

    Returns:
        SimulationResult with all time series and evaluation methods.
    """
    C = building.thermal_capacitance()
    UA = building.total_ua()

    start = weather.times[0]
    end = weather.times[-1]
    dt = timedelta(minutes=dt_minutes)
    n_steps = int((end - start).total_seconds() // dt.total_seconds()) + 1

    times = [start + i * dt for i in range(n_steps)]
    t_in = np.zeros(n_steps)
    t_out = np.array([_interp(weather, weather.t_out, t) for t in times])
    q_solar = np.zeros(n_steps)
    q_env = np.zeros(n_steps)
    q_irr = np.zeros(n_steps)
    ach_series = np.zeros(n_steps) if ventilation else None

    t_in[0] = t_out[0] if t_start is None else t_start
    dt_s = dt.total_seconds()

    if method == "scipy":
        return _simulate_scipy(building, weather, times, t_out, C, UA, dt_s)

    ua_transmission = building.transmission_ua()

    # Explicit Euler method:  T_{n+1} = T_n + (dt/C) · ΣQ
    for i in range(n_steps):
        qs, irr = _solar_gain_watts(building, weather, times[i])
        if ventilation is not None:
            ach = ventilation.ach(times[i], t_out[i], t_in[i])
            ua_now = ua_transmission + building.ventilation_ua_for_ach(ach)
            ach_series[i] = ach
        else:
            ua_now = UA
        qe = ua_now * (t_out[i] - t_in[i])
        q_solar[i] = qs
        q_env[i] = qe
        q_irr[i] = irr
        if i + 1 < n_steps:
            t_in[i + 1] = t_in[i] + dt_s / C * (qs + qe)

    return SimulationResult(
        times=times,
        t_in=t_in,
        t_out=t_out,
        q_solar=q_solar,
        q_envelope=q_env,
        q_total_solar_irr=q_irr,
        building=building,
        dt_seconds=dt_s,
        ach_series=ach_series,
    )


def _simulate_scipy(building, weather, times, t_out, C, UA, dt_s):
    """Variant using scipy.integrate.solve_ivp (smoother solution)."""
    from scipy.integrate import solve_ivp

    start = times[0]

    def rhs(t_seconds, y):
        when = start + timedelta(seconds=float(t_seconds))
        t_out_now = _interp(weather, weather.t_out, when)
        qs, _ = _solar_gain_watts(building, weather, when)
        qe = UA * (t_out_now - y[0])
        return [(qs + qe) / C]

    t_eval = np.array([(t - start).total_seconds() for t in times])
    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        [t_out[0]],
        t_eval=t_eval,
        method="RK45",
        max_step=dt_s,
    )
    t_in = sol.y[0]

    q_solar = np.zeros(len(times))
    q_env = np.zeros(len(times))
    q_irr = np.zeros(len(times))
    for i, when in enumerate(times):
        qs, irr = _solar_gain_watts(building, weather, when)
        q_solar[i] = qs
        q_irr[i] = irr
        q_env[i] = UA * (t_out[i] - t_in[i])

    return SimulationResult(
        times=times,
        t_in=t_in,
        t_out=t_out,
        q_solar=q_solar,
        q_envelope=q_env,
        q_total_solar_irr=q_irr,
        building=building,
        dt_seconds=dt_s,
    )


# ---------------------------------------------------------------------------
# 2-node model (air + storage mass separated)
# ---------------------------------------------------------------------------

def _two_node_time_constants(
    C_air: float, C_mass: float, H_ao: float, H_mo: float, H_am: float
) -> tuple:
    """The two time constants of the 2-node system in hours.

    The linear system dT/dt = A·T + b(t) has the system matrix

        A = [ -(H_ao+H_am)/C_air ,   H_am/C_air        ]
            [   H_am/C_mass       ,  -(H_mo+H_am)/C_mass ]

    Its (negative) eigenvalues λ give τ = −1/λ. The *large* time constant
    describes the sluggish reverberation of the storage mass, the *small* one
    the fast reaction of the air.
    """
    a11 = -(H_ao + H_am) / C_air
    a12 = H_am / C_air
    a21 = H_am / C_mass
    a22 = -(H_mo + H_am) / C_mass
    trace = a11 + a22
    det = a11 * a22 - a12 * a21
    disc = max(0.0, trace * trace - 4 * det)
    lam1 = (trace + math.sqrt(disc)) / 2.0
    lam2 = (trace - math.sqrt(disc)) / 2.0
    taus = sorted(
        (-1.0 / lam / 3600.0 for lam in (lam1, lam2) if lam < 0),
    )
    if len(taus) < 2:
        return (taus[0] if taus else float("nan"),) * 2
    return taus[0], taus[-1]  # (fast, slow)


def simulate_2node(
    building: Building,
    weather: WeatherSeries,
    dt_minutes: float = 15.0,
    t_start: Optional[float] = None,
    ventilation: Optional[VentilationStrategy] = None,
    solar_to_air_fraction: float = 0.3,
    h_surface: float = 7.7,
) -> SimulationResult:
    """2-node RC model: air and storage mass with their own temperature.

    Why more accurate than a single node? In the single-node model air and
    heavy components share *one* temperature -- so the air would react as
    sluggishly as a concrete ceiling, which underestimates the real heat-up and
    overestimates the phase lag. Here there are two coupled balances:

        C_air · dT_air/dt =
              H_ao·(T_out − T_air)      (windows + ventilation, fast path)
            + H_am·(T_mass − T_air)     (heat exchange with the surfaces)
            + f_air · Q_solar           (fraction of the sun that heats the air)

        C_mass · dT_mass/dt =
              H_mo·(T_out − T_mass)     (transmission through the walls)
            + H_am·(T_air − T_mass)
            + (1−f_air) · Q_solar       (the sun mainly hits floor/walls)

    Args:
        solar_to_air_fraction: Fraction of the solar radiation that directly
            heats the air (the rest heats the storage mass on its surfaces).
        h_surface: interior heat transfer coefficient in W/(m²·K).
        ventilation: optional (night) ventilation strategy; acts on H_ao.
    """
    C_air = building.air_heat_capacity()
    C_mass = building.storage_capacitance()
    if C_mass <= 0:
        # Without storage mass the model degenerates to the single-node case.
        return simulate(building, weather, dt_minutes, t_start,
                        ventilation=ventilation)

    H_mo = building.wall_ua()               # walls -> mass
    H_am = building.internal_coupling(h_surface)
    window_ua = building.window_ua()
    f_air = solar_to_air_fraction

    start = weather.times[0]
    end = weather.times[-1]
    dt = timedelta(minutes=dt_minutes)
    dt_s = dt.total_seconds()
    n_steps = int((end - start).total_seconds() // dt_s) + 1

    times = [start + i * dt for i in range(n_steps)]
    t_out = np.array([_interp(weather, weather.t_out, t) for t in times])
    t_air = np.zeros(n_steps)
    t_mass = np.zeros(n_steps)
    q_solar = np.zeros(n_steps)
    q_env = np.zeros(n_steps)
    q_irr = np.zeros(n_steps)
    ach_series = np.zeros(n_steps)

    t_air[0] = t_out[0] if t_start is None else t_start
    t_mass[0] = t_air[0]

    base_ach = ventilation.day_ach if ventilation else building.ventilation_ach
    max_ach = ventilation.night_ach if ventilation else building.ventilation_ach

    # The air node is "stiff" (small capacity, strong coupling to the mass) ->
    # very short time constant. To keep the explicit Euler method stable, each
    # output step is split internally into n_sub small steps (criterion:
    # sub-step < 0.4 · fastest time constant).
    h_ao_max = window_ua + building.ventilation_ua_for_ach(max_ach)
    fast_tau_air = C_air / (h_ao_max + H_am)
    n_sub = max(1, int(math.ceil(dt_s / (0.4 * fast_tau_air))))
    sub_dt = dt_s / n_sub

    for i in range(n_steps):
        qs, irr = _solar_gain_watts(building, weather, times[i])
        ach = (ventilation.ach(times[i], t_out[i], t_air[i])
               if ventilation else building.ventilation_ach)
        ach_series[i] = ach
        H_ao = window_ua + building.ventilation_ua_for_ach(ach)

        # Diagnostic quantities at the output instant.
        q_solar[i] = qs
        q_irr[i] = irr
        q_env[i] = H_ao * (t_out[i] - t_air[i]) + H_mo * (t_out[i] - t_mass[i])

        if i + 1 < n_steps:
            # Keep the forcing (sun, ventilation) constant across the step,
            # interpolate the outdoor temperature linearly between the samples.
            ta, tm = t_air[i], t_mass[i]
            for k in range(n_sub):
                frac = (k + 0.5) / n_sub
                t_out_sub = t_out[i] + (t_out[i + 1] - t_out[i]) * frac
                flow_air_env = H_ao * (t_out_sub - ta)
                flow_air_mass = H_am * (tm - ta)
                flow_mass_env = H_mo * (t_out_sub - tm)
                dT_air = (flow_air_env + flow_air_mass + f_air * qs) / C_air
                dT_mass = (flow_mass_env - flow_air_mass + (1 - f_air) * qs) / C_mass
                ta += sub_dt * dT_air
                tm += sub_dt * dT_mass
            t_air[i + 1] = ta
            t_mass[i + 1] = tm

    H_ao_base = window_ua + building.ventilation_ua_for_ach(base_ach)
    tau_fast, tau_slow = _two_node_time_constants(
        C_air, C_mass, H_ao_base, H_mo, H_am
    )

    return SimulationResult(
        times=times,
        t_in=t_air,
        t_out=t_out,
        q_solar=q_solar,
        q_envelope=q_env,
        q_total_solar_irr=q_irr,
        building=building,
        dt_seconds=dt_s,
        t_mass=t_mass,
        ach_series=ach_series,
        tau_fast_h=tau_fast,
        tau_slow_h=tau_slow,
    )
