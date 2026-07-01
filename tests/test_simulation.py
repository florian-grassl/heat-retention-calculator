"""Unit tests for the physical core calculation.

The focus is on physical edge cases whose outcome can be predicted without
running a simulation -- that is the best safeguard for a numerical model.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from heat_calc.building import Building, StorageMass, Wall, Window
from heat_calc.materials import get_material
from heat_calc.simulation import simulate, simulate_2node
from heat_calc.ventilation import VentilationStrategy, constant_strategy
from heat_calc.weather import synthetic_heatwave


def _demo_building() -> Building:
    return Building(
        floor_area=60, ceiling_height=2.5,
        windows=[Window("south", area=8.0, g_value=0.6)],
        walls=[Wall(area=45.0, u_value=0.3)],
        storage=[StorageMass("concrete", area=60.0, thickness=0.12)],
        ventilation_ach=0.4,
    )


# ---------------------------------------------------------------------------
# Data model / derived quantities
# ---------------------------------------------------------------------------

def test_air_volume_and_capacitance():
    b = Building(floor_area=50.0, ceiling_height=2.5)
    assert b.air_volume == pytest.approx(125.0)
    # air capacity = V * rho * c_p = 125 * 1.2 * 1005
    assert b.air_heat_capacity() == pytest.approx(125 * 1.2 * 1005)


def test_storage_heat_capacity():
    # 10 m² concrete, 10 cm thick: m = 10*0.1*2400 = 2400 kg; C = m*1000 = 2.4e6 J/K
    s = StorageMass(material="concrete", area=10.0, thickness=0.10)
    assert s.heat_capacity() == pytest.approx(2400 * 1000.0)


def test_time_constant_scales_with_capacity():
    """More storage mass -> larger time constant tau."""
    base = Building(
        windows=[Window("south", area=5.0)],
        walls=[Wall(area=40.0, u_value=0.3)],
        ventilation_ach=0.5,
    )
    heavy = Building(
        windows=[Window("south", area=5.0)],
        walls=[Wall(area=40.0, u_value=0.3)],
        storage=[StorageMass("concrete", area=50.0, thickness=0.15)],
        ventilation_ach=0.5,
    )
    assert heavy.time_constant_hours() > base.time_constant_hours()


def test_ua_positive():
    b = Building(
        windows=[Window("south", area=5.0, u_value=1.1)],
        walls=[Wall(area=40.0, u_value=0.3)],
    )
    assert b.transmission_ua() == pytest.approx(5.0 * 1.1 + 40.0 * 0.3)
    assert b.total_ua() > b.transmission_ua()  # + ventilation


# ---------------------------------------------------------------------------
# Edge case 1: no sun, no window solar -> indoor tracks outdoor
# ---------------------------------------------------------------------------

def test_no_sun_indoor_tracks_outdoor_mean():
    """Without solar gain the indoor temperature settles around the mean of the
    outdoor temperature and stays between its min and max."""
    weather = synthetic_heatwave(hours=72, peak_dni=0.0)  # sun off
    b = Building(
        floor_area=50, ceiling_height=2.5,
        windows=[Window("north", area=4.0, g_value=0.0)],  # no solar gain
        walls=[Wall(area=40.0, u_value=0.4)],
        storage=[StorageMass("concrete", area=40.0, thickness=0.1)],
    )
    res = simulate(b, weather, dt_minutes=15)
    # After settling, the indoor temperature must lie within the outdoor band.
    tail = slice(len(res.t_in) // 3, None)
    assert res.t_in[tail].min() >= res.t_out.min() - 0.5
    assert res.t_in[tail].max() <= res.t_out.max() + 0.5


def test_no_gains_constant_outside_stays_constant():
    """Constant outdoor temperature, no solar -> indoor temperature stays constant."""
    weather = synthetic_heatwave(hours=48, peak_dni=0.0)
    weather.t_out[:] = 25.0  # constant outdoor temperature
    b = Building(
        windows=[Window("north", area=2.0, g_value=0.0)],
        walls=[Wall(area=30.0, u_value=0.3)],
    )
    res = simulate(b, weather, dt_minutes=15, t_start=25.0)
    assert np.allclose(res.t_in, 25.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge case 2: perfect insulation (UA -> 0) stores solar heat
# ---------------------------------------------------------------------------

def test_perfect_insulation_only_heats_up():
    """Very good insulation + solar gain: the apartment can only get warmer (no
    meaningful loss path) and barely cools down at night."""
    weather = synthetic_heatwave(hours=48, peak_dni=800.0)
    b = Building(
        floor_area=50, ceiling_height=2.5,
        windows=[Window("south", area=6.0, u_value=0.01, g_value=0.6)],
        walls=[Wall(area=40.0, u_value=0.001)],
        ventilation_ach=0.0,  # no ventilation
        storage=[StorageMass("concrete", area=40.0, thickness=0.1)],
    )
    res = simulate(b, weather, dt_minutes=15, t_start=20.0)
    # Monotonically rising (solar heats, hardly any loss): end well above start.
    assert res.t_in[-1] > res.t_in[0] + 3.0
    # no cooling phase that drops below the start
    assert res.t_in.min() >= res.t_in[0] - 0.1


# ---------------------------------------------------------------------------
# Edge case 3: phase shift due to storage mass
# ---------------------------------------------------------------------------

def test_thermal_mass_causes_phase_lag():
    """More storage mass -> the indoor maximum arrives later (larger phase lag)."""
    weather = synthetic_heatwave(hours=72, peak_dni=700.0)
    light = Building(
        windows=[Window("south", area=6.0)],
        walls=[Wall(area=40.0, u_value=0.3)],
        storage=[StorageMass("wood", area=20.0, thickness=0.02)],
    )
    heavy = Building(
        windows=[Window("south", area=6.0)],
        walls=[Wall(area=40.0, u_value=0.3)],
        storage=[StorageMass("concrete", area=60.0, thickness=0.2)],
    )
    res_light = simulate(light, weather, dt_minutes=15)
    res_heavy = simulate(heavy, weather, dt_minutes=15)
    assert res_heavy.phase_shift_hours() >= res_light.phase_shift_hours()


# ---------------------------------------------------------------------------
# Energy conservation / plausibility
# ---------------------------------------------------------------------------

def test_energy_shares_sum_to_one():
    weather = synthetic_heatwave(hours=48, peak_dni=700.0)
    b = Building(
        windows=[Window("south", area=6.0)],
        walls=[Wall(area=40.0, u_value=0.3)],
        storage=[StorageMass("concrete", area=40.0, thickness=0.1)],
    )
    res = simulate(b, weather, dt_minutes=15)
    shares = res.energy_shares()
    assert shares["solar_share"] + shares["convective_share"] == pytest.approx(1.0)
    assert 0.0 <= shares["solar_share"] <= 1.0


def test_euler_matches_scipy_reasonably():
    """Euler and scipy should be close for a fine time step."""
    weather = synthetic_heatwave(hours=48, peak_dni=700.0)
    b = Building(
        windows=[Window("south", area=6.0)],
        walls=[Wall(area=40.0, u_value=0.3)],
        storage=[StorageMass("concrete", area=40.0, thickness=0.1)],
    )
    res_e = simulate(b, weather, dt_minutes=5, method="euler")
    res_s = simulate(b, weather, dt_minutes=5, method="scipy")
    # small mean deviation
    assert np.mean(np.abs(res_e.t_in - res_s.t_in)) < 0.5


def test_overheating_hours_monotonic_in_threshold():
    """A lower threshold can only ever yield more overheating hours."""
    weather = synthetic_heatwave(hours=72, peak_dni=800.0)
    res = simulate_2node(_demo_building(), weather, dt_minutes=15)
    assert res.overheating_hours(24.0) >= res.overheating_hours(28.0)
    assert res.overheating_hours(100.0) == 0.0  # never that hot


# ---------------------------------------------------------------------------
# Solar position
# ---------------------------------------------------------------------------

def test_solar_noon_elevation_reasonable():
    """Summer noon in Munich: sun high (>55°) and ~south (~180°)."""
    from heat_calc.weather import solar_position
    # 1 July, 11:00 UTC ~ 13:00 CEST (near solar noon in CET+DST)
    when = datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc)
    elev, azi = solar_position(48.14, 11.58, when)
    assert elev > 55.0
    assert 150.0 < azi < 210.0


def test_material_lookup():
    assert get_material("CONCRETE").density == 2400
    with pytest.raises(KeyError):
        get_material("unobtanium")


# ---------------------------------------------------------------------------
# 2-node model
# ---------------------------------------------------------------------------

def test_two_node_is_numerically_stable():
    """The stiff air node must not "explode" at a coarse time step."""
    weather = synthetic_heatwave(hours=72, peak_dni=800.0)
    res = simulate_2node(_demo_building(), weather, dt_minutes=30)
    assert np.all(np.isfinite(res.t_in))
    assert np.all(np.isfinite(res.t_mass))
    # Temperatures stay within a physically plausible band.
    assert res.t_in.max() < 70.0
    assert res.t_in.min() > 0.0


def test_two_node_reports_two_time_constants():
    """There is a fast (air) and a slow (mass) time constant."""
    weather = synthetic_heatwave(hours=48, peak_dni=700.0)
    res = simulate_2node(_demo_building(), weather, dt_minutes=15)
    assert res.t_mass is not None
    assert res.tau_fast_h is not None and res.tau_slow_h is not None
    # The mass is more sluggish than the air.
    assert res.tau_slow_h > res.tau_fast_h
    assert res.model_name == "2-node"


def test_two_node_mass_is_more_sluggish_than_air():
    """The storage mass swings less than the room air."""
    weather = synthetic_heatwave(hours=72, peak_dni=800.0)
    res = simulate_2node(_demo_building(), weather, dt_minutes=15)
    air_swing = res.t_in.max() - res.t_in.min()
    mass_swing = res.t_mass.max() - res.t_mass.min()
    assert mass_swing <= air_swing + 1e-6


def test_two_node_without_storage_falls_back_to_one_node():
    """Without storage mass the 2-node model degenerates to the 1-node model."""
    weather = synthetic_heatwave(hours=48, peak_dni=600.0)
    b = Building(
        windows=[Window("south", area=6.0)],
        walls=[Wall(area=40.0, u_value=0.3)],
        storage=[],  # no mass
    )
    res = simulate_2node(b, weather, dt_minutes=15)
    assert res.t_mass is None  # 1-node result


def test_two_node_no_sun_stays_in_band():
    """Without solar gain both nodes stay within the outdoor-temperature band."""
    weather = synthetic_heatwave(hours=72, peak_dni=0.0)
    b = Building(
        floor_area=50, ceiling_height=2.5,
        windows=[Window("north", area=4.0, g_value=0.0)],
        walls=[Wall(area=40.0, u_value=0.4)],
        storage=[StorageMass("concrete", area=40.0, thickness=0.1)],
    )
    res = simulate_2node(b, weather, dt_minutes=15)
    tail = slice(len(res.t_in) // 3, None)
    assert res.t_in[tail].min() >= res.t_out.min() - 0.5
    assert res.t_in[tail].max() <= res.t_out.max() + 0.5


# ---------------------------------------------------------------------------
# Night ventilation
# ---------------------------------------------------------------------------

def test_night_ventilation_lowers_peak():
    """Night ventilation lowers the indoor-temperature maximum noticeably."""
    weather = synthetic_heatwave(hours=72, peak_dni=800.0)
    b = _demo_building()
    no_vent = simulate_2node(b, weather, dt_minutes=15,
                             ventilation=constant_strategy(0.4))
    night = simulate_2node(
        b, weather, dt_minutes=15,
        ventilation=VentilationStrategy(day_ach=0.4, night_ach=3.0, smart=True),
    )
    _, peak_no = no_vent.peak_indoor()
    _, peak_night = night.peak_indoor()
    assert peak_night < peak_no


def test_smart_ventilation_never_ventilates_when_outside_warmer():
    """'Smart': during the night window in heat -> only base air change."""
    strat = VentilationStrategy(day_ach=0.4, night_ach=3.0, smart=True,
                                night_start=21, night_end=7)
    night_time = datetime(2026, 7, 1, 2, 0, tzinfo=timezone.utc)  # 02:00 = night
    # cooler outside than inside -> ventilate
    assert strat.ach(night_time, t_out=20.0, t_in=28.0) == 3.0
    # warmer outside than inside -> do not ventilate (pointless)
    assert strat.ach(night_time, t_out=30.0, t_in=25.0) == 0.4


def test_ventilation_night_window_over_midnight():
    """The night window 21:00->07:00 is correctly detected across midnight."""
    strat = VentilationStrategy(night_start=21, night_end=7, utc_offset=0.0)
    assert strat.is_night(datetime(2026, 7, 1, 23, 0, tzinfo=timezone.utc))
    assert strat.is_night(datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc))
    assert not strat.is_night(datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc))


def test_one_node_accepts_ventilation_strategy():
    """The 1-node model also accepts a ventilation strategy."""
    weather = synthetic_heatwave(hours=48, peak_dni=700.0)
    res = simulate(_demo_building(), weather, dt_minutes=15,
                   ventilation=VentilationStrategy(day_ach=0.4, night_ach=3.0))
    assert res.ach_series is not None
    assert res.ach_series.max() == pytest.approx(3.0)
