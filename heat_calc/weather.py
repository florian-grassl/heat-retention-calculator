"""Weather data: outdoor temperature and solar radiation.

The primary source is the free Open-Meteo API (no API key required). If the
network is unavailable or you want reproducible tests, ``synthetic_heatwave``
generates a realistic artificial heatwave daily cycle.

This module also contains a compact calculation of the solar position
(elevation + azimuth) so that the radiation can be projected onto vertical,
differently oriented window surfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@dataclass
class WeatherSeries:
    """Hourly weather time series.

    Attributes:
        times:    List of timezone-aware datetimes (UTC).
        t_out:    Outdoor temperature in °C (np.ndarray).
        dni:      Direct normal irradiance in W/m² (on a sun-facing surface).
        dhi:      Diffuse horizontal irradiance in W/m².
        ghi:      Global horizontal irradiance in W/m².
        latitude, longitude: Location (for the solar-position calculation).
    """

    times: List[datetime]
    t_out: np.ndarray
    dni: np.ndarray
    dhi: np.ndarray
    ghi: np.ndarray
    latitude: float
    longitude: float

    def __len__(self) -> int:
        return len(self.times)


# ---------------------------------------------------------------------------
# Solar position
# ---------------------------------------------------------------------------

def solar_position(lat: float, lon: float, when: datetime) -> tuple:
    """Solar elevation and azimuth for a location and instant (UTC).

    Simplified NOAA algorithm. Returned in degrees:
        elevation -- height above the horizon (negative = sun below horizon)
        azimuth   -- direction clockwise from north (0=N, 90=E, 180=S, 270=W)

    For heat-load purposes this accuracy is more than sufficient.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    when = when.astimezone(timezone.utc)

    # Fractional year (gamma) in radians.
    day_of_year = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour - 12) / 24.0)

    # Solar declination (radians).
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    # Equation of time (minutes).
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    # True solar time -> hour angle (computed in UTC, hence only longitude).
    time_offset = eqtime + 4.0 * lon      # minutes
    true_solar_time = hour * 60.0 + time_offset
    hour_angle = math.radians(true_solar_time / 4.0 - 180.0)

    lat_r = math.radians(lat)
    cos_zenith = (
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elevation = 90.0 - math.degrees(zenith)

    # Azimuth.
    sin_elev = math.cos(zenith)
    denom = math.cos(math.asin(sin_elev)) * math.cos(lat_r)
    if abs(denom) < 1e-6:
        azimuth = 180.0
    else:
        cos_az = (math.sin(decl) - sin_elev * math.sin(lat_r)) / denom
        cos_az = max(-1.0, min(1.0, cos_az))
        azimuth = math.degrees(math.acos(cos_az))
        if hour_angle > 0:          # afternoon -> west
            azimuth = 360.0 - azimuth
    return elevation, azimuth


def irradiance_on_surface(
    surface_azimuth: float,
    dni: float,
    dhi: float,
    ghi: float,
    sun_elevation: float,
    sun_azimuth: float,
    albedo: float = 0.2,
) -> float:
    """Irradiance on a vertical facade (W/m²).

    It is made up of three components:

      1. Direct:    DNI · cos(angle of incidence)  (only when the sun hits the face)
      2. Diffuse:   DHI · (1+cos 90°)/2 = DHI · 0.5 (half the sky is visible)
      3. Reflected: GHI · albedo · 0.5             (reflected off the ground)

    For a vertical surface:
        cos(incidence) = cos(sun elevation) · cos(sun azimuth − facade azimuth)
    """
    if sun_elevation <= 0:
        direct = 0.0
    else:
        cos_incidence = math.cos(math.radians(sun_elevation)) * math.cos(
            math.radians(sun_azimuth - surface_azimuth)
        )
        direct = dni * max(0.0, cos_incidence)
    diffuse = dhi * 0.5
    reflected = ghi * albedo * 0.5
    return direct + diffuse + reflected


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------

def fetch_open_meteo(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    use_archive: bool = False,
    timeout: float = 20.0,
) -> WeatherSeries:
    """Load hourly weather data from Open-Meteo.

    Args:
        latitude, longitude: Location in decimal degrees.
        start_date, end_date: 'YYYY-MM-DD' (inclusive).
        use_archive: True -> historical archive (for periods in the past);
                     False -> forecast endpoint.

    Raises:
        RuntimeError on network/API errors (the caller can fall back to the
        synthetic generator).
    """
    import requests  # imported locally so the rest works without requests

    url = OPEN_METEO_ARCHIVE_URL if use_archive else OPEN_METEO_URL
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(
            [
                "temperature_2m",
                "direct_normal_irradiance",
                "diffuse_radiation",
                "shortwave_radiation",
            ]
        ),
        "timezone": "GMT",
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - intentionally broad, for fallback
        raise RuntimeError(f"Open-Meteo request failed: {exc}") from exc

    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("Open-Meteo returned no hourly data.")

    times = [
        datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        for t in hourly["time"]
    ]

    def arr(key: str) -> np.ndarray:
        vals = hourly.get(key)
        if vals is None:
            return np.zeros(len(times))
        return np.array([v if v is not None else 0.0 for v in vals], dtype=float)

    return WeatherSeries(
        times=times,
        t_out=arr("temperature_2m"),
        dni=arr("direct_normal_irradiance"),
        dhi=arr("diffuse_radiation"),
        ghi=arr("shortwave_radiation"),
        latitude=latitude,
        longitude=longitude,
    )


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def synthetic_heatwave(
    latitude: float = 48.14,
    longitude: float = 11.58,
    start: Optional[datetime] = None,
    hours: int = 72,
    t_min: float = 20.0,
    t_max: float = 36.0,
    peak_dni: float = 850.0,
) -> WeatherSeries:
    """Artificial heatwave with a realistic daily cycle.

    Useful for tests and for offline operation. The temperature follows a sine
    with its maximum around 16:00, the radiation follows the computed solar
    position (clear sky).
    """
    if start is None:
        start = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    times = [start + timedelta(hours=i) for i in range(hours)]
    t_out = np.zeros(hours)
    dni = np.zeros(hours)
    dhi = np.zeros(hours)
    ghi = np.zeros(hours)

    amp = (t_max - t_min) / 2.0
    mean = (t_max + t_min) / 2.0
    for i, ts in enumerate(times):
        local_hour = ts.hour + ts.minute / 60.0
        # Temperature daily cycle, maximum around 16:00 (phase -10h -> cos).
        t_out[i] = mean - amp * math.cos(2 * math.pi * (local_hour - 4) / 24.0)

        elev, _ = solar_position(latitude, longitude, ts)
        if elev > 0:
            clear = math.sin(math.radians(elev))
            dni[i] = peak_dni * clear
            dhi[i] = 0.15 * peak_dni * clear
            ghi[i] = dni[i] * clear + dhi[i]
    return WeatherSeries(
        times=times,
        t_out=t_out,
        dni=dni,
        dhi=dhi,
        ghi=ghi,
        latitude=latitude,
        longitude=longitude,
    )


def load_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    use_archive: bool = False,
    allow_synthetic_fallback: bool = True,
) -> tuple:
    """Load weather, optionally falling back to synthetic data on error.

    Returns:
        (WeatherSeries, source) with source in {'open-meteo', 'synthetic'}.
    """
    try:
        return fetch_open_meteo(
            latitude, longitude, start_date, end_date, use_archive=use_archive
        ), "open-meteo"
    except RuntimeError:
        if not allow_synthetic_fallback:
            raise
        start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        hours = int((end - start).total_seconds() // 3600) + 24
        return synthetic_heatwave(
            latitude, longitude, start=start, hours=max(24, hours)
        ), "synthetic"
