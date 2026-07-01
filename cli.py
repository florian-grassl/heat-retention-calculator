#!/usr/bin/env python3
"""Command-line tool for the heat-retention-calculator.

Examples:
    # Default run with config.yaml, location Munich, 72 h from today
    python cli.py

    # Different location and period, result as CSV
    python cli.py --lat 52.52 --lon 13.40 --start 2026-07-10 --days 3 --csv out.csv

    # Scenario comparison: without vs. with shutters (25 % shading)
    python cli.py --compare-shading

    # Scenario comparison: without vs. with night ventilation
    python cli.py --compare-ventilation

    # Offline / reproducible (synthetic heatwave, no network)
    python cli.py --synthetic
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone

from heat_calc.config import building_from_yaml
from heat_calc.simulation import SimulationResult, simulate, simulate_2node
from heat_calc.ventilation import VentilationStrategy
from heat_calc.visualization import plot_scenarios, plot_simulation
from heat_calc.weather import load_weather, synthetic_heatwave


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="heat-retention-calculator: simulates the indoor "
        "temperature of an apartment during a heatwave (thermal RC model)."
    )
    p.add_argument("--config", default="config.yaml", help="Path to the building YAML")
    p.add_argument("--lat", type=float, default=48.14, help="Latitude")
    p.add_argument("--lon", type=float, default=11.58, help="Longitude")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: today)")
    p.add_argument("--days", type=int, default=3, help="Simulation length in days (1-3)")
    p.add_argument("--dt", type=float, default=15.0, help="Time step in minutes")
    p.add_argument("--method", choices=["euler", "scipy"], default="euler")
    p.add_argument("--model", choices=["onenode", "twonode"], default="twonode",
                   help="1-node (air=mass) or 2-node (separated, more accurate)")
    p.add_argument("--utc-offset", type=float, default=2.0,
                   help="Local-time offset to UTC in hours (summer CET = +2)")
    p.add_argument("--night-vent", action="store_true",
                   help="Enable night ventilation (windows open at night)")
    p.add_argument("--night-ach", type=float, default=2.5,
                   help="Air change rate during night ventilation in 1/h")
    p.add_argument("--compare-ventilation", action="store_true",
                   help="Scenario comparison without vs. with night ventilation")
    p.add_argument("--archive", action="store_true",
                   help="Use the Open-Meteo archive (period in the past)")
    p.add_argument("--synthetic", action="store_true",
                   help="No network: use a synthetic heatwave")
    p.add_argument("--compare-shading", action="store_true",
                   help="Scenario comparison without vs. with shutters")
    p.add_argument("--shading", type=float, default=0.25,
                   help="Shading factor for the shutter scenario (0..1)")
    p.add_argument("--plot", default="heat_load.png", help="Path for the PNG figure")
    p.add_argument("--csv", default=None, help="Also save the results as CSV")
    p.add_argument("--no-plot", action="store_true", help="Do not create a figure")
    return p.parse_args()


def _dates(args) -> tuple:
    if args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    else:
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    end = start + timedelta(days=max(1, args.days) - 1)
    return start, end


def get_weather(args):
    start, end = _dates(args)
    if args.synthetic:
        hours = max(24, args.days * 24)
        return synthetic_heatwave(args.lat, args.lon, start=start, hours=hours), "synthetic"
    return load_weather(
        args.lat, args.lon,
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        use_archive=args.archive,
    )


def export_csv(result: SimulationResult, path: str) -> None:
    """Export the time series as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "time_utc", "t_outdoor_C", "t_indoor_C",
            "q_solar_W", "q_envelope_W",
        ])
        for i, t in enumerate(result.times):
            w.writerow([
                t.strftime("%Y-%m-%d %H:%M"),
                f"{result.t_out[i]:.2f}",
                f"{result.t_in[i]:.2f}",
                f"{result.q_solar[i]:.1f}",
                f"{result.q_envelope[i]:.1f}",
            ])
    print(f"CSV written: {path}")


def make_ventilation(args, active: bool) -> VentilationStrategy:
    """Build a ventilation strategy matching the CLI options."""
    if active:
        return VentilationStrategy(
            day_ach=0.4, night_ach=args.night_ach,
            smart=True, utc_offset=args.utc_offset,
        )
    # no night ventilation: constant daytime air change rate
    return VentilationStrategy(
        day_ach=0.4, night_ach=0.4, smart=False, utc_offset=args.utc_offset,
    )


def run_model(building, weather, args, ventilation) -> SimulationResult:
    """Run the simulation with the chosen model (1- or 2-node)."""
    if args.model == "twonode":
        return simulate_2node(
            building, weather, dt_minutes=args.dt, ventilation=ventilation
        )
    return simulate(
        building, weather, dt_minutes=args.dt, method=args.method,
        ventilation=ventilation,
    )


def main() -> None:
    args = parse_args()
    building = building_from_yaml(args.config)
    weather, source = get_weather(args)
    print(f"Weather source: {source}  ({len(weather)} hourly values)")
    print(f"Model: {args.model}\n")

    ventilation = make_ventilation(args, active=args.night_vent)
    result = run_model(building, weather, args, ventilation)
    print(result.report())

    if args.csv:
        export_csv(result, args.csv)

    if args.compare_ventilation:
        vent_on = make_ventilation(args, active=True)
        result_vent = run_model(building, weather, args, vent_on)
        print("\n" + result_vent.report())
        _, peak_no = result.peak_indoor()
        _, peak_vent = result_vent.peak_indoor()
        print(f"\n=> Night ventilation lowers the indoor maximum by "
              f"{peak_no - peak_vent:.1f} K.")
        if not args.no_plot:
            plot_scenarios(
                [result, result_vent],
                labels=["without night ventilation", "with night ventilation"],
                savepath=args.plot,
            )
            print(f"Figure written: {args.plot}")
        return

    if args.compare_shading:
        shaded = copy.deepcopy(building)
        shaded.name = f"{building.name} + shutters"
        for win in shaded.windows:
            # Shutters mainly on south/west/east, north less relevant.
            if win.orientation.lower() in ("south", "west", "east"):
                win.shading = args.shading
        result_shaded = run_model(shaded, weather, args, ventilation)
        print("\n" + result_shaded.report())

        _, peak_open = result.peak_indoor()
        _, peak_shaded = result_shaded.peak_indoor()
        print(f"\n=> Shading lowers the indoor maximum by "
              f"{peak_open - peak_shaded:.1f} K.")

        if not args.no_plot:
            plot_scenarios(
                [result, result_shaded],
                labels=["without shading", "with shutters"],
                savepath=args.plot,
            )
            print(f"Figure written: {args.plot}")
        return

    if not args.no_plot:
        plot_simulation(result, savepath=args.plot)
        print(f"\nFigure written: {args.plot}")


if __name__ == "__main__":
    main()
