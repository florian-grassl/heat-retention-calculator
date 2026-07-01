"""Matplotlib visualization of the simulation results."""

from __future__ import annotations

from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # works without a display; irrelevant for Streamlit
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .simulation import SimulationResult


def plot_simulation(
    result: SimulationResult,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = False,
):
    """Plot indoor vs. outdoor temperature with irradiance in the background.

    Args:
        result:   SimulationResult.
        title:    Heading (default: scenario name).
        savepath: If set, the figure is saved as PNG.
        show:     If True, plt.show() is called (interactive).

    Returns:
        The matplotlib figure.
    """
    fig, ax_t = plt.subplots(figsize=(12, 6))
    ax_irr = ax_t.twinx()

    times = result.times

    # --- Background: solar gain through the windows (filled area) ---
    ax_irr.fill_between(
        times,
        result.q_solar / 1000.0,
        color="#f9d976",
        alpha=0.45,
        label="Solar gain through windows",
        zorder=1,
    )
    ax_irr.set_ylabel("Solar heat gain [kW]", color="#c9a227")
    ax_irr.tick_params(axis="y", labelcolor="#c9a227")
    ax_irr.set_ylim(bottom=0)

    # --- Temperature curves ---
    ax_t.plot(
        times, result.t_out, color="#1f77b4", lw=2,
        label="Outdoor temperature", zorder=3,
    )
    ax_t.plot(
        times, result.t_in, color="#d62728", lw=2.4,
        label="Indoor temperature (air)", zorder=4,
    )
    if result.t_mass is not None:
        ax_t.plot(
            times, result.t_mass, color="#8c564b", lw=1.8, ls=(0, (4, 2)),
            label="Storage mass", zorder=3.5,
        )

    # --- Mark the maxima ---
    t_in_time, t_in_val = result.peak_indoor()
    t_out_time, t_out_val = result.peak_outdoor()
    ax_t.scatter([t_out_time], [t_out_val], color="#1f77b4", zorder=5, s=40)
    ax_t.scatter([t_in_time], [t_in_val], color="#d62728", zorder=5, s=40)
    ax_t.annotate(
        f"indoor max {t_in_val:.1f} °C",
        (t_in_time, t_in_val),
        textcoords="offset points", xytext=(8, 8), color="#d62728", fontsize=9,
    )
    ax_t.annotate(
        f"outdoor max {t_out_val:.1f} °C",
        (t_out_time, t_out_val),
        textcoords="offset points", xytext=(8, -14), color="#1f77b4", fontsize=9,
    )

    # Indicate the phase shift with a double arrow.
    shift = result.phase_shift_hours()
    if abs(shift) > 0.1:
        y_arrow = min(t_in_val, t_out_val) - 1.0
        ax_t.annotate(
            "",
            xy=(t_in_time, y_arrow), xytext=(t_out_time, y_arrow),
            arrowprops=dict(arrowstyle="<->", color="gray"),
        )
        mid = t_out_time + (t_in_time - t_out_time) / 2
        ax_t.text(
            mid, y_arrow + 0.2, f"Δt = {shift:+.1f} h",
            ha="center", color="gray", fontsize=9,
        )

    ax_t.set_xlabel("Time (UTC)")
    ax_t.set_ylabel("Temperature [°C]")
    ax_t.grid(True, alpha=0.3)
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax_t.xaxis.set_major_locator(mdates.HourLocator(interval=6))

    b = result.building
    subtitle = (
        f"tau = {b.time_constant_hours():.1f} h   ·   "
        f"C = {b.thermal_capacitance() / 1000:.0f} kJ/K   ·   "
        f"UA = {b.total_ua():.1f} W/K"
    )
    fig.suptitle(title or f"Heat load: {b.name}", fontsize=14, y=0.98)
    ax_t.set_title(subtitle, fontsize=10, color="dimgray")

    # shared legend
    lines1, labels1 = ax_t.get_legend_handles_labels()
    lines2, labels2 = ax_irr.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_scenarios(
    results: List[SimulationResult],
    labels: Optional[List[str]] = None,
    savepath: Optional[str] = None,
    show: bool = False,
):
    """Compare several scenarios (e.g. with/without shutters) on one axis."""
    fig, ax = plt.subplots(figsize=(12, 6))
    if results:
        ax.plot(
            results[0].times, results[0].t_out,
            color="#1f77b4", lw=1.6, ls="--", label="Outdoor temperature",
        )
    cmap = plt.get_cmap("autumn")
    for i, res in enumerate(results):
        lbl = labels[i] if labels else res.building.name
        color = cmap(0.15 + 0.6 * i / max(1, len(results) - 1))
        ax.plot(res.times, res.t_in, lw=2.2, color=color, label=lbl)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Indoor temperature [°C]")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Scenario comparison of the indoor temperature", fontsize=14)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig
