"""Interactive Streamlit dashboard for the heat-retention-calculator.

Start with:
    streamlit run streamlit_app.py

You can change window areas, orientation, shading and floor material live and
immediately see how the maximum temperature, phase shift and time constant
change.
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from heat_calc.building import Building, StorageMass, Wall, Window
from heat_calc.materials import MATERIALS
from heat_calc.simulation import simulate, simulate_2node
from heat_calc.ventilation import VentilationStrategy, constant_strategy
from heat_calc.visualization import plot_simulation
from heat_calc.weather import load_weather, synthetic_heatwave

st.set_page_config(page_title="Heat-retention-calculator", layout="wide")
st.title("🌡️ Heat-retention-calculator")
st.caption(
    "Why is the apartment still hot in the evening even though it is already "
    "cooling down outside? A thermal RC model makes it visible."
)

with st.sidebar:
    st.header("Location & period")
    lat = st.number_input("Latitude", value=48.14, format="%.4f")
    lon = st.number_input("Longitude", value=11.58, format="%.4f")
    start = st.date_input("Start date", value=datetime(2026, 7, 1))
    days = st.slider("Duration (days)", 1, 3, 3)
    synthetic = st.checkbox("Synthetic heatwave (offline)", value=True)

    st.header("Apartment")
    floor_area = st.slider("Floor area [m²]", 20, 150, 65)
    ceiling = st.slider("Ceiling height [m]", 2.2, 3.5, 2.5, 0.1)
    ach = st.slider("Air change rate [1/h]", 0.0, 3.0, 0.4, 0.1)

    st.header("Windows")
    orient = st.selectbox("Main orientation", ["south", "west", "east", "north"])
    win_area = st.slider("Total window area [m²]", 1.0, 25.0, 12.0, 0.5)
    g_value = st.slider("g-value of the glazing", 0.2, 0.8, 0.6, 0.05)
    shading = st.slider("Shading (1=open, 0.2=shutter)", 0.1, 1.0, 1.0, 0.05)

    st.header("Floor / storage mass")
    floor_material = st.selectbox(
        "Floor material",
        list(MATERIALS.keys()),
        format_func=lambda k: MATERIALS[k].name,
        index=list(MATERIALS.keys()).index("wood"),
    )
    floor_thickness = st.slider("Effective thickness [cm]", 1, 20, 6) / 100.0
    furniture_kg = st.slider(
        "Furniture / contents [kg per m² floor]", 0, 60, 15,
        help="Rough wood-equivalent thermal mass of furniture, books, "
        "cupboards etc. 0 = empty room. More 'stuff' buffers the peak.",
    )

    st.header("Model & ventilation")
    model = st.radio(
        "Calculation model", ["twonode", "onenode"],
        format_func=lambda m: "2-node (air + mass)" if m == "twonode"
        else "1-node (simplified)",
    )
    night_vent = st.checkbox("Night ventilation (windows open at night)", value=False)
    night_ach = st.slider("Night air change rate [1/h]", 0.5, 6.0, 2.5, 0.5)
    utc_offset = st.slider("Local-time offset to UTC [h]", -12, 14, 2)

    st.header("Comfort")
    overheat_threshold = st.slider("Overheating threshold [°C]", 24, 32, 26)


@st.cache_data(show_spinner=False)
def get_weather(lat, lon, start_str, days, synthetic):
    if synthetic:
        return synthetic_heatwave(
            lat, lon,
            start=datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc),
            hours=days * 24,
        ), "synthetic"
    from datetime import timedelta
    s = datetime.fromisoformat(start_str)
    e = s + timedelta(days=days - 1)
    return load_weather(lat, lon, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")), "open-meteo"


def build() -> Building:
    storage = [
        StorageMass(floor_material, area=floor_area, thickness=floor_thickness),
        StorageMass("plaster", area=floor_area * 2, thickness=0.015),
    ]
    # Furniture / contents as a wood-equivalent mass spread over the floor.
    # thickness = (kg per m²) / wood density (500 kg/m³) -> matches m·c_p.
    if furniture_kg > 0:
        storage.append(
            StorageMass("wood", area=floor_area, thickness=furniture_kg / 500.0)
        )
    return Building(
        name="Interactive",
        floor_area=floor_area,
        ceiling_height=ceiling,
        ventilation_ach=ach,
        windows=[Window(orient, area=win_area, g_value=g_value, shading=shading)],
        walls=[Wall(area=floor_area * 0.9, u_value=0.28),
               Wall(area=floor_area, u_value=0.20)],
        storage=storage,
    )


weather, source = get_weather(lat, lon, str(start), days, synthetic)
building = build()

if night_vent:
    strategy = VentilationStrategy(
        day_ach=ach, night_ach=night_ach, smart=True, utc_offset=utc_offset,
    )
else:
    strategy = constant_strategy(ach)

if model == "twonode":
    result = simulate_2node(building, weather, dt_minutes=15, ventilation=strategy)
else:
    result = simulate(building, weather, dt_minutes=15, ventilation=strategy)

t_in_time, t_in_val = result.peak_indoor()
t_out_time, t_out_val = result.peak_outdoor()
shares = result.energy_shares()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Max. indoor temperature", f"{t_in_val:.1f} °C")
c2.metric("Phase shift", f"{result.phase_shift_hours():+.1f} h",
          help="How much later than outside the apartment reaches its maximum.")
tau = result.tau_slow_h if result.tau_slow_h else building.time_constant_hours()
c3.metric("Time constant tau", f"{tau:.1f} h",
          help="How long stored heat keeps reverberating (slow mass).")
c4.metric(f"Overheating > {overheat_threshold} °C",
          f"{result.overheating_hours(float(overheat_threshold)):.0f} h")

fig = plot_simulation(result)
st.pyplot(fig)

with st.expander("Physics & key figures"):
    st.json(building.summary())
    st.markdown(result.report().replace("\n", "  \n"))
    st.caption(f"Weather source: {source}")
