"""Heat-retention-calculator.

A thermal RC model that simulates how an apartment absorbs, stores and
releases heat during a heatwave.

Modules:
    materials      -- Small material database (density, specific heat)
    building       -- Data model of the apartment (windows, walls, storage mass)
    weather        -- Weather data from the Open-Meteo API (+ synthetic fallback)
    ventilation    -- Ventilation strategies (night ventilation, "smart")
    simulation     -- Numerical solution of the heat-balance ODE (1-/2-node)
    visualization  -- Matplotlib plots
"""

__version__ = "0.1.0"
