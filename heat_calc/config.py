"""Loading the building/scenario configuration from a YAML file."""

from __future__ import annotations

from typing import Any, Dict

from .building import Building, StorageMass, Wall, Window


def building_from_dict(data: Dict[str, Any]) -> Building:
    """Create a ``Building`` from a dictionary (e.g. from YAML)."""
    windows = [Window(**w) for w in data.get("windows", [])]
    walls = [Wall(**w) for w in data.get("walls", [])]
    storage = [StorageMass(**s) for s in data.get("storage", [])]
    return Building(
        name=data.get("name", "Apartment"),
        floor_area=data.get("floor_area", 60.0),
        ceiling_height=data.get("ceiling_height", 2.7),
        windows=windows,
        walls=walls,
        storage=storage,
        ventilation_ach=data.get("ventilation_ach", 0.5),
    )


def load_config(path: str) -> Dict[str, Any]:
    """Read a YAML configuration file and return it as a dictionary."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def building_from_yaml(path: str) -> Building:
    """Convenient one-liner: YAML file -> ``Building``."""
    return building_from_dict(load_config(path))
