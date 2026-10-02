"""Feature schema, input validation, and presets for FactoryGuard AI.

This module is the single source of truth for the model input schema:
feature names, column order, valid ranges, and the preset definitions
shared by ``scripts/train.py`` and ``app.py``.

No sklearn, no streamlit — pure data handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import pandas as pd

PRODUCT_TYPES: tuple[str, str, str] = ("L", "M", "H")
NUMERIC_FEATURES: tuple[str, ...] = (
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
)
FEATURE_COLUMNS: tuple[str, ...] = (
    *NUMERIC_FEATURES,
    "product_L",
    "product_M",
    "product_H",
)
# 8 model input columns, order fixed forever. Saved into the bundle and
# asserted at load and at predict time.

# Validation ranges (hard limits accepted by validate()).
TEMP_RANGE: tuple[float, float] = (280.0, 330.0)      # K
SPEED_RANGE: tuple[float, float] = (0.0, 5000.0)      # rpm

# UI widget ranges — clamped to the training support of the benchmark.
UI_TEMP_RANGE: tuple[float, float] = (280.0, 330.0)   # K
UI_SPEED_RANGE: tuple[float, float] = (500.0, 3000.0)  # rpm
UI_TORQUE_RANGE: tuple[float, float] = (0.0, 80.0)    # Nm
UI_WEAR_RANGE: tuple[float, float] = (0.0, 250.0)     # min

_FIELD_LABELS: dict[str, str] = {
    "product_type": "Product type",
    "air_temperature": "Air temperature",
    "process_temperature": "Process temperature",
    "rotational_speed": "Rotational speed",
    "torque": "Torque",
    "tool_wear": "Tool wear",
}


@dataclass(frozen=True)
class MachineInputs:
    """Validated operator inputs for one machine assessment."""

    product_type: str            # "L" | "M" | "H"
    air_temperature: float       # K
    process_temperature: float   # K
    rotational_speed: float      # rpm
    torque: float                # Nm
    tool_wear: float             # min


PRESETS: dict[str, MachineInputs] = {
    # Normal machine: nominal temperatures, mid speed, low torque, little wear.
    "LOW": MachineInputs(
        product_type="M",
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1500.0,
        torque=37.0,
        tool_wear=30.0,
    ),
    # Elevated tool wear and torque (nominal speed, normal temperatures).
    "WATCH": MachineInputs(
        product_type="M",
        air_temperature=302.0,
        process_temperature=313.0,
        rotational_speed=1500.0,
        torque=55.0,
        tool_wear=210.0,
    ),
    # Clearly abnormal: heavy wear, high torque, large speed deviation, hot end.
    "HIGH": MachineInputs(
        product_type="M",
        air_temperature=305.0,
        process_temperature=317.0,
        rotational_speed=1000.0,
        torque=60.0,
        tool_wear=210.0,
    ),
}


def _as_float(value: object) -> float | None:
    """Coerce to a finite float; return None when impossible."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not isfinite(number):
        return None
    return number


def validate(raw: Mapping[str, object]) -> tuple[MachineInputs | None, list[str]]:
    """Reject non-finite/out-of-range values.

    Checks: all six fields present; product_type in PRODUCT_TYPES;
    temps in [280, 330] K; speed in [0, 5000]; torque >= 0; wear >= 0;
    all numeric values finite.

    Returns ``(inputs, [])`` on success or ``(None, errors)`` on failure.
    """
    errors: list[str] = []

    # Presence.
    for key in (
        "product_type",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ):
        if key not in raw or raw[key] is None or raw[key] == "":
            errors.append(f"Missing value: {_FIELD_LABELS[key]}.")
    if errors:
        return None, errors

    product_type = raw["product_type"]
    if not isinstance(product_type, str) or product_type not in PRODUCT_TYPES:
        errors.append("Product type must be one of L, M, H.")

    numeric_ranges: dict[str, tuple[float, float] | None] = {
        "air_temperature": TEMP_RANGE,
        "process_temperature": TEMP_RANGE,
        "rotational_speed": SPEED_RANGE,
        "torque": (0.0, None),          # >= 0, no upper limit
        "tool_wear": (0.0, None),       # >= 0, no upper limit
    }

    parsed: dict[str, float] = {}
    for key, bounds in numeric_ranges.items():
        number = _as_float(raw[key])
        if number is None:
            errors.append(f"{_FIELD_LABELS[key]} must be a finite number.")
            continue
        if bounds is not None:
            low, high = bounds
            if number < low or (high is not None and number > high):
                if high is None:
                    errors.append(f"{_FIELD_LABELS[key]} must be >= {low:g}.")
                else:
                    unit = " K" if key.endswith("temperature") else (
                        " rpm" if key == "rotational_speed" else ""
                    )
                    errors.append(
                        f"{_FIELD_LABELS[key]} must be between {low:g} and "
                        f"{high:g}{unit}."
                    )
                continue
        parsed[key] = number

    if errors:
        return None, errors

    inputs = MachineInputs(
        product_type=product_type,  # type: ignore[arg-type]
        air_temperature=parsed["air_temperature"],
        process_temperature=parsed["process_temperature"],
        rotational_speed=parsed["rotational_speed"],
        torque=parsed["torque"],
        tool_wear=parsed["tool_wear"],
    )
    return inputs, []


def transform(inputs: MachineInputs) -> pd.DataFrame:
    """One row -> DataFrame with exactly FEATURE_COLUMNS (order asserted).

    One-hot encodes product type. No scaling (Random Forest is
    scale-invariant).
    """
    row = {
        "air_temperature": inputs.air_temperature,
        "process_temperature": inputs.process_temperature,
        "rotational_speed": inputs.rotational_speed,
        "torque": inputs.torque,
        "tool_wear": inputs.tool_wear,
        "product_L": 1.0 if inputs.product_type == "L" else 0.0,
        "product_M": 1.0 if inputs.product_type == "M" else 0.0,
        "product_H": 1.0 if inputs.product_type == "H" else 0.0,
    }
    frame = pd.DataFrame([row], columns=list(FEATURE_COLUMNS))
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    return frame


def apply_preset(name: str) -> MachineInputs:
    """Return the preset inputs for ``LOW`` / ``WATCH`` / ``HIGH``."""
    key = name.upper()
    if key not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; expected one of {sorted(PRESETS)}.")
    return PRESETS[key]
