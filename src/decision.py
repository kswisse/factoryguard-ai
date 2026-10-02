"""Risk classification, actions, and cost-aware threshold policy.

Pure policy module: no sklearn, no streamlit, no file I/O. Thresholds live
only here; ``scripts/train.py`` imports ``THRESHOLD_COST_AWARE`` from here
so the UI and the training gate can never drift apart.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

THRESHOLD_WATCH: float = 0.05
THRESHOLD_COST_AWARE: float = 0.0895
THRESHOLD_DEFAULT: float = 0.50

RESEARCH_COSTS: dict[str, int] = {"FN": 100, "FP": 5, "TP": 1, "TN": 0}

POLICY_DISCLAIMER: str = "Demo policy thresholds \u2014 not universal industrial safety standards."
DECISION_DISCLAIMER: str = "AI output is decision support, not an autonomous maintenance command."


class RiskLevel(str, Enum):
    LOW = "LOW"
    WATCH = "WATCH"
    HIGH = "HIGH"

    @property
    def card_label(self) -> str:
        return f"{self.value} RISK"

    @property
    def color(self) -> str:
        return _RISK_COLORS[self.value]


_RISK_COLORS: dict[str, str] = {
    "LOW": "#3B9EFF",
    "WATCH": "#F5A623",
    "HIGH": "#E5484D",
}


ACTIONS: dict[RiskLevel, tuple[str, str]] = {
    RiskLevel.LOW: (
        "Continue Monitoring",
        "Current operating conditions indicate relatively low predicted failure "
        "risk. Continue routine monitoring.",
    ),
    RiskLevel.WATCH: (
        "Inspect & Increase Monitoring",
        "Risk is elevated. Inspect tool wear, torque and rotational-speed trends "
        "and increase monitoring frequency.",
    ),
    RiskLevel.HIGH: (
        "Schedule Maintenance",
        "Predicted failure risk is elevated. Inspect the machine and consider "
        "maintenance before the next production cycle.",
    ),
}


def classify(probability: float) -> RiskLevel:
    """p < 0.05 -> LOW; 0.05 <= p < 0.0895 -> WATCH; p >= 0.0895 -> HIGH."""
    if probability < THRESHOLD_WATCH:
        return RiskLevel.LOW
    if probability < THRESHOLD_COST_AWARE:
        return RiskLevel.WATCH
    return RiskLevel.HIGH


def action_for(level: RiskLevel) -> tuple[str, str]:
    """Return the ``(title, detail)`` maintenance action for a risk level."""
    return ACTIONS[level]


def expected_cost(y_true, y_prob, threshold: float,
                  costs: dict[str, int] | None = None) -> float:
    """Mean per-sample misclassification cost; pred = (y_prob >= threshold)."""
    if costs is None:
        costs = RESEARCH_COSTS
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    predicted = (y_prob >= threshold).astype(int)
    actual = y_true.astype(int)
    fn = np.logical_and(actual == 1, predicted == 0).sum()
    fp = np.logical_and(actual == 0, predicted == 1).sum()
    tp = np.logical_and(actual == 1, predicted == 1).sum()
    tn = np.logical_and(actual == 0, predicted == 0).sum()
    total = costs["FN"] * fn + costs["FP"] * fp + costs["TP"] * tp + costs["TN"] * tn
    return float(total) / max(len(y_true), 1)


def cost_curve(y_true, y_prob, costs: dict[str, int] | None = None,
               grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(thresholds, expected_cost)`` over a threshold grid.

    Powers the "Why not use 50%?" chart.
    """
    if costs is None:
        costs = RESEARCH_COSTS
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    costs_out = np.array([expected_cost(y_true, y_prob, t, costs) for t in grid])
    return np.asarray(grid, dtype=float), costs_out
