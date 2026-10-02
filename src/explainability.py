"""Model-contribution explanations for FactoryGuard AI.

Honesty framing (hard rule): every chart title, list header, and sentence
uses "Model contribution" wording. NO causal language anywhere. Local
contributions use normalized input-deviation scoring; global context uses
permutation importance computed at train time.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import numpy as np  # noqa: E402

from src.decision import THRESHOLD_COST_AWARE, THRESHOLD_DEFAULT, cost_curve  # noqa: E402
from src.model import ModelBundle  # noqa: E402
from src.preprocessing import (  # noqa: E402
    NUMERIC_FEATURES,
    PRODUCT_TYPES,
    MachineInputs,
    transform,
)

FEATURE_LABELS: dict[str, str] = {
    "tool_wear": "Tool Wear",
    "torque": "Torque",
    "rotational_speed": "Rotational Speed",
    "air_temperature": "Air Temperature",
    "process_temperature": "Process Temperature",
    "product_L": "Product Type L",
    "product_M": "Product Type M",
    "product_H": "Product H",
}

CANONICAL_EXPLANATION: str = (
    "Higher tool wear, torque and abnormal rotational speed "
    "contributed strongly to the model's risk assessment."
)


@dataclass(frozen=True)
class Contribution:
    """One feature's share of the model's risk score."""

    feature_key: str     # e.g. "tool_wear"
    label: str           # display label, e.g. "Tool Wear"
    score: float         # normalized share, 0..1, sums to 1.0
    direction: int       # +1 raised risk, -1 lowered, 0 neutral
    delta_prob: float    # signed change in failure probability from perturbation


def _predict(bundle: ModelBundle, frame) -> float:
    return float(bundle.model.predict_proba(frame)[0, 1])


def local_contributions(
    inputs: MachineInputs, bundle: ModelBundle, top_n: int = 6
) -> tuple[Contribution, ...]:
    """Normalized input-deviation scoring for one prediction.

    Numeric features: perturb by +1 training std, take delta-p for the
    upward perturbation. Product type: one-at-a-time flip to each
    alternative, take max |delta-p|. Sensitivities are normalized to sum
    to 1.0, ranked by score descending, truncated to ``top_n``.
    """
    base = transform(inputs)
    p0 = _predict(bundle, base)

    sensitivities: dict[str, float] = {}
    deltas: dict[str, float] = {}

    for col in NUMERIC_FEATURES:
        perturbed = base.copy()
        perturbed[col] = perturbed[col] + bundle.perturbation_scales[col]
        delta = _predict(bundle, perturbed) - p0
        sensitivities[col] = abs(delta)
        deltas[col] = delta

    # Product type: flip one at a time, keep the strongest effect.
    active = inputs.product_type
    best_delta, best_abs = 0.0, 0.0
    for alternative in PRODUCT_TYPES:
        if alternative == active:
            continue
        perturbed = base.copy()
        perturbed[f"product_{active}"] = 0.0
        perturbed[f"product_{alternative}"] = 1.0
        delta = _predict(bundle, perturbed) - p0
        if abs(delta) > best_abs:
            best_abs, best_delta = abs(delta), delta
    product_key = f"product_{active}"
    sensitivities[product_key] = best_abs
    deltas[product_key] = best_delta

    total = sum(sensitivities.values())
    n = len(sensitivities)
    contributions: list[Contribution] = []
    for key, sens in sensitivities.items():
        score = (sens / total) if total > 0 else (1.0 / n)
        delta = deltas[key]
        direction = 1 if delta > 0 else (-1 if delta < 0 else 0)
        contributions.append(
            Contribution(
                feature_key=key,
                label=FEATURE_LABELS[key],
                score=float(score),
                direction=direction,
                delta_prob=float(delta),
            )
        )
    contributions.sort(key=lambda c: c.score, reverse=True)
    return tuple(contributions[:top_n])


def global_importance_list(bundle: ModelBundle) -> tuple[Contribution, ...]:
    """Ranked global model contribution from held-out permutation importance."""
    items = sorted(bundle.global_importance.items(), key=lambda kv: -kv[1])
    clipped = {key: max(value, 0.0) for key, value in items}
    total = sum(clipped.values())
    out: list[Contribution] = []
    for key, _raw in items:
        share = (clipped[key] / total) if total > 0 else 0.0
        out.append(
            Contribution(
                feature_key=key,
                label=FEATURE_LABELS[key],
                score=float(share),
                direction=0,
                delta_prob=0.0,
            )
        )
    return tuple(out)


def contribution_figure(contribs: tuple[Contribution, ...]) -> Figure:
    """Horizontal bar chart of local model contributions (top item on top)."""
    ordered = list(contribs)[::-1]
    labels = [c.label for c in ordered]
    scores = [c.score for c in ordered]
    palette = {1: "#E5484D", -1: "#3B9EFF", 0: "#9AA5B1"}
    colors = [palette.get(c.direction, "#9AA5B1") for c in ordered]

    fig, ax = plt.subplots(figsize=(8, 0.62 * max(len(ordered), 4) + 1.2))
    bars = ax.barh(labels, scores, color=colors)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.0%}",
            va="center",
            ha="left",
            fontsize=9,
            color="#334155",
        )
    ax.set_xlim(0, max(max(scores), 1e-6) * 1.2 + 0.02)
    ax.set_xlabel("Normalized contribution score")
    ax.set_title("Model contribution to predicted failure risk (local)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def threshold_cost_figure(y_true, y_prob) -> Figure:
    """Cost curve with vertical lines at the cost-aware and default thresholds."""
    thresholds, costs = cost_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(thresholds, costs, color="#3B9EFF", linewidth=2, label="Expected cost")
    ymax = float(np.max(costs))
    ax.axvline(
        THRESHOLD_COST_AWARE, color="#E5484D", linestyle="--", linewidth=1.6,
        label=f"Cost-aware threshold ({THRESHOLD_COST_AWARE})",
    )
    ax.axvline(
        THRESHOLD_DEFAULT, color="#6B7280", linestyle="--", linewidth=1.6,
        label=f"Default threshold ({THRESHOLD_DEFAULT})",
    )
    ax.annotate(
        "Cost-aware threshold (research)",
        xy=(THRESHOLD_COST_AWARE, ymax),
        xytext=(THRESHOLD_COST_AWARE + 0.04, ymax * 0.92),
        fontsize=9, color="#E5484D",
    )
    ax.annotate(
        "Default threshold",
        xy=(THRESHOLD_DEFAULT, ymax),
        xytext=(THRESHOLD_DEFAULT - 0.24, ymax * 0.75),
        fontsize=9, color="#6B7280",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost (mean per sample)")
    ax.set_title("Expected misclassification cost vs threshold (research cost matrix)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper center")
    fig.tight_layout()
    return fig


def format_signals(contribs: tuple[Contribution, ...], k: int = 3) -> str:
    """e.g. 'Tool Wear ↑, Torque ↑, Rotational Speed ↑' — arrows from direction.

    Always returns arrows: the only consumer is the browser UI (Streamlit
    renders UTF-8 correctly); console printing is not part of the app.
    """
    arrows = {1: "\u2191", -1: "\u2193", 0: "\u2192"}
    return ", ".join(
        f"{c.label} {arrows.get(c.direction, '\u2192')}" for c in contribs[:k]
    )
