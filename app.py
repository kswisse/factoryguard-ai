"""FactoryGuard AI — AI Predictive Maintenance Dashboard (Streamlit demo).

Machine Data -> ML Prediction -> Risk Score -> Cost-aware Decision ->
Maintenance Recommendation.

Research prototype: all displayed benchmark metrics are research benchmark
results on a synthetic dataset; the UI never presents them as live
predictions for the operator's machine.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.decision import (
    ACTIONS,
    DECISION_DISCLAIMER,
    POLICY_DISCLAIMER,
    RESEARCH_COSTS,
    THRESHOLD_COST_AWARE,
    THRESHOLD_DEFAULT,
    RiskLevel,
    classify,
    expected_cost,
)
from src.explainability import (
    CANONICAL_EXPLANATION,
    Contribution,
    contribution_figure,
    format_signals,
    global_importance_list,
    local_contributions,
    threshold_cost_figure,
)
from src.model import (
    RESEARCH_METRICS,
    ModelBundle,
    ensure_bundle,
    predict_failure_probability,
)
from src.preprocessing import (
    PRESETS,
    UI_SPEED_RANGE,
    UI_TEMP_RANGE,
    UI_TORQUE_RANGE,
    UI_WEAR_RANGE,
    MachineInputs,
    transform,
    validate,
)

INTEGRITY_STATEMENT = (
    "This prototype demonstrates a research workflow using synthetic "
    "industrial telemetry. It has not been validated on a real production line."
)

WIDGET_KEYS = {
    "product_type": "product_type",
    "air_temperature": "air_temp",
    "process_temperature": "proc_temp",
    "rotational_speed": "speed",
    "torque": "torque",
    "tool_wear": "wear",
}


@dataclass(frozen=True)
class PredictionResult:
    """One prediction rendered by the result card (persists across reruns)."""

    inputs: MachineInputs
    probability: float
    risk_level: RiskLevel
    action_title: str
    action_detail: str
    contributions: tuple[Contribution, ...]
    threshold_used: float
    generated_at: str


@st.cache_resource
def get_bundle() -> ModelBundle:
    """Load the persisted bundle; retrain only if the pickle is missing/broken."""
    return ensure_bundle()


def _apply_preset(name: str) -> None:
    """Preset button callback: write the six widget keys before instantiation."""
    preset = PRESETS[name]
    st.session_state[WIDGET_KEYS["product_type"]] = preset.product_type
    st.session_state[WIDGET_KEYS["air_temperature"]] = preset.air_temperature
    st.session_state[WIDGET_KEYS["process_temperature"]] = preset.process_temperature
    st.session_state[WIDGET_KEYS["rotational_speed"]] = preset.rotational_speed
    st.session_state[WIDGET_KEYS["torque"]] = preset.torque
    st.session_state[WIDGET_KEYS["tool_wear"]] = preset.tool_wear
    st.session_state["active_preset"] = name


def _run_prediction(bundle: ModelBundle) -> None:
    """raw widgets -> validate -> transform -> predict -> classify -> store."""
    raw = {
        "product_type": st.session_state.get(WIDGET_KEYS["product_type"]),
        "air_temperature": st.session_state.get(WIDGET_KEYS["air_temperature"]),
        "process_temperature": st.session_state.get(WIDGET_KEYS["process_temperature"]),
        "rotational_speed": st.session_state.get(WIDGET_KEYS["rotational_speed"]),
        "torque": st.session_state.get(WIDGET_KEYS["torque"]),
        "tool_wear": st.session_state.get(WIDGET_KEYS["tool_wear"]),
    }
    inputs, errors = validate(raw)
    if errors or inputs is None:
        st.session_state["last_errors"] = errors
        return
    st.session_state["last_errors"] = []

    frame = transform(inputs)
    probability = predict_failure_probability(bundle, frame)
    level = classify(probability)
    title, detail = ACTIONS[level]
    contributions = local_contributions(inputs, bundle, top_n=6)
    result = PredictionResult(
        inputs=inputs,
        probability=probability,
        risk_level=level,
        action_title=title,
        action_detail=detail,
        contributions=contributions,
        threshold_used=THRESHOLD_COST_AWARE,
        generated_at=datetime.now().strftime("%H:%M:%S"),
    )
    st.session_state["last_result"] = result


def _init_defaults() -> None:
    """Reasonable defaults = LOW preset (a normal operating machine)."""
    if "preset_initialized" not in st.session_state:
        _apply_preset("LOW")
        st.session_state["last_result"] = None
        st.session_state["last_errors"] = []
        st.session_state["preset_initialized"] = True


def _render_result_card(result: PredictionResult) -> None:
    """Right column: MACHINE RISK -> % -> badge -> action -> signals."""
    color = result.risk_level.color

    st.markdown("#### MACHINE RISK")
    st.markdown(
        f"<div style='font-size:3rem;font-weight:700;color:{color};"
        f"line-height:1.05;margin-bottom:0.15rem;'>{result.probability:.1%}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Predicted failure probability — model output for the current inputs "
        "(not a measurement from the machine)."
    )
    st.markdown(
        f"<span style='display:inline-block;background:{color};color:#FFFFFF;"
        f"font-weight:700;font-size:0.95rem;padding:0.2rem 0.7rem;"
        f"border-radius:0.3rem;letter-spacing:0.05em;'>"
        f"{result.risk_level.card_label}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Risk category: {result.risk_level.value} — classified by demo policy "
        f"thresholds from the research cost-aware setup (HIGH ≥ "
        f"{result.threshold_used}, WATCH ≥ 0.05) · analyzed at "
        f"{result.generated_at}"
    )
    st.caption(POLICY_DISCLAIMER)

    st.markdown("#### Recommended Action")
    with st.container(border=True):
        st.markdown(f"**{result.action_title}**")
        st.write(result.action_detail)
        st.caption(DECISION_DISCLAIMER)

    signals = format_signals(result.contributions, k=3)
    st.markdown("#### Main Signals")
    st.markdown(f"**{signals}**")
    st.caption(
        "Arrows show how the model's risk score responds when each input is "
        "increased (↑ raises the score, ↓ lowers it). Model contribution "
        "ranking for this prediction — not a causal attribution."
    )


def _render_contributions(result: PredictionResult, bundle: ModelBundle) -> None:
    """Lower section: local contribution bars + canonical explanation + global list."""
    st.subheader("Model Contributions")
    fig = contribution_figure(result.contributions)
    st.pyplot(fig, clear_figure=True)

    st.info(CANONICAL_EXPLANATION)

    st.markdown("**Global model contribution (held-out permutation importance)**")
    for item in global_importance_list(bundle):
        st.markdown(f"- **{item.label}** — {item.score:.0%}")
    st.caption(
        "Contribution scores describe the model's behavior on the synthetic "
        "benchmark; they are not causal effects on the machine."
    )


def _render_cost_explainer(bundle: ModelBundle) -> None:
    """Cost-aware threshold expander: curve + matrix + research-assumption label."""
    with st.expander("Why not use 50%? Cost-aware threshold logic", expanded=False):
        st.markdown(
            "**Default threshold `0.50`** vs **Cost-aware threshold `0.0895`**"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                st.metric("Default threshold", f"{THRESHOLD_DEFAULT:.2f}")
                st.caption("Conventional 50/50 decision cutoff.")
        with col_b:
            with st.container(border=True):
                st.metric("Cost-aware threshold (research)", f"{THRESHOLD_COST_AWARE}")
                st.caption(
                    "Research cost-aware cutoff — demo policy threshold, "
                    "not an industrial safety standard."
                )
        st.markdown(
            "Why so low? Under the research cost matrix, a **missed failure "
            "(FN = 100)** is assumed far more expensive than a **false alarm "
            "(FP = 5)**, so the cheaper decision is to flag earlier."
        )

        fig = threshold_cost_figure(bundle.y_true_test, bundle.y_prob_test)
        st.pyplot(fig, clear_figure=True)

        st.markdown(
            "| Outcome | Assumed cost | Meaning |\n"
            "|---|---:|---|\n"
            f"| Missed failure (FN) | {RESEARCH_COSTS['FN']} | "
            "Missing a failure = high assumed cost |\n"
            f"| False alarm (FP) | {RESEARCH_COSTS['FP']} | "
            "False alarm = lower assumed cost |\n"
            f"| True positive (TP) | {RESEARCH_COSTS['TP']} | Detection |\n"
            f"| True negative (TN) | {RESEARCH_COSTS['TN']} | Correct pass |"
        )
        cost_ca = expected_cost(bundle.y_true_test, bundle.y_prob_test, THRESHOLD_COST_AWARE)
        cost_def = expected_cost(bundle.y_true_test, bundle.y_prob_test, THRESHOLD_DEFAULT)
        st.markdown(
            f"Expected cost on the synthetic benchmark (research cost matrix): "
            f"**{cost_ca:.2f}** at threshold {THRESHOLD_COST_AWARE} vs "
            f"**{cost_def:.2f}** at threshold {THRESHOLD_DEFAULT}."
        )
        st.warning("Research assumption — not real factory economics.")
        st.caption(POLICY_DISCLAIMER)


def _render_research_context() -> None:
    """Collapsible research context: benchmark metrics + integrity statements."""
    with st.expander("Research context", expanded=False):
        m = RESEARCH_METRICS
        st.markdown(f"**Benchmark** — {m['benchmark']}")
        st.markdown(f"**Model** — {m['model']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")
        col2.metric("Test PR-AUC", f"{m['pr_auc']:.3f}")
        col3.metric("Test F1", f"{m['f1']:.3f}")
        col4.metric("Cost-aware threshold", f"{m['cost_aware_threshold']}")
        st.caption(f"Cost-aware recall (benchmark): {m['cost_aware_recall']:.1%}")
        st.markdown(
            "**Important** — All reported research metrics come from a synthetic "
            "computational benchmark. They are not measurements from a physical "
            "factory. They are research benchmark results, not live predictions "
            "for your machine."
        )
        st.markdown(f"**Data integrity** — {INTEGRITY_STATEMENT}")


def main() -> None:
    st.set_page_config(
        page_title="FactoryGuard AI",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_defaults()
    bundle = get_bundle()

    # ---- Header -----------------------------------------------------------
    st.title("FactoryGuard AI")
    st.caption("AI Predictive Maintenance Dashboard")
    st.info(INTEGRITY_STATEMENT)

    # ---- Input panel / result card ----------------------------------------
    col_inputs, col_result = st.columns([5, 7], gap="large")

    with col_inputs:
        st.subheader("Machine Inputs")
        preset_cols = st.columns(3)
        with preset_cols[0]:
            st.button(
                "LOW RISK preset",
                key="btn_preset_low",
                on_click=_apply_preset,
                args=("LOW",),
                use_container_width=True,
            )
        with preset_cols[1]:
            st.button(
                "WATCH preset",
                key="btn_preset_watch",
                on_click=_apply_preset,
                args=("WATCH",),
                use_container_width=True,
            )
        with preset_cols[2]:
            st.button(
                "HIGH preset",
                key="btn_preset_high",
                on_click=_apply_preset,
                args=("HIGH",),
                use_container_width=True,
            )
        active = st.session_state.get("active_preset")
        if active:
            st.caption(f"Active preset: {active} (defaults are demo examples)")

        st.selectbox(
            "Product Type",
            options=["L", "M", "H"],
            key=WIDGET_KEYS["product_type"],
            help="L / M / H product class (categorical input)",
        )
        st.number_input(
            "Air Temperature [K]",
            min_value=UI_TEMP_RANGE[0], max_value=UI_TEMP_RANGE[1],
            step=0.5, format="%.1f", key=WIDGET_KEYS["air_temperature"],
        )
        st.number_input(
            "Process Temperature [K]",
            min_value=UI_TEMP_RANGE[0], max_value=UI_TEMP_RANGE[1],
            step=0.5, format="%.1f", key=WIDGET_KEYS["process_temperature"],
        )
        st.number_input(
            "Rotational Speed [rpm]",
            min_value=UI_SPEED_RANGE[0], max_value=UI_SPEED_RANGE[1],
            step=10.0, format="%.0f", key=WIDGET_KEYS["rotational_speed"],
        )
        st.number_input(
            "Torque [Nm]",
            min_value=UI_TORQUE_RANGE[0], max_value=UI_TORQUE_RANGE[1],
            step=0.5, format="%.1f", key=WIDGET_KEYS["torque"],
        )
        st.number_input(
            "Tool Wear [min]",
            min_value=UI_WEAR_RANGE[0], max_value=UI_WEAR_RANGE[1],
            step=5.0, format="%.0f", key=WIDGET_KEYS["tool_wear"],
        )

        if st.button("Analyze", type="primary", key="btn_analyze"):
            _run_prediction(bundle)

        for error in st.session_state.get("last_errors", []):
            st.error(error)

    with col_result:
        st.subheader("Risk Result")
        result: PredictionResult | None = st.session_state.get("last_result")
        if result is None:
            st.caption(
                "Choose a preset or enter machine parameters, then press Analyze."
            )
        else:
            _render_result_card(result)

    # ---- Lower sections ----------------------------------------------------
    st.divider()
    if result is not None:
        _render_contributions(result, bundle)
    else:
        st.subheader("Model Contributions")
        st.caption("Run an analysis to see the model contribution breakdown.")

    _render_cost_explainer(bundle)
    _render_research_context()

    st.divider()
    st.caption(DECISION_DISCLAIMER)


if __name__ == "__main__":
    main()
