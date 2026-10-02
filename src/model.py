"""Model training, persistence, and inference for FactoryGuard AI.

Builds a synthetic AI4I-2020-style benchmark, trains a RandomForest
classifier, and persists a :class:`ModelBundle` that the Streamlit app
loads. ``RESEARCH_METRICS`` holds the ONLY metrics the UI may display;
computed test metrics are engineering diagnostics and are printed, never
rendered as claims.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.decision import THRESHOLD_COST_AWARE
from src.preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES, PRODUCT_TYPES

# The ONLY metrics the UI may display.
RESEARCH_METRICS: dict = {
    "model": "Random Forest",
    "roc_auc": 0.968,
    "pr_auc": 0.593,
    "f1": 0.642,
    "cost_aware_threshold": 0.0895,
    "cost_aware_recall": 0.964,
    "benchmark": (
        "Synthetic 10,000-row benchmark based on the documented structure and "
        "failure mechanisms of UCI AI4I 2020."
    ),
}

MODEL_PATH: Path = Path(__file__).resolve().parent.parent / "models" / "factoryguard_model.pkl"

_TARGET_FAILURE_RATE = 0.05
_TEST_SIZE = 0.2


@dataclass(frozen=True)
class ModelBundle:
    """Everything needed to predict, explain, and audit one trained model."""

    model: Any                              # fitted RandomForestClassifier
    feature_columns: tuple[str, ...]
    perturbation_scales: dict[str, float]   # training std per numeric feature
    categorical_values: dict[str, tuple[str, ...]]  # {"product_type": ("L","M","H")}
    global_importance: dict[str, float]     # permutation importance on test set, normalized
    y_true_test: np.ndarray                 # held-out labels (feeds cost curve)
    y_prob_test: np.ndarray                 # held-out P(failure) (feeds cost curve)
    test_metrics: dict[str, float]          # engineering diagnostics — LOGGED, never shown as claims
    sklearn_version: str
    trained_at: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_benchmark(n_rows: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic 10,000-row predictive-maintenance benchmark.

    Physics-inspired: product mix ~36/50/14 (L/M/H), air temperature
    ~ Normal(300, 2) K, process temperature = air + Normal(10, 0.8) K,
    rotational speed ~ Normal(1500, 400) rpm truncated > 0, torque as an
    inverse function of speed + Normal(0, 6) Nm floored at 0, tool wear ~
    Uniform(0, 240) min. Failure labels come from a logistic proxy whose
    log-odds rise with tool wear, torque, |speed - 1500| and a small
    product-type effect; an intercept search targets a ~5% failure rate,
    then labels are Bernoulli-sampled.

    This label mechanism is a synthetic proxy. It does not replicate the
    original AI4I 2020 label-generation rules (TWF/HDF/PWF/OSF/RNF). The
    dataset is a benchmark for demonstrating a workflow, not a
    reproduction of the published dataset.
    """
    rng = np.random.default_rng(random_state)

    product_type = rng.choice(PRODUCT_TYPES, size=n_rows, p=[0.36, 0.50, 0.14])

    air_temperature = rng.normal(300.0, 2.0, n_rows)
    process_temperature = air_temperature + rng.normal(10.0, 0.8, n_rows)

    rotational_speed = rng.normal(1500.0, 400.0, n_rows)
    while np.any(rotational_speed <= 0):           # truncated > 0
        rotational_speed[rotational_speed <= 0] = rng.normal(
            1500.0, 400.0, int((rotational_speed <= 0).sum())
        )

    # Inverse-of-speed base torque with Gaussian noise, floored at 0 Nm.
    torque = 40.0 + 30.0 * (1500.0 / rotational_speed - 1.0) + rng.normal(0.0, 6.0, n_rows)
    torque = np.clip(torque, 0.0, None)

    tool_wear = rng.uniform(0.0, 240.0, n_rows)

    product_effect = np.where(
        product_type == "L", -0.30,
        np.where(product_type == "M", 0.0, 0.40),
    )

    # Logistic proxy for failure log-odds.
    logit = (
        3.0 * (tool_wear / 240.0)
        + 1.5 * (torque / 50.0)
        + 1.2 * (np.abs(rotational_speed - 1500.0) / 500.0)
        + product_effect
    )

    # Interception search: choose intercept so mean(sigmoid(logit + b)) ~= 5%.
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if float(_sigmoid(logit + mid).mean()) < _TARGET_FAILURE_RATE:
            lo = mid
        else:
            hi = mid
    intercept = (lo + hi) / 2.0
    p_failure = _sigmoid(logit + intercept)
    target = rng.binomial(1, p_failure)

    df = pd.DataFrame({
        "uid": np.arange(1, n_rows + 1),
        "product_type": product_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "target": target,
    })
    return df


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Raw benchmark frame -> model matrix with exactly FEATURE_COLUMNS."""
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        out[col] = df[col].astype(float)
    for p in PRODUCT_TYPES:
        out[f"product_{p}"] = (df["product_type"] == p).astype(float)
    out = out[list(FEATURE_COLUMNS)]
    assert list(out.columns) == list(FEATURE_COLUMNS)
    return out


def train(df: pd.DataFrame, random_state: int = 42) -> ModelBundle:
    """Stratified 80/20 split -> RandomForest(300) -> ModelBundle."""
    X = design_matrix(df)
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, stratify=y, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_prob_test = model.predict_proba(X_test)[:, 1]

    preds_50 = (y_prob_test >= 0.50).astype(int)
    preds_ca = (y_prob_test >= THRESHOLD_COST_AWARE).astype(int)
    test_metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_test, y_prob_test)),
        "pr_auc": float(average_precision_score(y_test, y_prob_test)),
        "f1_at_0.50": float(f1_score(y_test, preds_50, zero_division=0)),
        "recall_at_cost_aware": float(
            ((y_test.to_numpy() == 1) & (preds_ca == 1)).sum()
            / max(int((y_test.to_numpy() == 1).sum()), 1)
        ),
        "positive_rate_test": float(y_test.mean()),
    }

    # Permutation importance on the held-out set -> normalized global share.
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    raw = np.clip(perm.importances_mean, 0.0, None)
    total = float(raw.sum())
    if total > 0:
        normalized = raw / total
    else:
        normalized = np.full(len(raw), 1.0 / len(raw))
    global_importance = {
        col: float(val) for col, val in zip(FEATURE_COLUMNS, normalized)
    }

    # Training std per numeric feature (perturbation scales for explanations).
    perturbation_scales = {
        col: float(X_train[col].std(ddof=0)) for col in NUMERIC_FEATURES
    }

    return ModelBundle(
        model=model,
        feature_columns=tuple(FEATURE_COLUMNS),
        perturbation_scales=perturbation_scales,
        categorical_values={"product_type": tuple(PRODUCT_TYPES)},
        global_importance=global_importance,
        y_true_test=y_test.to_numpy(),
        y_prob_test=np.asarray(y_prob_test),
        test_metrics=test_metrics,
        sklearn_version=sklearn.__version__,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def save_bundle(bundle: ModelBundle, path: Path = MODEL_PATH) -> Path:
    """Persist the bundle with pickle; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return path


def load_bundle(path: Path = MODEL_PATH) -> ModelBundle:
    """Load a persisted bundle; raises on missing file or schema mismatch."""
    path = Path(path)
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle.")
    if tuple(bundle.feature_columns) != tuple(FEATURE_COLUMNS):
        raise ValueError(
            "Bundle feature schema mismatch: "
            f"{bundle.feature_columns} != {FEATURE_COLUMNS}"
        )
    return bundle


def ensure_bundle(path: Path = MODEL_PATH) -> ModelBundle:
    """load_bundle(); on missing file OR unpickling/schema error -> regenerate
    + train + save. Guarantees ``streamlit run app.py`` works on a fresh clone.
    """
    path = Path(path)
    try:
        return load_bundle(path)
    except FileNotFoundError:
        print(f"[ensure_bundle] {path} missing — training a fresh bundle.")
    except Exception as exc:  # unpickling / schema / version errors
        print(f"[ensure_bundle] could not load {path} ({exc}) — retraining.")
    df = generate_synthetic_benchmark()
    bundle = train(df)
    save_bundle(bundle, path)
    return bundle


def predict_failure_probability(bundle: ModelBundle, X: pd.DataFrame) -> float:
    """Assert feature order matches the bundle, return P(failure) for row 0."""
    assert list(X.columns) == list(bundle.feature_columns), (
        f"Feature order mismatch: {list(X.columns)} != {list(bundle.feature_columns)}"
    )
    return float(bundle.model.predict_proba(X)[0, 1])
