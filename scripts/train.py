"""Train, sanity-gate, and persist the FactoryGuard model bundle.

Run:  python scripts/train.py

Engineering metrics printed below are diagnostics for developers only.
The UI displays RESEARCH_METRICS exclusively.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import (  # noqa: E402
    RESEARCH_COSTS,
    THRESHOLD_COST_AWARE,
    THRESHOLD_DEFAULT,
    classify,
    expected_cost,
)
from src.model import (  # noqa: E402
    MODEL_PATH,
    generate_synthetic_benchmark,
    predict_failure_probability,
    save_bundle,
    train,
)
from src.preprocessing import PRESETS, transform  # noqa: E402


def main() -> int:
    print("Generating synthetic benchmark...")
    df = generate_synthetic_benchmark()
    print(f"  rows={len(df)}  target_failure_rate={df['target'].mean():.4f}")

    bundle = train(df)

    print("\nEngineering diagnostics (logged only - never shown in the UI):")
    for key, value in bundle.test_metrics.items():
        print(f"  {key:<24} {value:.4f}")

    print("\nGlobal model contribution (permutation importance, normalized):")
    for col, val in sorted(bundle.global_importance.items(), key=lambda kv: -kv[1]):
        print(f"  {col:<22} {val:.4f}")

    cost_ca = expected_cost(bundle.y_true_test, bundle.y_prob_test, THRESHOLD_COST_AWARE)
    cost_def = expected_cost(bundle.y_true_test, bundle.y_prob_test, THRESHOLD_DEFAULT)
    print(f"\nExpected cost @ {THRESHOLD_COST_AWARE}: {cost_ca:.4f}")
    print(f"Expected cost @ {THRESHOLD_DEFAULT}: {cost_def:.4f}")
    print(f"Research costs: {RESEARCH_COSTS}")
    if not cost_ca < cost_def:
        print("SANITY GATE FAILED: cost-aware threshold is not cheaper than 0.50.")
        return 1

    # Demo-sanity gate (hard checkpoint): presets must land in their bands.
    print("\nPreset sanity gate:")
    probs: dict[str, float] = {}
    ok = True
    for name, inputs in PRESETS.items():
        p = predict_failure_probability(bundle, transform(inputs))
        level = classify(p)
        probs[name] = p
        print(f"  {name:<5} p={p:.4f} -> {level.value}")

    low_p, watch_p, high_p = probs["LOW"], probs["WATCH"], probs["HIGH"]
    checks = [
        ("LOW < 0.05", low_p < 0.05),
        ("0.05 <= WATCH < 0.0895", 0.05 <= watch_p < 0.0895),
        ("HIGH >= 0.0895", high_p >= 0.0895),
        ("LOW < WATCH < HIGH", low_p < watch_p < high_p),
    ]
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    if not ok:
        print("\nSANITY GATE FAILED: tune the generator intercept or preset values.")
        return 1

    path = save_bundle(bundle)
    print(f"\nSaved bundle -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
