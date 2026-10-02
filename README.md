# FactoryGuard AI

**Cost-Aware AI Predictive Maintenance for Manufacturing Equipment — research prototype demo**

## What is FactoryGuard AI?

FactoryGuard AI is a local web demo of an AI-assisted predictive-maintenance
workflow for manufacturing equipment. A user enters six machine operating
parameters (product type, air temperature, process temperature, rotational
speed, torque, tool wear) and receives:

1. a failure-risk score,
2. a risk level (LOW / WATCH / HIGH),
3. a maintenance recommendation,
4. the main model contributions behind the prediction, and
5. a compact explanation of the cost-aware decision logic.

Core workflow: **Machine Data → ML Prediction → Risk Score → Cost-aware
Decision → Maintenance Recommendation**

> **This prototype demonstrates a research workflow using synthetic industrial
> telemetry. It has not been validated on a real production line.**

AI output is decision support, not an autonomous maintenance command.

## Research Basis

The project is reference-based on the **UCI AI4I 2020 Predictive Maintenance
Dataset** — the documented structure and failure mechanisms of that dataset
(product types L/M/H, air and process temperature, rotational speed, torque,
tool wear) inspire the synthetic benchmark used here.

- Benchmark: a **synthetic 10,000-row** benchmark generated locally by
  `scripts/train.py`. It is *inspired by* the documented structure of UCI
  AI4I 2020; it does **not** replicate the original AI4I label-generation
  rules and is not a reproduction of the published dataset.
- Model: Random Forest classifier (scikit-learn), trained offline and
  persisted to `models/factoryguard_model.pkl`.
- Research benchmark metrics reported by the study (displayed in the app's
  *Research context* panel as **benchmark results only**):
  - Test ROC-AUC: **0.968**
  - Test PR-AUC: **0.593**
  - Test F1: **0.642**
  - Cost-aware threshold: **0.0895**
  - Cost-aware recall: **96.4%**

All reported research metrics come from a synthetic computational benchmark.
They are **not** measurements from a physical factory, and they are **not**
live predictions for any particular machine. The demo does not claim
"96.4% accurate for your machine" or any cost-savings figure.

## How to Run

Requires Python 3.10+. Network access is needed only for the initial
`pip install`; the app runs fully offline afterwards.

```bash
cd factoryguard-ai
pip install -r requirements.txt
streamlit run app.py
```

The model pickle is trained automatically if missing (`scripts/train.py` is
the recommended explicit step):

```bash
python scripts/train.py
```

## Demo Workflow

The dashboard is designed for a 60–90 second walkthrough:

1. Open the dashboard (left column: **Machine Inputs**, right column: **Risk Result**).
2. Press **LOW RISK preset** → **Analyze** → low predicted risk, action
   *Continue Monitoring*.
3. Press **WATCH preset** → **Analyze** → elevated risk (band
   0.05 ≤ p < 0.0895), action *Inspect & Increase Monitoring*, model
   contributions highlighted.
4. Press **HIGH preset** → **Analyze** → high risk (p ≥ 0.0895), action
   *Schedule Maintenance*, main signals shown.
5. Expand **"Why not use 50%?"** → point at the cost-aware threshold
   **0.0895** vs the default 0.50 and the research cost matrix.
6. Expand **Research context** → show the benchmark metrics.

Preset definitions (demo examples only):

| Preset | Meaning |
|---|---|
| LOW RISK | A normal operating machine (nominal temperatures, mid speed, low torque, little wear). |
| WATCH | Elevated tool wear and torque. |
| HIGH | Clearly abnormal conditions (heavy wear, high torque, large speed deviation). |

Risk bands are **demo policy thresholds**, not universal industrial safety
standards: LOW p < 0.05, WATCH 0.05 ≤ p < 0.0895, HIGH p ≥ 0.0895.

## Limitations

- **Synthetic data.** The model is trained on a synthetic benchmark inspired
  by UCI AI4I 2020; it has never seen a real production line.
- **Hypothetical cost matrix.** The cost-aware threshold (0.0895) derives
  from research assumptions FN = 100, FP = 5, TP = 1, TN = 0 — a research
  assumption, not real factory economics.
- **No real factory validation.** No SCADA/PLC integration, no field trials,
  no measured maintenance outcomes. Not production-ready.
- **Decision support only.** AI output is decision support, not an
  autonomous maintenance command. Explainability sections report *model
  contribution*, never causal effects.
- Benchmark metrics (ROC-AUC 0.968 / PR-AUC 0.593 / F1 0.642 / recall 96.4%)
  are research benchmark results on the synthetic test split — they do not
  describe accuracy on any specific machine.

## AI Usage Disclosure

AI tools were used for development assistance and code generation in the
creation of this prototype (architecture design, module implementation, and
documentation drafting). All benchmark metrics shown in the app originate
from the documented research scope; no real-world performance results were
generated or implied by AI tooling.
