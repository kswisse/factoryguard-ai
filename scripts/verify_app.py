"""AppTest acceptance walk for FactoryGuard AI.

Run:  python scripts/verify_app.py

Exercises the full demo flow (three presets -> Analyze) and asserts the
result card, contributions, cost-aware explainer, research context, and
required disclaimers render — with no NaN/None and no forbidden claims.
"""

from __future__ import annotations

import re

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

FORBIDDEN = [
    "accurate for your machine",
    "cost savings",
    "production-ready",
    "96.4% accurate",
    "causes failure",
    "validated on a real production line",  # only allowed inside the negation
]

REQUIRED = [
    "FactoryGuard AI",
    "AI Predictive Maintenance Dashboard",
    "has not been validated on a real production line",
    "AI output is decision support, not an autonomous maintenance command.",
    "Demo policy thresholds",
    "Research assumption — not real factory economics.",
    "Research context",
    "Why not use 50%?",
    "All reported research metrics come from a synthetic computational benchmark.",
    "0.968",
    "0.593",
    "0.642",
    "0.0895",
    "Model Contributions",
]

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(f"{label}: {detail}")


def page_text(at: AppTest) -> str:
    parts: list[str] = []
    parts += [m.value for m in at.markdown]
    parts += [t.value for t in at.title]
    parts += [c.value for c in at.caption]
    parts += [i.value for i in at.info]
    parts += [w.value for w in at.warning]
    parts += [e.value for e in at.error]
    parts += [b.label for b in at.button]
    parts += [w.label for w in at.selectbox]
    parts += [w.label for w in at.number_input]
    parts += [s.value for s in at.get("subheader")]
    parts += [e.label for e in at.get("expander")]
    for m in at.metric:
        parts += [m.label, m.value]
    return "\n".join(parts)


def has_bad_value(text: str) -> bool:
    """True when a literal NaN/None/null token is rendered in the page."""
    return re.search(r"\b(NaN|None|null|NoneType)\b", text) is not None


def run_preset(at: AppTest, btn_key: str) -> None:
    at.button(key=btn_key).click().run()
    at.button(key="btn_analyze").click().run()


def main() -> int:
    print("Loading app (first run trains/loads the bundle if needed)...")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    check("app runs without exception", not at.exception, str(at.exception))

    text = page_text(at)
    for phrase in REQUIRED:
        check(f"required text present: {phrase[:60]!r}", phrase in text)

    # --- Preset LOW -------------------------------------------------------
    print("\nPreset LOW:")
    run_preset(at, "btn_preset_low")
    check("no exception after LOW", not at.exception, str(at.exception))
    text = page_text(at)
    check("LOW risk badge", "LOW RISK" in text)
    check("LOW action", "Continue Monitoring" in text)
    check("probability rendered", "%" in text)
    check("no NaN/None rendered", not has_bad_value(text))

    # --- Preset WATCH -----------------------------------------------------
    print("\nPreset WATCH:")
    run_preset(at, "btn_preset_watch")
    check("no exception after WATCH", not at.exception, str(at.exception))
    text = page_text(at)
    check("WATCH risk badge", "WATCH RISK" in text)
    check("WATCH action", "Inspect & Increase Monitoring" in text)

    # --- Preset HIGH ------------------------------------------------------
    print("\nPreset HIGH:")
    run_preset(at, "btn_preset_high")
    check("no exception after HIGH", not at.exception, str(at.exception))
    text = page_text(at)
    check("HIGH risk badge", "HIGH RISK" in text)
    check("HIGH action", "Schedule Maintenance" in text)
    check("main signals rendered", "Main Signals" in text)
    check("decision-support disclaimer", "AI output is decision support" in text)
    check("contribution explanation rendered",
          "Higher tool wear, torque and abnormal rotational speed" in text)

    # --- Forbidden claims (page-wide) -------------------------------------
    print("\nClaim audit:")
    lowered = text.lower()
    for phrase in FORBIDDEN:
        if phrase == "validated on a real production line":
            # allowed only inside the negated integrity statement
            occurrences = lowered.count(phrase)
            negations = lowered.count("not been validated on a real production line")
            check(f"forbidden phrase absent (or negated): {phrase!r}",
                  occurrences <= negations, f"{occurrences} hits, {negations} negations")
        else:
            check(f"forbidden phrase absent: {phrase!r}", phrase not in lowered)

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES:'}")
    for f in failures:
        print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
