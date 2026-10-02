"""Real-browser UI QA for FactoryGuard AI (headless Chromium via Playwright).

Run:
  python scripts/qa_browser.py            # if the app is already running on :8520
  (or via webapp-testing with_server.py for server lifecycle management)

Verifies layout, demo flow (LOW -> WATCH -> HIGH -> cost-aware -> research
context), offline behavior (no external requests), no console errors, and
no horizontal overflow. Saves screenshots to artifacts/qa/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "qa"
OUT.mkdir(parents=True, exist_ok=True)

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8520"

checks: dict[str, bool] = {}
notes: dict[str, object] = {}
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


requests_seen: list[str] = []
console_errors: list[str] = []


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("request", lambda r: requests_seen.append(r.url))
        page.on(
            "console",
            lambda m: console_errors.append(m.text) if m.type == "error" else None,
        )

        print(f"\n[1] Load: {BASE}")
        page.goto(BASE, wait_until="domcontentloaded")
        page.get_by_role("heading", name="FactoryGuard AI").first.wait_for(
            timeout=60000
        )
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "01_initial.png"), full_page=True)

        check("header visible",
              page.get_by_role("heading", name="FactoryGuard AI").first.is_visible())
        check("subtitle visible",
              page.get_by_text("AI Predictive Maintenance Dashboard").first.is_visible())
        check("integrity disclaimer visible",
              page.get_by_text("has not been validated on a real production line").first.is_visible())

        print("\n[2] Inputs & controls")
        for label in ["Product Type", "Air Temperature [K]", "Process Temperature [K]",
                      "Rotational Speed [rpm]", "Torque [Nm]", "Tool Wear [min]"]:
            visible = page.get_by_label(label, exact=False).first.is_visible()
            check(f"input visible: {label}", visible)
        for name in ["LOW RISK preset", "WATCH preset", "HIGH preset"]:
            check(f"preset visible: {name}",
                  page.get_by_role("button", name=name).first.is_visible())
        check("Analyze visible",
              page.get_by_role("button", name="Analyze").first.is_visible())
        check("empty-state prompt",
              page.get_by_text("Choose a preset or enter machine parameters").first.is_visible())

        print("\n[3] Demo flow")
        flow = [
            ("LOW RISK preset", "Continue Monitoring", "0.0%"),
            ("WATCH preset", "Inspect & Increase Monitoring", None),
            ("HIGH preset", "Schedule Maintenance", None),
        ]
        for preset, action, pct in flow:
            level = preset.split()[0]
            page.get_by_role("button", name=preset).first.click()
            # Wait for the preset callback's rerun to land (caption updates).
            page.get_by_text(f"Active preset: {level}").first.wait_for(timeout=20000)
            page.wait_for_timeout(300)
            page.get_by_role("button", name="Analyze").first.click()
            page.get_by_text(action).first.wait_for(timeout=20000)
            page.wait_for_timeout(900)
            # Streamlit scrolls inside its container; center the result card top.
            page.get_by_text("MACHINE RISK").first.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            shot = OUT / f"02_result_{level.lower()}.png"
            page.screenshot(path=str(shot))
            page.get_by_text("Main Signals").first.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            page.mouse.wheel(0, 450)
            page.wait_for_timeout(300)
            page.screenshot(path=str(OUT / f"02_signals_{level.lower()}.png"))
            badge_ok = page.get_by_text(
                f"{level} RISK", exact=True
            ).first.is_visible()
            check(f"{preset}: badge + action rendered", badge_ok)
            if pct:
                check(f"{preset}: probability {pct} shown",
                      page.get_by_text(pct, exact=False).first.is_visible())
            check(f"{preset}: decision-support disclaimer",
                  page.get_by_text("AI output is decision support").first.is_visible())
            check(f"{preset}: Main Signals readable",
                  page.get_by_text("Main Signals").first.is_visible())

        print("\n[4] Explainability (HIGH state)")
        check("contribution chart section",
              page.get_by_text("Model Contributions").first.is_visible())
        check("canonical explanation",
              page.get_by_text("Higher tool wear, torque and abnormal rotational speed").first.is_visible())
        check("non-causal caption",
              page.get_by_text("not a causal attribution").first.is_visible())

        print("\n[5] Cost-aware expander")
        page.get_by_text("Why not use 50%? Cost-aware threshold logic").first.scroll_into_view_if_needed()
        page.get_by_text("Why not use 50%? Cost-aware threshold logic").first.click()
        page.wait_for_timeout(1500)
        page.get_by_text("Cost-aware threshold (research)").first.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "03_cost_top.png"))
        page.get_by_text("Research assumption — not real factory economics.").first.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "03_cost_aware.png"))
        check("default threshold 0.50 shown",
              page.get_by_text("Default threshold").first.is_visible())
        check("cost-aware threshold shown",
              page.get_by_text("Cost-aware threshold (research)").first.is_visible())
        check("research assumption label",
              page.get_by_text("Research assumption — not real factory economics.").first.is_visible())
        for cost in ["Missed failure (FN)", "False alarm (FP)"]:
            check(f"cost matrix row: {cost}", page.get_by_text(cost).first.is_visible())
        check("policy disclaimer",
              page.get_by_text("not universal industrial safety standards").first.is_visible())

        print("\n[6] Research context expander")
        page.get_by_text("Research context", exact=True).first.scroll_into_view_if_needed()
        page.get_by_text("Research context", exact=True).first.click()
        page.wait_for_timeout(1200)
        page.get_by_text("Test ROC-AUC").first.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "04_research_context.png"))
        for metric in ["Test ROC-AUC", "Test PR-AUC", "Test F1", "0.968", "0.593", "0.642", "0.0895"]:
            check(f"research metric shown: {metric}",
                  page.get_by_text(metric, exact=False).first.is_visible())
        check("synthetic benchmark note",
              page.get_by_text("They are not measurements from a physical factory").first.is_visible())

        print("\n[7] Layout / robustness")
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - window.innerWidth"
        )
        notes["horizontal_overflow_px"] = overflow
        check("no horizontal overflow", overflow <= 2, f"{overflow}px")

        clipped = page.evaluate(
            """() => {
                const bad = [];
                for (const el of document.querySelectorAll('h1,h2,h3,h4,p,span,button,label,div')) {
                    if (!el.innerText || el.children.length > 0) continue;
                    const st = getComputedStyle(el);
                    if (st.overflow === 'hidden' && el.scrollWidth > el.clientWidth + 2) {
                        bad.push(el.innerText.slice(0, 60));
                    }
                }
                return bad.slice(0, 10);
            }"""
        )
        notes["clipped_text"] = clipped
        check("no clipped text", not clipped, str(clipped))

        print("\n[8] Offline / network")
        external = sorted({
            u.split("/")[2] for u in requests_seen
            if u.startswith("http") and "localhost" not in u and "127.0.0.1" not in u
        })
        notes["external_hosts"] = external
        check("no external network requests", not external, str(external))

        real_errors = [
            e for e in console_errors
            if "favicon" not in e.lower() and "manifest" not in e.lower()
        ]
        notes["console_errors"] = real_errors[:5]
        check("no console errors", not real_errors, str(real_errors[:3]))

        browser.close()

    (OUT / "qa_report.json").write_text(
        json.dumps({"checks": checks, "notes": notes, "failures": failures}, indent=2),
        encoding="utf-8",
    )
    print(f"\n{'QA PASSED' if not failures else 'QA FAILURES: ' + str(failures)}")
    print(f"screenshots -> {OUT}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
