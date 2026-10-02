"""Final acceptance + claim audit for FactoryGuard AI.

Run:  python scripts/final_audit.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import classify  # noqa: E402
from src.model import ensure_bundle, predict_failure_probability  # noqa: E402
from src.preprocessing import transform, validate  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)


def source_files() -> list[Path]:
    return [
        p for p in ROOT.rglob("*")
        if p.suffix in {".py", ".md", ".csv", ".txt"}
        and "__pycache__" not in p.parts
        and p.name != "final_audit.py"
    ]


def main() -> int:
    texts = {p: p.read_text(encoding="utf-8") for p in source_files()}
    all_text = "\n".join(texts.values())
    # Normalize so line-wrapped / quote-concatenated string literals still match.
    norm_all = re.sub(r"[\s\"']+", "", all_text)

    print("\n[1] Mandated sentences present:")
    mandated = [
        "This prototype demonstrates a research workflow using synthetic industrial telemetry. It has not been validated on a real production line.",
        "AI output is decision support, not an autonomous maintenance command.",
        "Research assumption — not real factory economics.",
        "Higher tool wear, torque and abnormal rotational speed contributed strongly to the model's risk assessment.",
        "All reported research metrics come from a synthetic computational benchmark.",
        "Demo policy thresholds",
    ]
    for phrase in mandated:
        needle = re.sub(r"[\s\"']+", "", phrase)
        check(f"present: {phrase[:70]!r}", needle in norm_all)

    print("\n[2] Claim audit (source files, excluding audit/verify tooling):")
    project_text = "\n".join(
        t for p, t in texts.items() if p.name not in {"verify_app.py", "README.md"}
    )
    readme = texts.get(ROOT / "README.md", "")

    for pattern in ["accurate for your machine", "cost savings", "production-ready",
                    "causes failure", "autonomous shutdown"]:
        # app/src code: only allowed inside explicit negations or mandated disclaimers
        hits = [ln for ln in project_text.splitlines()
                if re.search(re.escape(pattern), ln, re.I)]
        bad = [ln for ln in hits
               if not re.search(r"not |never|no |n't|DISCLAIMER|forbidden|FORBIDDEN", ln)]
        check(f"code: {pattern!r} only in negated/disclaimer context", not bad, "; ".join(bad[:2]))

    neg = len(re.findall(r"not been validated on a real production line", all_text, re.I))
    pos = len(re.findall(r"(?<!not )been validated on a real production line", all_text, re.I))
    check("no positive 'validated on a real production line' claim", pos == 0,
          f"positive={pos} negated={neg}")

    for pattern in ["96.4% accurate for your machine", "cost savings", "production-ready"]:
        paras = [p for p in re.split(r"\n\s*\n", readme)
                 if re.search(re.escape(pattern), p, re.I)]
        bad = [p.strip().replace("\n", " ")[:120] for p in paras
               if not re.search(r"not|never|no claim|does not", p, re.I)]
        check(f"README: {pattern!r} only negated in its paragraph", not bad, "; ".join(bad[:2]))

    print("\n[3] Offline check:")
    urls = [f"{p.name}:{m.group(0)}" for p, t in texts.items()
            for m in re.finditer(r"https?://[^\s\"')]+", t)
            if "localhost" not in m.group(0) and "127.0.0.1" not in m.group(0)]
    check("no external URLs in code/deps", not urls, str(urls[:3]))
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    allowed = {"streamlit", "pandas", "numpy", "scikit-learn", "matplotlib"}
    deps = {re.split(r"[<>=]", ln.strip())[0] for ln in req.splitlines() if ln.strip() and not ln.startswith("#")}
    check("requirements only allowed deps", deps <= allowed, str(deps - allowed))
    check("no plotly/requests/openai refs",
          not re.search(r"plotly|openai|requests\.|urllib", project_text))

    print("\n[4] README structure:")
    readme = texts[ROOT / "README.md"]
    for section in ["What is FactoryGuard AI?", "Research Basis", "How to Run",
                    "Demo Workflow", "Limitations", "AI Usage Disclosure"]:
        check(f"README section: {section}", section in readme)
    check("README run command", "streamlit run app.py" in readme)
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    check("requirements.txt non-empty", all(d in reqs for d in allowed))

    print("\n[5] sample_data.csv classifications:")
    bundle = ensure_bundle()
    with open(ROOT / "data" / "sample_data.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = {k: row[k] for k in (
                "product_type", "air_temperature", "process_temperature",
                "rotational_speed", "torque", "tool_wear")}
            inputs, errs = validate(raw)
            check(f"validate {row['row_name']}", not errs, str(errs))
            p = predict_failure_probability(bundle, transform(inputs))
            level = classify(p).value
            check(f"{row['row_name']:<14} p={p:.4f} -> {level} (expected {row['expected_level']})",
                  level == row["expected_level"])

    print("\n[6] Model artifact:")
    pkl = ROOT / "models" / "factoryguard_model.pkl"
    check("factoryguard_model.pkl exists", pkl.exists())
    check("bundle loads", bundle is not None)

    print(f"\n{'ALL AUDIT CHECKS PASSED' if not failures else 'AUDIT FAILURES:'}")
    for f in failures:
        print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
