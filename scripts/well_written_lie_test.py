"""The "well-written lie" test.

25 calm, correctly written sentences: 15 false, 10 true. No shocking words,
no CAPS, no "!!!". A style-only model has nothing to go on here.

    python -m scripts.well_written_lie_test                  # style model only
    python -m scripts.well_written_lie_test --with-evidence  # full checker (needs internet + NLI model)
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-evidence", action="store_true")
    args = ap.parse_args()

    from fakenews.style_model import StyleModel
    df = pd.read_csv(ROOT / "data" / "well_written_lies.csv")
    style = StyleModel.load()
    df["style_says"] = [style.predict(c).label.lower() for c in df["claim"]]
    cols = ["claim", "label", "style_says"]

    if args.with_evidence:
        from fakenews.pipeline import FakeNewsChecker
        checker = FakeNewsChecker(style)
        verdicts = [checker.check(c) for c in df["claim"]]
        df["checker_says"] = ["fake" if "fake" in v["verdict"].lower() else
                              "real" if "real" in v["verdict"].lower() else "unsure" for v in verdicts]
        df["decided_by"] = [v["decided_by"] for v in verdicts]
        cols += ["checker_says", "decided_by"]

    lines = ["# Well-written lie test", "",
             "25 calm sentences (15 false, 10 true) with no sensational style.", ""]
    for col in cols[2:]:
        if col.endswith("_says"):
            acc = (df[col] == df["label"]).mean()
            caught = ((df[col] == "fake") & (df["label"] == "fake")).sum()
            lines.append(f"- **{col}**: accuracy {acc:.0%}, caught {caught}/15 false claims")
    lines += ["", df[cols].to_markdown(index=False), ""]
    name = "well_written_lies_full.md" if args.with_evidence else "well_written_lies_style.md"
    (ROOT / "reports" / name).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
