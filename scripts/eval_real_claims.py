"""Evaluate the full checker on REAL fact-checked claims (LIAR test set).

Reports two honest numbers:
* coverage  - how often evidence was strong enough to decide
* accuracy  - when evidence decided, how often it was right
               (compared with the style model on the same claims)

Needs internet + the NLI model, so run it in Colab:
    python -m scripts.eval_real_claims --n 200
"""
import argparse
import json
from pathlib import Path

from fakenews.data import download_all, load_splits
from fakenews.pipeline import FakeNewsChecker

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    download_all()
    _, test = load_splits("liar")
    sample = test.sample(args.n, random_state=7).reset_index(drop=True)
    checker = FakeNewsChecker()

    rows = []
    for i, r in sample.iterrows():
        res = checker.check(r["text"])
        truth = "fake" if r["label"] == 1 else "real"
        rows.append({
            "truth": truth,
            "style": res["style"]["label"].lower(),
            "decided_by": res["decided_by"],
            "evidence": "fake" if res["verdict"] == "Likely fake" else
                        "real" if res["verdict"] == "Likely real" else None,
        })
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(sample)} done")

    decided = [r for r in rows if r["evidence"]]
    out = {
        "n": len(rows),
        "coverage": round(len(decided) / len(rows), 3),
        "evidence_accuracy_when_decided": round(sum(r["evidence"] == r["truth"] for r in decided) / max(len(decided), 1), 3),
        "style_accuracy_same_claims": round(sum(r["style"] == r["truth"] for r in decided) / max(len(decided), 1), 3),
        "style_accuracy_all": round(sum(r["style"] == r["truth"] for r in rows) / len(rows), 3),
    }
    (ROOT / "reports" / "real_claims_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
