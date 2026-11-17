"""Train the style model and run the real-world tests.

Run:  python -m scripts.train_style_model

What it does
1. In-dataset test   : train on a dataset, test on its own held-out part.
2. Cross-dataset test: train on one dataset, test on a DIFFERENT one.
   This is the "does it work on real-world data?" test. A big drop means the
   model learned the style of one source, not general fake-news signals.
3. Trains the final model on all datasets together and saves it.
Results go to reports/results.json and reports/results.md.
"""
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from fakenews.data import DATASETS, download_all, load_splits
from fakenews.style_model import StyleModel, build_pipeline

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def metrics(y_true, p_fake) -> dict:
    y_pred = (p_fake >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "fake_f1": round(f1_score(y_true, y_pred, pos_label=1), 3),
        "roc_auc": round(roc_auc_score(y_true, p_fake), 3),
        "n": int(len(y_true)),
    }


def fit(train: pd.DataFrame, tune: bool = True):
    if not tune:
        return build_pipeline().fit(train["text"], train["label"])
    grid = GridSearchCV(build_pipeline(), {"clf__C": [0.3, 1.0, 3.0]},
                        cv=3, scoring="f1_macro", n_jobs=-1)
    grid.fit(train["text"], train["label"])
    print(f"   best C = {grid.best_params_['clf__C']}")
    return grid.best_estimator_


def main():
    t0 = time.time()
    download_all()
    splits = {name: load_splits(name) for name in DATASETS}
    for name, (tr, te) in splits.items():
        print(f"{name:11s} train={len(tr):6d} test={len(te):5d} fake share={tr['label'].mean():.2f}")

    results = {"label_rule": "1 = fake, 0 = real", "datasets": {}, "cross": {}, "final": {}}
    for name, (tr, te) in splits.items():
        results["datasets"][name] = {"train": len(tr), "test": len(te),
                                     "fake_share": round(float(tr["label"].mean()), 3)}

    # 1 + 2: train on each dataset, test on every dataset
    for src in DATASETS:
        print(f"\ntrain on {src}")
        model = fit(splits[src][0])
        results["cross"][src] = {}
        for dst in DATASETS:
            te = splits[dst][1]
            m = metrics(te["label"].values, model.predict_proba(te["text"])[:, 1])
            results["cross"][src][dst] = m
            print(f"   test on {dst:11s} macro-F1={m['macro_f1']:.3f} acc={m['accuracy']:.3f}")

    # 3: final model on everything
    print("\ntrain final model on all datasets")
    all_train = pd.concat([splits[n][0] for n in DATASETS], ignore_index=True)
    final = StyleModel(fit(all_train))
    for dst in DATASETS:
        te = splits[dst][1]
        results["final"][dst] = metrics(te["label"].values,
                                        final.pipeline.predict_proba(te["text"])[:, 1])
    results["top_words"] = final.global_top_words(15)
    final.save()

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    (REPORTS / "results.md").write_text(to_markdown(results))
    print(f"\nsaved model + reports in {time.time() - t0:.0f}s")


def to_markdown(r: dict) -> str:
    names = list(DATASETS)
    lines = ["# Style model results (TF-IDF + Logistic Regression)", "",
             "Label rule: **1 = fake, 0 = real**. Metric: **macro-F1** (fair when classes are unbalanced).", "",
             "## Cross-dataset test (rows = trained on, columns = tested on)", "",
             "| trained on \\ tested on | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for src in names:
        cells = []
        for dst in names:
            v = f"{r['cross'][src][dst]['macro_f1']:.3f}"
            cells.append(f"**{v}**" if src == dst else v)
        lines.append(f"| {src} | " + " | ".join(cells) + " |")
    lines += ["", "Bold = same dataset. Everything else = a dataset the model never saw.", "",
              "## Final model (trained on all three)", "",
              "| test set | accuracy | macro-F1 | fake-F1 | ROC-AUC | n |", "|---|---|---|---|---|---|"]
    for dst, m in r["final"].items():
        lines.append(f"| {dst} | {m['accuracy']} | {m['macro_f1']} | {m['fake_f1']} | {m['roc_auc']} | {m['n']} |")
    lines += ["", "## Words the model links most with each class", "",
              "- **Fake:** " + ", ".join(r["top_words"]["fake"]),
              "- **Real:** " + ", ".join(r["top_words"]["real"]), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
