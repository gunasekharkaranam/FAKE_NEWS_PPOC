"""Fine-tune a transformer (DistilBERT by default) on the same data and splits
as the style model, so the two can be compared fairly.

Needs a GPU to be quick -> run it in Google Colab:
    notebooks/01_distilbert_training.ipynb

Examples
    python -m scripts.train_transformer --train-on all
    python -m scripts.train_transformer --train-on liar politifact gossipcop all   # full cross-dataset table
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fakenews.data import DATASETS, download_all, load_splits
from scripts.train_style_model import metrics

ROOT = Path(__file__).resolve().parent.parent


def train_one(train_df, test_sets, args, save_dir=None):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def encode(df):
        ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)

    # Balance the loss like class_weight="balanced" in the style model.
    counts = np.bincount(train_df["label"], minlength=2)
    weights = torch.tensor(len(train_df) / (2 * counts), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    # warmup = first 10% of training steps (computed by hand: works on transformers v4 and v5)
    total_steps = args.max_steps if args.max_steps > 0 else int(np.ceil(len(train_df) / args.batch) * args.epochs)
    targs = TrainingArguments(
        output_dir=str(ROOT / "runs" / "tmp"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=64,
        learning_rate=args.lr, weight_decay=0.01, warmup_steps=int(0.1 * total_steps),
        fp16=torch.cuda.is_available(), save_strategy="no", report_to=[],
        logging_steps=50, seed=42, max_steps=args.max_steps,
    )
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=encode(train_df),
                              data_collator=DataCollatorWithPadding(tok))
    trainer.train()

    results = {}
    for name, te in test_sets.items():
        logits = trainer.predict(encode(te)).predictions
        p_fake = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        results[name] = metrics(te["label"].values, p_fake)
        print(f"   test on {name:11s} macro-F1={results[name]['macro_f1']:.3f}")
    if save_dir:
        trainer.save_model(save_dir)
        tok.save_pretrained(save_dir)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--train-on", nargs="+", default=["all"], choices=list(DATASETS) + ["all"])
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=-1, help="for quick smoke tests only")
    ap.add_argument("--limit", type=int, default=0, help="use only N rows per split (smoke tests)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "transformer_results.json"))
    args = ap.parse_args()

    download_all()
    splits = {n: load_splits(n) for n in DATASETS}
    if args.limit:
        splits = {n: (tr.sample(min(args.limit, len(tr)), random_state=0),
                      te.sample(min(args.limit, len(te)), random_state=0)) for n, (tr, te) in splits.items()}
    tests = {n: te for n, (_, te) in splits.items()}

    out = {"model": args.model, "epochs": args.epochs, "results": {}}
    for src in args.train_on:
        print(f"\ntrain {args.model} on {src}")
        train = (pd.concat([tr for tr, _ in splits.values()], ignore_index=True)
                 if src == "all" else splits[src][0])
        save = str(ROOT / "models" / "transformer") if src == "all" else None
        out["results"][src] = train_one(train, tests, args, save)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print_comparison(out)


def print_comparison(out):
    style = json.loads((ROOT / "reports" / "results.json").read_text())
    print("\nmacro-F1: style model (TF-IDF) vs transformer")
    for src, res in out["results"].items():
        base = style["final"] if src == "all" else style["cross"][src]
        for dst, m in res.items():
            print(f"  train {src:10s} test {dst:10s}  TF-IDF {base[dst]['macro_f1']:.3f}  ->  "
                  f"transformer {m['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
