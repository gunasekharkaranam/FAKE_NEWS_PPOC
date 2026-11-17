"""Download and load real-world fact-checked datasets.

Label rule used everywhere in this project:  1 = FAKE,  0 = REAL.

* LIAR       - 12.8k short political claims rated by PolitiFact fact-checkers.
               6 ratings -> 2 classes:
               pants-fire, false, barely-true  -> FAKE (1)
               half-true, mostly-true, true    -> REAL (0)
* PolitiFact - news headlines checked by PolitiFact (from FakeNewsNet).
* GossipCop  - entertainment headlines checked by GossipCop (from FakeNewsNet).
"""
from pathlib import Path
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

LIAR_URL = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/{}.tsv"
FNN_URL = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/{}.csv"

LIAR_FAKE = {"pants-fire", "false", "barely-true"}
LIAR_REAL = {"half-true", "mostly-true", "true"}
LIAR_COLS = ["id", "label", "statement", "subject", "speaker", "job", "state",
             "party", "c1", "c2", "c3", "c4", "c5", "context"]

DATASETS = ("liar", "politifact", "gossipcop")


def download_all(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {f"liar_{s}.tsv": LIAR_URL.format(s) for s in ("train", "valid", "test")}
    for src in ("politifact", "gossipcop"):
        for kind in ("fake", "real"):
            files[f"{src}_{kind}.csv"] = FNN_URL.format(f"{src}_{kind}")
    for name, url in files.items():
        path = raw_dir / name
        if path.exists():
            print(f"exists  {name}")
            continue
        print(f"get     {name}")
        urllib.request.urlretrieve(url, path)


def _load_liar_split(split: str, raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / f"liar_{split}.tsv", sep="\t", header=None,
                     names=LIAR_COLS, quoting=3)
    df = df[df["label"].isin(LIAR_FAKE | LIAR_REAL)]
    out = pd.DataFrame({"text": df["statement"].astype(str),
                        "label": df["label"].isin(LIAR_FAKE).astype(int)})
    return out.reset_index(drop=True)


def _load_fnn(source: str, raw_dir: Path) -> pd.DataFrame:
    parts = []
    for kind, label in (("fake", 1), ("real", 0)):
        df = pd.read_csv(raw_dir / f"{source}_{kind}.csv", usecols=["title"])
        parts.append(pd.DataFrame({"text": df["title"].astype(str), "label": label}))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["text"].str.strip().str.len() > 10]
    # The same headline sometimes appears twice; keep one copy so test rows never leak into train.
    return df.drop_duplicates("text").reset_index(drop=True)


def load_splits(name: str, raw_dir: Path = RAW_DIR, seed: int = 42):
    """Return (train_df, test_df) for one dataset with columns text, label."""
    if name == "liar":
        # LIAR ships official splits; we merge train+valid and keep the official test.
        train = pd.concat([_load_liar_split("train", raw_dir),
                           _load_liar_split("valid", raw_dir)], ignore_index=True)
        return train, _load_liar_split("test", raw_dir)
    if name in ("politifact", "gossipcop"):
        df = _load_fnn(name, raw_dir)
        train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
        return train.reset_index(drop=True), test.reset_index(drop=True)
    raise ValueError(f"unknown dataset {name!r}, choose from {DATASETS}")


if __name__ == "__main__":
    download_all()
