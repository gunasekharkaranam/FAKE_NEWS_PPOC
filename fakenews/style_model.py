"""Style model: TF-IDF + Logistic Regression.

It looks at HOW a text is written (word choice, sensational phrases).
It does NOT know whether a claim is true; that is the fact-checker's job.
"""
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocess import _stemmer, clean_text, style_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "style_model.joblib"


def build_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                  min_df=2, max_features=50_000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
    ])


@dataclass
class StyleResult:
    label: str                 # "Fake" | "Real" | "Out of scope"
    fake_probability: float    # 0..1
    confidence: float          # probability of the chosen label
    top_words: list = field(default_factory=list)   # [{word, weight}] (+ = pushes to Fake)
    features: dict = field(default_factory=dict)


class StyleModel:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.vec: TfidfVectorizer = pipeline.named_steps["tfidf"]
        self.clf: LogisticRegression = pipeline.named_steps["clf"]
        self.vocab = np.array(self.vec.get_feature_names_out())

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "StyleModel":
        return cls(joblib.load(path))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path, compress=3)

    def predict(self, text: str, n_words: int = 8) -> StyleResult:
        x = self.vec.transform([text])
        feats = style_features(text)
        if x.nnz == 0:  # none of the words are known to the model
            return StyleResult("Out of scope", 0.5, 0.0, [], feats)
        p_fake = float(self.clf.predict_proba(x)[0, 1])
        label = "Fake" if p_fake >= 0.5 else "Real"
        conf = p_fake if label == "Fake" else 1 - p_fake
        return StyleResult(label, round(p_fake, 4), round(conf, 4),
                           self.explain(text, x, n_words), feats)

    def explain(self, text: str, x=None, n_words: int = 8) -> list:
        """Word-level reasons. For a linear model this is exact:
        contribution = tf-idf value x learned weight (+ means 'looks fake')."""
        x = self.vec.transform([text]) if x is None else x
        idx = x.indices
        contrib = x.data * self.clf.coef_[0, idx]
        order = np.argsort(-np.abs(contrib))[:n_words]
        stem_to_word = {}
        for w in str(text).split():
            core = "".join(c for c in w.lower() if c.isalpha())
            if core:
                stem_to_word.setdefault(_stemmer.stem(core), core)
        out = []
        for i in order:
            term = self.vocab[idx[i]]
            shown = " ".join(stem_to_word.get(t, t) for t in term.split())
            out.append({"word": shown, "weight": round(float(contrib[i]), 4)})
        return out

    def global_top_words(self, n: int = 20) -> dict:
        coef = self.clf.coef_[0]
        return {"fake": self.vocab[np.argsort(-coef)[:n]].tolist(),
                "real": self.vocab[np.argsort(coef)[:n]].tolist()}
