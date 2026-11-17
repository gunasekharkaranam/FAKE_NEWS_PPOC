"""Step 3 of fact-checking: does the evidence SUPPORT or REFUTE the claim?

Two ways to decide:
1. Fact-checker rating (e.g. "False", "Misleading") -> read it directly.
2. NLI model (Natural Language Inference). It reads two texts:
       premise    = evidence sentence
       hypothesis = the claim
   and outputs entailment (supports) / neutral / contradiction (refutes).
   This is what catches a calm, well-written lie: style does not matter,
   only whether the facts agree.

Default NLI model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
(trained on MNLI + FEVER fact-verification + ANLI). Needs `transformers` + `torch`.
"""
from functools import lru_cache
import os
import re

from .retrieve import _overlap as word_overlap

SUPPORTS, REFUTES, NEI, MIXED = "SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "MIXED"

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

_REFUTE_WORDS = r"false|fake|pants on fire|incorrect|wrong|hoax|misleading|no evidence|" \
                r"unproven|fabricated|baseless|not true|scam|satire|distorts|exaggerat|mostly false|four pinocchios"
_SUPPORT_WORDS = r"\btrue\b|correct|accurate|mostly true|legit|confirmed"
_MIXED_WORDS = r"half|mixture|mixed|partly|partially|needs context|missing context"


def rating_to_verdict(rating: str) -> str:
    r = rating.lower()
    if re.search(_MIXED_WORDS, r):
        return MIXED
    if re.search(_REFUTE_WORDS, r):
        return REFUTES
    if re.search(_SUPPORT_WORDS, r):
        return SUPPORTS
    return NEI


class LocalNLI:
    """Runs the NLI model on this machine (CPU is fine, ~0.3 s per pair)."""

    def __init__(self, model_name: str = None):
        model_name = model_name or os.getenv("NLI_MODEL") or DEFAULT_NLI_MODEL
        from transformers import pipeline  # imported here so the app still runs without it
        self.pipe = pipeline("text-classification", model=model_name, top_k=None, truncation=True)

    def __call__(self, premise: str, hypothesis: str) -> dict:
        out = self.pipe({"text": premise, "text_pair": hypothesis})
        if out and isinstance(out[0], list):
            out = out[0]
        return {d["label"].lower(): float(d["score"]) for d in out}


@lru_cache(maxsize=1)
def get_nli():
    """Return an NLI callable, or None if it is turned off / not installed."""
    if os.getenv("ENABLE_NLI", "1") == "0":
        return None
    try:
        return LocalNLI()
    except Exception as e:
        print(f"[verify] NLI model not available ({type(e).__name__}); using fact-check ratings only")
        return None


def nli_verdict(scores: dict, threshold: float = 0.6) -> tuple:
    ent = scores.get("entailment", 0.0)
    con = scores.get("contradiction", 0.0)
    if con >= threshold and con > ent:
        return REFUTES, con
    if ent >= threshold and ent > con:
        return SUPPORTS, ent
    return NEI, max(scores.get("neutral", 0.0), 1 - max(ent, con))


def verify_claim(claim: str, evidence: list, nli=None) -> dict:
    """Judge one claim against all its evidence.

    Fact-checker ratings come first (experts already checked it).
    Then NLI on each evidence text. The strongest decision wins.
    """
    judged = []
    for ev in evidence:
        if ev.source == "factcheck" and ev.rating:
            v = rating_to_verdict(ev.rating)
            # A fact-check about a DIFFERENT claim must not decide ours: check word overlap first.
            if word_overlap(claim, ev.text) < 0.3:
                v = NEI
            conf = 0.9 if v in (SUPPORTS, REFUTES) else 0.5 if v == MIXED else 0.0
            judged.append({**ev.to_dict(), "verdict": v, "confidence": conf, "method": "fact-checker rating"})
        elif nli is not None and ev.text:
            v, conf = nli_verdict(nli(ev.text, claim))
            judged.append({**ev.to_dict(), "verdict": v, "confidence": round(conf, 3), "method": "NLI model"})
        else:
            judged.append({**ev.to_dict(), "verdict": NEI, "confidence": 0.0, "method": "not checked"})

    decisive = [j for j in judged if j["verdict"] in (SUPPORTS, REFUTES)]
    if not decisive:
        mixed = [j for j in judged if j["verdict"] == MIXED]
        verdict = MIXED if mixed else NEI
        conf = 0.5 if mixed else 0.0
    else:
        best = max(decisive, key=lambda j: j["confidence"])
        verdict, conf = best["verdict"], best["confidence"]
    return {"claim": claim, "verdict": verdict, "confidence": conf, "evidence": judged}
