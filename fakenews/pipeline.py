"""The full checker: style model + evidence-based fact-checking.

Decision rule (evidence beats style):
1. If any claim is REFUTED by evidence      -> "Likely fake"  (decided by evidence)
2. If every claim is SUPPORTED by evidence  -> "Likely real"  (decided by evidence)
3. Otherwise there is not enough evidence   -> use the style model,
   and say clearly that the facts could not be checked.
"""
from dataclasses import asdict

from .factcheck import REFUTES, SUPPORTS, extract_claims, gather_evidence, get_nli, verify_claim
from .style_model import StyleModel


class FakeNewsChecker:
    def __init__(self, style_model: StyleModel = None, nli="auto", sources=None, max_claims: int = 3):
        self.style = style_model or StyleModel.load()
        self.nli = get_nli() if nli == "auto" else nli
        self.sources = sources          # None = default sources (fact-check API + Wikipedia)
        self.max_claims = max_claims

    def check(self, text: str, use_evidence: bool = True) -> dict:
        style = self.style.predict(text)
        claims, errors = [], []
        if use_evidence:
            for c in extract_claims(text, self.max_claims):
                evidence, errs = gather_evidence(c, self.sources)
                errors += errs
                claims.append(verify_claim(c, evidence, self.nli))
        verdict, decided_by, reason = self.decide(style, claims)
        return {
            "verdict": verdict,
            "decided_by": decided_by,
            "reason": reason,
            "style": asdict(style),
            "claims": claims,
            "nli_enabled": self.nli is not None,
            "source_errors": sorted(set(errors)),
        }

    @staticmethod
    def decide(style, claims: list) -> tuple:
        refuted = [c for c in claims if c["verdict"] == REFUTES]
        supported = [c for c in claims if c["verdict"] == SUPPORTS]
        if refuted:
            c = max(refuted, key=lambda c: c["confidence"])
            return "Likely fake", "evidence", f'Evidence contradicts: "{c["claim"]}"'
        if claims and len(supported) == len(claims):
            return "Likely real", "evidence", "Evidence supports every claim found."
        if style.label == "Out of scope":
            return "Unverified", "none", "No evidence found and the words are unknown to the style model."
        word = "fake" if style.label == "Fake" else "real"
        return (f"Unverified (style looks {word})", "style",
                f"No clear evidence found, so this is only a writing-style guess "
                f"({style.confidence:.0%} sure). Check trusted sources.")
