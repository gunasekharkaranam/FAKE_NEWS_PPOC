"""Step 1 of fact-checking: find the sentences that make checkable claims.

A "checkable" sentence states a fact: it names things (capital letters),
often has numbers, and is not a question or an opinion.
Simple rules, so every choice can be explained.
"""
import re

_sent_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_opinion = re.compile(r"\b(i think|i believe|in my opinion|we feel|should|must|amazing|terrible)\b", re.I)


def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return [s.strip() for s in _sent_split.split(text) if s.strip()]


def claim_score(sentence: str) -> float:
    words = sentence.split()
    if not 4 <= len(words) <= 60 or sentence.endswith("?"):
        return 0.0
    score = 1.0
    score += 1.0 * bool(re.search(r"\d", sentence))                  # numbers, dates, money
    score += 0.5 * sum(1 for w in words[1:] if w[:1].isupper())      # names of people/places
    score += 0.5 * bool(re.search(r"\b(is|are|was|were|has|have|had|will|did)\b", sentence, re.I))
    score -= 1.5 * bool(_opinion.search(sentence))
    return max(score, 0.0)


def extract_claims(text: str, max_claims: int = 3) -> list:
    """Return up to `max_claims` sentences, best first."""
    sents = split_sentences(text)
    scored = [(claim_score(s), i, s) for i, s in enumerate(sents)]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: (-x[0], x[1]))
    claims = [s.strip(" \"'") for _, _, s in scored[:max_claims]]
    # A one-line input (e.g. a headline without a full stop) is itself the claim.
    if not claims and text.strip():
        claims = [text.strip()[:400]]
    return claims
