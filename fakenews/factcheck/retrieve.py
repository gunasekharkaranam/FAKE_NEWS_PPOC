"""Step 2 of fact-checking: find evidence about a claim.

Sources
* Google Fact Check Tools API: claims already checked by real fact-checkers
  (PolitiFact, Alt News, BOOM, Reuters ...). Needs a free API key in
  GOOGLE_FACTCHECK_API_KEY.
* Wikipedia: background facts. No key needed.
Each source returns a list of Evidence objects, so new sources are easy to add.
"""
from dataclasses import asdict, dataclass
import os
import re

import requests

from ..preprocess import STOP_WORDS

UA = {"User-Agent": "FakeNewsChecker/2.0 (student project; https://github.com/gunasekharkaranam/FAKE_NEWS_PPOC)"}
TIMEOUT = 8


@dataclass
class Evidence:
    source: str              # "factcheck" | "wikipedia"
    title: str
    url: str
    text: str                # the sentence(s) we compare with the claim
    rating: str = ""         # fact-checker's rating, e.g. "False"
    publisher: str = ""

    def to_dict(self):
        return asdict(self)


def keywords(text: str, limit: int = 10) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text)
    kept = [w for w in words if w.lower() not in STOP_WORDS and len(w) > 2]
    return " ".join(kept[:limit])


# ---------- Google Fact Check Tools ----------
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def search_factchecks(claim: str, api_key: str = None, limit: int = 3, session=requests) -> list:
    api_key = api_key or os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        return []
    r = session.get(FACTCHECK_URL, params={"query": keywords(claim, 12), "key": api_key,
                                           "languageCode": "en", "pageSize": limit},
                    headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for item in r.json().get("claims", []):
        for review in item.get("claimReview", [])[:1]:
            out.append(Evidence(
                source="factcheck",
                title=review.get("title") or item.get("text", ""),
                url=review.get("url", ""),
                text=item.get("text", ""),
                rating=review.get("textualRating", ""),
                publisher=review.get("publisher", {}).get("name", ""),
            ))
    return out[:limit]


# ---------- Wikipedia ----------
WIKI_API = "https://en.wikipedia.org/w/api.php"


def _overlap(a: str, b: str) -> float:
    """Share of the claim's key words (a) that also appear in b."""
    wa = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", a) if w.lower() not in STOP_WORDS}
    wb = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", b) if w.lower() not in STOP_WORDS}
    return len(wa & wb) / (len(wa) or 1)


def best_sentences(claim: str, passage: str, k: int = 2) -> str:
    """Pick the k sentences from a passage that share most words with the claim."""
    sents = re.split(r"(?<=[.!?])\s+", passage)
    sents = [s for s in sents if 5 <= len(s.split()) <= 80]
    sents.sort(key=lambda s: -_overlap(claim, s))
    return " ".join(sents[:k])


def search_wikipedia(claim: str, limit: int = 3, session=requests) -> list:
    r = session.get(WIKI_API, params={"action": "query", "list": "search", "format": "json",
                                      "srsearch": keywords(claim), "srlimit": limit},
                    headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    titles = [h["title"] for h in r.json().get("query", {}).get("search", [])]
    if not titles:
        return []
    r = session.get(WIKI_API, params={"action": "query", "prop": "extracts", "explaintext": 1,
                                      "exintro": 1, "format": "json", "titles": "|".join(titles)},
                    headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {}).values()
    by_title = {p.get("title"): p.get("extract", "") for p in pages}
    out = []
    for t in titles:  # keep search ranking order
        text = best_sentences(claim, by_title.get(t, ""))
        if text:
            out.append(Evidence("wikipedia", t, "https://en.wikipedia.org/wiki/" + t.replace(" ", "_"), text))
    return out


def gather_evidence(claim: str, sources=None) -> tuple:
    """Run every source; a failing source is reported, not fatal."""
    sources = sources or {"factcheck": search_factchecks, "wikipedia": search_wikipedia}
    evidence, errors = [], []
    for name, fn in sources.items():
        try:
            evidence.extend(fn(claim))
        except Exception as e:  # network down, quota, bad key ...
            errors.append(f"{name}: {type(e).__name__}")
    return evidence, errors
