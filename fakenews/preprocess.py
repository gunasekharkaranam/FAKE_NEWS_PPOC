"""Text cleaning shared by training AND the web app.

Keeping one function in one place means the model always sees text cleaned
the same way it was trained on (the old version had two different copies).
"""
import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negation words carry meaning for claims ("did NOT vote"), so we keep them.
KEEP_WORDS = {"no", "not", "nor", "never", "none", "nothing", "cannot"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS - KEEP_WORDS)

_stemmer = PorterStemmer()
_url_re = re.compile(r"https?://\S+|www\.\S+")
_mention_re = re.compile(r"@\w+")
_non_letter_re = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase, drop links/mentions/non-letters, remove stop words, stem."""
    text = str(text).lower()
    text = _url_re.sub(" ", text)
    text = _mention_re.sub(" ", text)
    text = text.replace("#", " ")
    text = _non_letter_re.sub(" ", text)
    tokens = [_stemmer.stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def style_features(text: str) -> dict:
    """Simple, human-readable writing-style signals shown in the UI."""
    text = str(text)
    words = text.split()
    n = max(len(words), 1)
    caps = sum(1 for w in words if len(w) > 2 and w.isupper())
    return {
        "exclamations": text.count("!"),
        "question_marks": text.count("?"),
        "all_caps_words": caps,
        "all_caps_ratio": round(caps / n, 3),
        "word_count": len(words),
    }
