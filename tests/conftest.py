import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("ENABLE_NLI", "0")  # tests never download the big model


class FakeNLI:
    """Pretend NLI model: says 'contradiction' if the evidence contains a word
    marked as contradicting, 'entailment' if it repeats the key word."""

    def __init__(self, contradict_if: str, support_if: str):
        self.c, self.s = contradict_if, support_if

    def __call__(self, premise, hypothesis):
        if self.c in premise:
            return {"contradiction": 0.95, "neutral": 0.04, "entailment": 0.01}
        if self.s in premise:
            return {"contradiction": 0.01, "neutral": 0.04, "entailment": 0.95}
        return {"contradiction": 0.1, "neutral": 0.8, "entailment": 0.1}


@pytest.fixture(scope="session")
def style_model():
    from fakenews.style_model import StyleModel
    return StyleModel.load()
