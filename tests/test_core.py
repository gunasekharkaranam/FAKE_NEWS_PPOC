from fakenews.factcheck.claims import extract_claims, split_sentences
from fakenews.factcheck.retrieve import Evidence, keywords
from fakenews.factcheck.verify import MIXED, NEI, REFUTES, SUPPORTS, rating_to_verdict, verify_claim
from fakenews.preprocess import clean_text, style_features

from conftest import FakeNLI


# ---------- preprocessing ----------
def test_clean_text_removes_links_mentions_and_symbols():
    out = clean_text("BREAKING!!! @user says https://t.co/x #Election rigged 2020")
    assert "http" not in out and "user" not in out and "2020" not in out
    assert "elect" in out and "rig" in out


def test_clean_text_keeps_negation():
    assert "not" in clean_text("The senator did not vote for the bill").split()


def test_style_features_counts():
    f = style_features("WOW this is HUGE!!")
    assert f["exclamations"] == 2 and f["all_caps_words"] == 2


# ---------- style model ----------
def test_style_model_labels_and_probabilities(style_model):
    r = style_model.predict("SHOCKING!!! Doctors HATE this secret cure, Obama BANNED it!")
    assert r.label == "Fake" and 0.5 <= r.fake_probability <= 1
    assert r.top_words and all("word" in w for w in r.top_words)


def test_style_model_out_of_scope(style_model):
    assert style_model.predict("qwzx plmk").label == "Out of scope"


# ---------- claims ----------
def test_extract_claims_prefers_factual_sentences():
    text = ("What do you think? The Eiffel Tower is located in Rome and opened in 1952. "
            "I think this is amazing.")
    claims = extract_claims(text)
    assert claims[0].startswith("The Eiffel Tower")
    assert all(not c.endswith("?") for c in claims)


def test_single_headline_is_a_claim():
    assert extract_claims("pope endorses trump") == ["pope endorses trump"]


def test_split_sentences():
    assert len(split_sentences("One fact here. Two facts here! Three?")) == 3


def test_keywords_drop_stopwords():
    assert keywords("The tower is in the city of Rome") == "tower city Rome"


# ---------- verification ----------
def test_rating_to_verdict():
    assert rating_to_verdict("Pants on Fire") == REFUTES
    assert rating_to_verdict("False") == REFUTES
    assert rating_to_verdict("Mostly True") == SUPPORTS
    assert rating_to_verdict("Half True") == MIXED
    assert rating_to_verdict("Explainer") == NEI


def test_nli_refutes_well_written_lie():
    claim = "The Eiffel Tower is located in Rome, Italy."
    ev = [Evidence("wikipedia", "Eiffel Tower", "u", "The Eiffel Tower is a tower in Paris, France.")]
    out = verify_claim(claim, ev, FakeNLI(contradict_if="Paris", support_if="Rome"))
    assert out["verdict"] == REFUTES and out["confidence"] > 0.9


def test_factcheck_rating_used_only_when_claim_matches():
    claim = "Drinking hot water cures coronavirus infection"
    same = Evidence("factcheck", "t", "u", "Hot water cures coronavirus", rating="False")
    other = Evidence("factcheck", "t", "u", "Election results were changed in Ohio", rating="False")
    assert verify_claim(claim, [same])["verdict"] == REFUTES
    assert verify_claim(claim, [other])["verdict"] == NEI


def test_no_evidence_means_not_enough_info():
    assert verify_claim("Some claim about things", [])["verdict"] == NEI
