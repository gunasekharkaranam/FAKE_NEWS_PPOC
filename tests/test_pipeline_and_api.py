import pytest
from fastapi.testclient import TestClient

from fakenews.article import UnsafeURL, check_url, extract_text
from fakenews.factcheck.retrieve import Evidence, search_wikipedia
from fakenews.feedback import feedback_stats, save_feedback
from fakenews.pipeline import FakeNewsChecker

from conftest import FakeNLI

PARIS = Evidence("wikipedia", "Eiffel Tower", "https://en.wikipedia.org/wiki/Eiffel_Tower",
                 "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.")


@pytest.fixture
def checker(style_model):
    return FakeNewsChecker(style_model, nli=FakeNLI("Paris", "Himalayas"),
                           sources={"stub": lambda claim: [PARIS] if "Eiffel" in claim else []})


def test_calm_lie_is_caught_by_evidence_not_style(checker):
    r = checker.check("The Eiffel Tower is located in Rome, Italy, and was completed in 1952.")
    assert r["verdict"] == "Likely fake" and r["decided_by"] == "evidence"


def test_falls_back_to_style_when_no_evidence(checker):
    r = checker.check("SHOCKING!!! Doctors HATE this secret cure the government BANNED!")
    assert r["decided_by"] == "style" and "style looks fake" in r["verdict"]


def test_failing_source_is_reported_not_crashing(style_model):
    def broken(claim):
        raise TimeoutError
    c = FakeNewsChecker(style_model, nli=None, sources={"broken": broken})
    r = c.check("The moon is made of cheese according to NASA scientists.")
    assert r["source_errors"] == ["broken: TimeoutError"]


def test_wikipedia_parsing_with_fake_http():
    class Resp:
        def __init__(self, data): self.data = data
        def raise_for_status(self): pass
        def json(self): return self.data

    class Session:
        def get(self, url, params, **kw):
            if params.get("list") == "search":
                return Resp({"query": {"search": [{"title": "Eiffel Tower"}]}})
            return Resp({"query": {"pages": {"1": {"title": "Eiffel Tower", "extract":
                "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is named after Gustave Eiffel."}}}})

    ev = search_wikipedia("Eiffel Tower located in Rome", session=Session())
    assert ev[0].title == "Eiffel Tower" and "Paris" in ev[0].text


# ---------- link reading safety ----------
@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1/admin", "http://localhost:8000"])
def test_unsafe_urls_blocked(url):
    with pytest.raises(UnsafeURL):
        check_url(url)


def test_extract_text_keeps_paragraphs():
    html = ("<html><title>T</title><body><nav>menu menu</nav><h1>Big News</h1>"
            "<p>This is a real paragraph of the article with enough words in it.</p>"
            "<p>short</p><script>evil()</script></body></html>")
    title, text = extract_text(html)
    assert title == "Big News" and "real paragraph" in text and "evil" not in text and "short" not in text


# ---------- feedback ----------
def test_feedback_saved(tmp_path):
    db = tmp_path / "fb.db"
    save_feedback("some text", "Likely fake", True, path=db)
    save_feedback("other", "Likely real", False, path=db)
    assert feedback_stats(db) == {"total": 2, "marked_correct": 1}


# ---------- API ----------
@pytest.fixture
def client(checker, monkeypatch, tmp_path):
    import app.main as m
    import fakenews.feedback as fb
    monkeypatch.setattr(m, "get_checker", lambda: checker)
    monkeypatch.setattr(fb, "DB_PATH", tmp_path / "fb.db")
    monkeypatch.setattr(m, "save_feedback", lambda *a: fb.save_feedback(*a, path=tmp_path / "fb.db"))
    monkeypatch.setattr(m, "feedback_stats", lambda: fb.feedback_stats(tmp_path / "fb.db"))
    return TestClient(m.app)


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Fake News Checker" in r.text


def test_api_check(client):
    r = client.post("/api/check", json={"text": "The Eiffel Tower is located in Rome, Italy."})
    assert r.status_code == 200 and r.json()["verdict"] == "Likely fake"


def test_api_rejects_empty_and_unsafe(client):
    assert client.post("/api/check", json={"text": "hi"}).status_code == 422
    assert client.post("/api/check", json={"url": "http://127.0.0.1/"}).status_code == 400


def test_api_feedback(client):
    r = client.post("/api/feedback", json={"text": "x y z", "verdict": "Likely fake", "correct": True})
    assert r.json()["total"] == 1
