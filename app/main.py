"""FastAPI web app.

Run locally:   uvicorn app.main:app --reload
Then open:     http://127.0.0.1:8000        (API docs at /docs)
"""
from functools import lru_cache
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from fakenews.article import UnsafeURL, fetch_article
from fakenews.feedback import feedback_stats, save_feedback
from fakenews.pipeline import FakeNewsChecker

ROOT = Path(__file__).resolve().parent


def load_env_file(path: Path = ROOT.parent / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (real env vars win)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if value.strip():
                    os.environ.setdefault(key.strip(), value.strip())


load_env_file()
app = FastAPI(title="Fake News Checker", version="2.0",
              description="Writing-style model + evidence-based fact-checking.")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


@lru_cache(maxsize=1)
def get_checker() -> FakeNewsChecker:
    return FakeNewsChecker()


class CheckRequest(BaseModel):
    text: str | None = Field(None, max_length=20_000)
    url: str | None = Field(None, max_length=2_000)
    use_evidence: bool = True


class FeedbackRequest(BaseModel):
    text: str = Field(..., max_length=20_000)
    verdict: str = Field(..., max_length=100)
    correct: bool


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/check")
def check(req: CheckRequest):
    article = None
    text = (req.text or "").strip()
    if req.url:
        try:
            article = fetch_article(req.url.strip())
        except UnsafeURL as e:
            raise HTTPException(400, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception:
            raise HTTPException(502, "Could not open that link. Paste the article text instead.")
        text = f"{article['title']}. {article['text']}" if article["title"] else article["text"]
    if len(text.split()) < 3:
        raise HTTPException(422, "Please enter at least a full sentence.")
    result = get_checker().check(text, use_evidence=req.use_evidence)
    result["article"] = article and {"url": article["url"], "title": article["title"]}
    return result


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    save_feedback(req.text, req.verdict, req.correct)
    return {"saved": True, **feedback_stats()}


@app.get("/api/model-info")
def model_info():
    path = ROOT.parent / "reports" / "results.json"
    return json.loads(path.read_text()) if path.exists() else {}
