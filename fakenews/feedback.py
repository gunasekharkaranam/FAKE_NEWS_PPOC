"""Save user feedback ("was this correct?") so the model can be retrained later."""
from pathlib import Path
import sqlite3
import time

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback.db"


def _conn(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created REAL, text TEXT, verdict TEXT, user_says_correct INTEGER, user_label TEXT)""")
    return con


def save_feedback(text: str, verdict: str, correct: bool, user_label: str = "", path: Path = DB_PATH) -> int:
    with _conn(path) as con:
        cur = con.execute("INSERT INTO feedback (created, text, verdict, user_says_correct, user_label) "
                          "VALUES (?, ?, ?, ?, ?)", (time.time(), text[:5000], verdict, int(correct), user_label))
        return cur.lastrowid


def feedback_stats(path: Path = DB_PATH) -> dict:
    with _conn(path) as con:
        total, correct = con.execute("SELECT COUNT(*), COALESCE(SUM(user_says_correct), 0) FROM feedback").fetchone()
    return {"total": total, "marked_correct": correct}
