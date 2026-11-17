# Learn this project step by step

Go in this order. Each step has: **what to read**, **what to run**, and **questions to answer yourself**.
If you can answer the questions without looking, you are ready for interviews.

---

## Step 1 - The data (1 hour)

**Read:** `fakenews/data.py`

**Run:**
```bash
python -c "from fakenews.data import download_all, load_splits; download_all(); tr, te = load_splits('liar'); print(tr.head(10)); print(tr.label.value_counts())"
```

**Understand:**
- LIAR has 6 ratings. We turn them into 2 classes: pants-fire, false, barely-true = fake (1); the rest = real (0).
- FakeNewsNet gives headlines from PolitiFact (politics) and GossipCop (celebrity news).
- We use one rule everywhere: **1 = fake, 0 = real**. (Version 1 mixed this up.)

**Ask yourself:** Why is "barely-true" put in fake? What happens if you move it to real?

---

## Step 2 - Text cleaning (30 min)

**Read:** `fakenews/preprocess.py`

**Run:**
```bash
python -c "from fakenews.preprocess import clean_text; print(clean_text('BREAKING!!! The senator did NOT vote https://t.co/x #Election'))"
```

**Understand:**
- Lowercase, remove links and @mentions, keep only letters, remove stop words, stem ("voting" becomes "vote").
- We **keep "not", "no", "never"**, because "did not vote" and "did vote" mean opposite things.
- Training and the app use **the same function**, so they never clean text differently.

**Ask yourself:** What is stemming? What could go wrong if the app cleaned text differently from training?

---

## Step 3 - The style model (1-2 hours)

**Read:** `fakenews/style_model.py`, then `scripts/train_style_model.py`

**Run:**
```bash
python -m scripts.train_style_model
cat reports/results.md
```

**Understand:**
- **TF-IDF** gives each word a score: high if it is common in this text but rare in general. Bigrams (2-word phrases) are included.
- **Logistic Regression** learns one weight per word. A positive weight pushes toward fake.
- **Explanation:** for a linear model, a word's contribution = its tf-idf value × its weight. This is exact, so we don't need LIME here.
- `class_weight="balanced"`: GossipCop is only 24% fake, so mistakes on fake count more.
- **GridSearchCV** tries C = 0.3, 1, 3 with 3-fold cross-validation.
- **Macro-F1** = the average of F1 for fake and F1 for real. It is fair when classes are unbalanced. Accuracy can look good just by always saying "real".

**Ask yourself:** Why does the cross-dataset table drop to about 0.5? (Answer: each source has its own style and topics; the model learns those, not truth.)

---

## Step 4 - Why style is not enough (30 min)

**Run:**
```bash
python -m scripts.well_written_lie_test
```

**Understand:** 25 calm sentences, 52% accuracy, a coin flip. "The Eiffel Tower is in Rome" has no "fake-looking" words. This is the interviewer's question, answered with numbers.

---

## Step 5 - Finding claims (30 min)

**Read:** `fakenews/factcheck/claims.py`

**Understand:** We score each sentence: numbers +1, capitalised names +0.5 each, "is/was/has" +0.5, opinion words -1.5, questions = 0. Then we keep the top 3.

**Ask yourself:** Why skip questions and opinions? (They cannot be true or false.)

---

## Step 6 - Finding evidence (1 hour)

**Read:** `fakenews/factcheck/retrieve.py`

**Understand:**
- **Google Fact Check Tools API**: returns claims that real fact-checkers already rated ("False", "Mostly True").
- **Wikipedia API**: search with the claim's keywords, then pick the 2 sentences that share the most words with the claim.
- Every source returns the same `Evidence` object, so adding a news API later is one new function.
- If a source fails (no internet, bad key), it is reported in `source_errors` and the app keeps working.

---

## Step 7 - Verifying with NLI (1-2 hours, most important)

**Read:** `fakenews/factcheck/verify.py`

**Understand:**
- **NLI (Natural Language Inference)**: given a *premise* (evidence) and a *hypothesis* (claim), the model outputs **entailment** (supports), **neutral** or **contradiction** (refutes).
- Model: `DeBERTa-v3-base-mnli-fever-anli`, trained on MNLI, **FEVER** (a real fact-verification dataset) and ANLI.
- Premise = "The Eiffel Tower is in Paris." Hypothesis = "The Eiffel Tower is in Rome." Output: contradiction, so the claim is refuted. Writing style plays no part.
- **Fact-checker ratings** are used first, but only if their claim shares at least 30% of its key words with ours. Otherwise a "False" rating about a different story could wrongly decide our claim.

**Run (in Colab):** `notebooks/02_factcheck_evaluation.ipynb`

---

## Step 8 - Putting it together (30 min)

**Read:** `fakenews/pipeline.py`

**Rule:** refuted → Likely fake. All supported → Likely real. Otherwise → Unverified + the style guess, clearly labelled as a guess.

**Ask yourself:** Why not average style and evidence? (Evidence is about truth; style is a weak hint. Averaging would let style override facts.)

---

## Step 9 - The web app (1 hour)

**Read:** `app/main.py`, `app/static/app.js`

**Run:** `uvicorn app.main:app --reload`, then open http://127.0.0.1:8000 and http://127.0.0.1:8000/docs

**Understand:**
- FastAPI checks inputs with Pydantic (max length, required fields).
- **Security 1:** the page inserts server text with `textContent`, never `innerHTML`, so a malicious article can't run JavaScript (XSS).
- **Security 2:** `article.py` blocks links to private addresses like `127.0.0.1` (SSRF), and limits redirects, size and time.
- Feedback is saved in SQLite for future retraining.

---

## Step 10 - DistilBERT (2 hours, in Colab)

**Read:** `scripts/train_transformer.py`. **Run:** `notebooks/01_distilbert_training.ipynb` on a T4 GPU.

**Understand:**
- Same splits and same metric as TF-IDF, so the comparison is fair.
- Weighted loss = the same idea as `class_weight="balanced"`.
- Expect better same-dataset scores, but cross-dataset will still be weak. A better style model is still a style model.

---

## Step 11 - Engineering (1 hour)

**Read:** `tests/`, `Dockerfile`, `.github/workflows/ci.yml`, `render.yaml`

**Understand:**
- Tests use a **fake NLI** and **fake sources**, so they run in 2 seconds without internet.
- GitHub Actions runs the tests on every push.
- The Docker image downloads the NLI model at build time, so the first request is fast.

**Run:** `pytest -q`
