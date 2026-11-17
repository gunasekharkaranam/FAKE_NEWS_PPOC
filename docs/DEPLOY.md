# Deploy

## Option A - Hugging Face Spaces (full version with NLI, free, recommended)

The free CPU Space has 16 GB RAM, enough for the NLI model.

1. Create an account at huggingface.co, then **New Space** → SDK: **Docker** → Blank → Public.
2. In the Space, open **Files** → upload the repo (or link it: `git remote add space https://huggingface.co/spaces/<you>/fake-news-checker` and `git push space master`).
3. The Space's `README.md` must start with this header (add it at the top of the copy you push to the Space):
   ```yaml
   ---
   title: Fake News Checker
   sdk: docker
   app_port: 7860
   ---
   ```
4. Optional: **Settings → Variables and secrets** → add secret `GOOGLE_FACTCHECK_API_KEY`.
5. Wait for the build (~10 min the first time). Put the Space link in your README and resume.

## Option B - Render (light mode, no NLI)

The free plan has 512 MB RAM, too small for the NLI model.
`render.yaml` runs the style model, Google fact-check ratings and the Wikipedia evidence list.

1. In Render: **New → Blueprint** → choose this repo (it reads `render.yaml`).
   If you keep your old Render service instead, change its **Start Command** to:
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
2. Add the `GOOGLE_FACTCHECK_API_KEY` environment variable (optional).

## Option C - Docker anywhere

```bash
docker build -t fakenews .                      # full version
docker build --build-arg WITH_NLI=0 -t fakenews-lite .
docker run -p 7860:7860 -e GOOGLE_FACTCHECK_API_KEY=... fakenews
```

## Getting a Google Fact Check API key (free)

1. Go to console.cloud.google.com and create a project.
2. **APIs & Services → Library** → search "Fact Check Tools API" → Enable.
3. **Credentials → Create credentials → API key**. Restrict it to the Fact Check Tools API.
4. Never commit the key. Put it in `.env` locally or in your host's secrets.
