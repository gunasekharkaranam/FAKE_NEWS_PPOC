FROM python:3.12-slim

# WITH_NLI=1 installs CPU PyTorch + transformers for the NLI fact-checker (bigger image).
ARG WITH_NLI=1
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 ENABLE_NLI=${WITH_NLI} HF_HOME=/app/.cache/huggingface

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt && \
    if [ "$WITH_NLI" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install "transformers>=4.44"; \
    fi

COPY fakenews ./fakenews
COPY app ./app
COPY models/style_model.joblib ./models/style_model.joblib
COPY reports ./reports

# Download the NLI model at build time so the first request is fast.
RUN if [ "$WITH_NLI" = "1" ]; then python -c "from fakenews.factcheck.verify import LocalNLI; LocalNLI()"; fi

# Run as a normal user, not root.
RUN useradd -m appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
