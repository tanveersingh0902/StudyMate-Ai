# StudyMate AI — production image (Streamlit Cloud uses requirements.txt directly;
# this Dockerfile is for Render / Railway / Hugging Face Spaces Docker / self-host).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false

WORKDIR /app

# System deps for faiss + pdf/text handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py ./
COPY src ./src
COPY .streamlit ./.streamlit

# Pre-warm HuggingFace cache dir (model downloads on first query otherwise)
ENV HF_HOME=/app/.hf_cache
RUN mkdir -p /app/.hf_cache /app/vector_store

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
