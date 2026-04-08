# ── Samridhi AI — Production Dockerfile ──────────────────────
# Multi-stage build: keeps image lean by not including build tools
# in the final image.

# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY samridhi/    ./samridhi/
COPY static/      ./static/
COPY server.py    .
COPY config.yaml  .

# ── Data volumes ──────────────────────────────────────────────
# These should be mounted from the host or a volume:
#   - faiss_index/   (required — your FAISS index)
#   - models/        (HuggingFace embedding cache)
#   - documents/     (source documents)
VOLUME ["/app/faiss_index", "/app/models", "/app/documents"]

# ── Environment ───────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# GROQ_API_KEY must be passed at runtime:
#   docker run -e GROQ_API_KEY=your_key ...
# or via docker-compose environment section.

# ── Port ──────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# ── Start ─────────────────────────────────────────────────────
# Single worker — SessionManager is in-memory.
# For multi-worker scale, switch to Redis sessions and increase workers.
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
