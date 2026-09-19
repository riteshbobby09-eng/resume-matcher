# Production Deployment Guide

This guide describes how to deploy Resume Matcher as a high-throughput, production-grade microservice.

---

## 1. Local / Dedicated Server Deployment

### System Prerequisites
- **Python:** 3.10, 3.11, 3.12, or 3.13
- **RAM:** Minimum 2 GB (4 GB recommended for high concurrent loads)
- **CPU:** 2+ cores (BGE model parallelizes efficiently on multi-core CPUs)

### Installation & Startup

```bash
# 1. Clone repository and set up virtualenv
git clone <repo-url>
cd resume-matcher
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -e .

# 3. Pre-download embedding weights (avoids cold start in production)
python -m scripts.download_model

# 4. Start production Uvicorn server with multiple workers
uvicorn resume_matcher.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --access-log
```

---

## 2. Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Copy application source code
COPY src/ ./src/
COPY config/ ./config/
COPY static/ ./static/
COPY scripts/ ./scripts/

# Pre-download model into container image
RUN python -m scripts.download_model

EXPOSE 8000

CMD ["uvicorn", "resume_matcher.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### Build & Run Container

```bash
docker build -t resume-matcher:latest .
docker run -d -p 8000:8000 --name resume-matcher resume-matcher:latest
```

---

## 3. Production Health & Observability

- **Liveness / Readiness Probe:** `GET /health`
  - Returns `{"status": "ok", "version": "0.1.0"}`
- **Interactive UI for HR:** `GET /`
- **Matching Endpoint:** `POST /match` (multipart/form-data: `jd_file`, `resume_file`)
