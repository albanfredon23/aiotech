# syntax=docker/dockerfile:1
# AIOTECH 45 – image de l'API.
#   docker build -t aiotech:45 .                         image d'exécution (sans PyTorch)
#   docker build -t aiotech:45 --build-arg WITH_TORCH=true .   avec le module de recherche
#   docker build --target test .                         lance toute la suite de tests
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

# ── tests : cœur + API (FastAPI) ; les tests PyTorch sont ignorés sans WITH_TORCH ──────
FROM base AS test
ARG WITH_TORCH=false
RUN pip install "pytest>=8.0" \
 && if [ "$WITH_TORCH" = "true" ]; then pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"; fi
COPY . .
RUN AIOTECH_OFFLINE=1 AIOTECH_NO_APP=1 python -m pytest -q -rs

# ── exécution ────────────────────────────────────────────────────────────────────────
FROM base AS runtime
ARG WITH_TORCH=false
RUN if [ "$WITH_TORCH" = "true" ]; then pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"; fi \
 && useradd --create-home --uid 10001 aiotech \
 && mkdir -p /data /app/sandbox \
 && chown aiotech /data /app/sandbox
COPY aiotech ./aiotech
COPY dashboard ./dashboard
USER aiotech
ENV AIOTECH_DB_PATH=/data/aiotech.sqlite3 \
    AIOTECH_TOOLS_DIR=/app/sandbox
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["uvicorn", "aiotech.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
