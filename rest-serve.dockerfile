# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /build

# Install build tools needed to compile any C-extension wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching —
# this layer only rebuilds when requirements-rest-serve.txt changes.
COPY requirements-rest-serve.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements-rest-serve.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy only the source modules needed at serve time
COPY src/data_handler/              ./data_handler/
COPY src/models/                    ./models/
COPY src/loss_func/                 ./loss_func/
COPY src/inference/                 ./inference/
COPY src/inference_service_rest/    ./inference_service_rest/

# Add /app to PYTHONPATH so data_handler, models, inference, etc. are importable
ENV PYTHONPATH=/app

# FastAPI default port
EXPOSE 8000

USER appuser

# Run from inside inference_service_rest/ so the app import string "rest_serve:app"
# resolves correctly when uvicorn re-imports it in each worker process.
WORKDIR /app/inference_service_rest

ENTRYPOINT ["python", "rest_serve.py"]
CMD ["--model-path", "/app/model_ckpt/model.pt", \
     "--host",       "0.0.0.0", \
     "--port",       "8000", \
     "--workers",    "1", \
     "--data-bands", "B", "G", "R", "NIR"]
