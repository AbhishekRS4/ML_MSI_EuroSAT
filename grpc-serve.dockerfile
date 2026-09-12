# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /build

# Install build tools needed to compile any C-extension wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching —
# this layer only rebuilds when requirements-grpc-serve.txt changes.
COPY requirements-grpc-serve.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements-grpc-serve.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy only the source modules needed at serve time
COPY src/data_handler/           ./data_handler/
COPY src/models/                 ./models/
COPY src/loss_func/              ./loss_func/
COPY src/inference_service_grpc/ ./inference_service_grpc/

# Add /app to PYTHONPATH so data_handler, models, etc. are importable
ENV PYTHONPATH=/app

# gRPC default port
EXPOSE 50051

USER appuser

# Run from inside inference_service_grpc/ so flat imports (inference_pb2, etc.) resolve
WORKDIR /app/inference_service_grpc

ENTRYPOINT ["python", "grpc_serve.py"]
CMD ["--model-path", "/app/model_ckpt/model.pt", \
     "--port",       "50051", \
     "--workers",    "4", \
     "--data-bands", "B", "G", "R", "NIR"]
