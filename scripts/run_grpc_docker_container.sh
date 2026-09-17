#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_grpc_docker_container.sh
#
# Runs the gRPC inference server Docker container, mounting a local
# model checkpoint into the container.
#
# Usage
# -----
#   ./scripts/run_grpc_docker_container.sh [OPTIONS]
#
# Options
#   -m, --model-path  <path>   Absolute or relative path to the model checkpoint
#                              (.pt file) to mount into the container. Required.
#   -i, --image       <image>  Docker image to run.
#                              Default: "eurosat-grpc-serve:v1"
#   -p, --port        <port>   Host port to map to container port 50051.
#                              Default: 50051
#   --workers         <n>      Number of gRPC thread pool workers.
#                              Default: 4
#   --no-gpu                   Disable GPU passthrough (run on CPU).
#   -h, --help                 Show this help message and exit.
#
# Examples
#   ./scripts/run_grpc_docker_container.sh --model-path ./model_for_prod/reskanet_115.pt
#   ./scripts/run_grpc_docker_container.sh --model-path /abs/path/model.pt --workers 8 --port 50052
# ------------------------------------------------------------------------------

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
MODEL_PATH=""
IMAGE="eurosat-grpc-serve:v1"
HOST_PORT=50051
WORKERS=4
GPU_FLAG="--gpus all"

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Runs the gRPC inference server Docker container."
    echo ""
    echo "Options:"
    echo "  -m, --model-path  <path>   Path to the model checkpoint (.pt file). Required."
    echo "  -i, --image       <image>  Docker image to run (default: eurosat-grpc-serve:v1)."
    echo "  -p, --port        <port>   Host port to map to container port 50051 (default: 50051)."
    echo "  --workers         <n>      Number of gRPC thread pool workers (default: 4)."
    echo "  --no-gpu                   Disable GPU passthrough (run on CPU)."
    echo "  -h, --help                 Show this help message and exit."
    echo ""
    echo "Examples:"
    echo "  $(basename "$0") --model-path ./model_for_prod/reskanet_115.pt"
    echo "  $(basename "$0") --model-path /abs/path/model.pt --workers 8 --port 50052"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        -p|--port)
            HOST_PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --no-gpu)
            GPU_FLAG=""
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
if [[ -z "$MODEL_PATH" ]]; then
    echo "[ERROR] --model-path is required." >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] Model checkpoint not found: ${MODEL_PATH}" >&2
    exit 1
fi

# Resolve to absolute path so docker -v works from any working directory
MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
echo "============================================================"
echo "  Starting EuroSAT gRPC inference server"
echo "============================================================"
echo "  Image       : ${IMAGE}"
echo "  Model path  : ${MODEL_PATH}"
echo "  Host port   : ${HOST_PORT}"
echo "  Workers     : ${WORKERS}"
echo "  GPU         : ${GPU_FLAG:-disabled}"
echo "============================================================"
echo ""

docker run --rm ${GPU_FLAG} \
    -p "${HOST_PORT}:50051" \
    -v "${MODEL_PATH}:/app/model_ckpt/model.pt:ro" \
    "${IMAGE}" \
    --model-path /app/model_ckpt/model.pt \
    --workers "${WORKERS}"
