#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_triton_docker_service.sh
#
# Starts the NVIDIA Triton Inference Server docker container, mounting a
# local model repository directory into the container.
#
# Usage
# -----
#   ./scripts/run_triton_docker_service.sh [OPTIONS]
#
# Options
#   -m, --model-repo  <path>   Absolute path to the local model repository
#                              directory to mount into the container.
#                              Required.
#   --shm-size        <size>   Shared memory size (default: 1g).
#   -h, --help                 Show this help message and exit.
#
# Examples
#   ./scripts/run_triton_docker_service.sh --model-repo /path/to/model_repository
#   ./scripts/run_triton_docker_service.sh --model-repo /path/to/model_repository --shm-size 4g
# ------------------------------------------------------------------------------

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
MODEL_REPO=""
SHM_SIZE="1g"
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.01-py3"

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Starts the NVIDIA Triton Inference Server docker container, mounting a"
    echo "local model repository directory into the container."
    echo ""
    echo "Options:"
    echo "  -m, --model-repo  <path>   Absolute or relative path to the local model"
    echo "                             repository directory to mount. Required."
    echo "  --shm-size        <size>   Shared memory size (default: 1g)."
    echo "  -h, --help                 Show this help message and exit."
    echo ""
    echo "Examples:"
    echo "  $(basename "$0") --model-repo ./model_for_triton"
    echo "  $(basename "$0") --model-repo /path/to/model_repository --shm-size 4g"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model-repo)
            MODEL_REPO="$2"
            shift 2
            ;;
        --shm-size)
            SHM_SIZE="$2"
            shift 2
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
if [[ -z "$MODEL_REPO" ]]; then
    echo "[ERROR] --model-repo is required." >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

if [[ ! -d "$MODEL_REPO" ]]; then
    echo "[ERROR] Model repository directory not found: ${MODEL_REPO}" >&2
    exit 1
fi

# Resolve to absolute path
MODEL_REPO="$(cd "${MODEL_REPO}" && pwd)"

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
echo "============================================================"
echo "  Starting Triton Inference Server"
echo "============================================================"
echo "  Model repo  : ${MODEL_REPO}"
echo "  SHM size    : ${SHM_SIZE}"
echo "  Image       : ${TRITON_IMAGE}"
echo "  Ports       : 8000 (HTTP), 8001 (gRPC), 8002 (metrics)"
echo "============================================================"
echo ""

docker run --gpus all --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    --shm-size="${SHM_SIZE}" \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${MODEL_REPO}:/models" \
    "${TRITON_IMAGE}" \
    tritonserver --model-repository=/models
