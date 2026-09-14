#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# build_rest_docker_container.sh
#
# Builds the Docker image for the FastAPI REST inference server that serves
# the EuroSAT MSI classification PyTorch model.
#
# Usage
# -----
#   ./scripts/build_rest_docker_container.sh [OPTIONS]
#
# Options
#   -r, --registry   <registry>   Container registry / Docker Hub username.
#                                 Default: "local"  (image stays on local daemon,
#                                 not pushed to any registry).
#   -t, --tag        <tag>        Image tag.
#                                 Default: "v1"
#   -p, --push                    Push the image to the registry after a
#                                 successful build.  Requires docker login.
#   --no-cache                    Pass --no-cache to docker build.
#   -h, --help                    Show this help message and exit.
#
# Examples
#   # Build locally (no push):
#   ./scripts/build_rest_docker_container.sh
#
#   # Build and push to Docker Hub:
#   ./scripts/build_rest_docker_container.sh --registry johndoe --push
#
#   # Build with a custom tag, no layer cache:
#   ./scripts/build_rest_docker_container.sh --registry johndoe --tag v2 --no-cache --push
# ------------------------------------------------------------------------------

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
REGISTRY="local"
TAG="v1"
PUSH=false
NO_CACHE=""
IMAGE_NAME="eurosat-rest-serve"
DOCKERFILE="rest-serve.dockerfile"

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
usage() {
    sed -n '/^# Usage/,/^# ---/p' "$0" | head -n -1 | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
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
# Derived values
# --------------------------------------------------------------------------
if [[ "$REGISTRY" == "local" ]]; then
    FULL_IMAGE="${IMAGE_NAME}:${TAG}"
else
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
fi

# --------------------------------------------------------------------------
# Resolve repo root from the location of this script.
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --------------------------------------------------------------------------
# Pre-flight checks
# --------------------------------------------------------------------------
echo "============================================================"
echo "  EuroSAT REST inference server image build"
echo "============================================================"
echo "  Repo root   : ${REPO_ROOT}"
echo "  Dockerfile  : ${DOCKERFILE}"
echo "  Image       : ${FULL_IMAGE}"
echo "  Push        : ${PUSH}"
echo "  No-cache    : ${NO_CACHE:-false}"
echo "============================================================"

# Verify docker is available.
if ! command -v docker &> /dev/null; then
    echo "[ERROR] docker is not installed or not on PATH." >&2
    exit 1
fi

# Verify the Dockerfile exists.
if [[ ! -f "${REPO_ROOT}/${DOCKERFILE}" ]]; then
    echo "[ERROR] Dockerfile not found: ${REPO_ROOT}/${DOCKERFILE}" >&2
    exit 1
fi

# Verify requirements file exists (copied into the image by the Dockerfile).
if [[ ! -f "${REPO_ROOT}/requirements-rest-serve.txt" ]]; then
    echo "[ERROR] requirements-rest-serve.txt not found in repo root." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
echo ""
echo "[INFO] Building image: ${FULL_IMAGE} ..."
echo ""

docker build \
    ${NO_CACHE} \
    --file "${REPO_ROOT}/${DOCKERFILE}" \
    --tag  "${FULL_IMAGE}" \
    "${REPO_ROOT}"

echo ""
echo "[INFO] Build succeeded: ${FULL_IMAGE}"

# --------------------------------------------------------------------------
# Optional push
# --------------------------------------------------------------------------
if [[ "${PUSH}" == "true" ]]; then
    if [[ "$REGISTRY" == "local" ]]; then
        echo "[WARN] --push requested but --registry is 'local'. Skipping push." >&2
    else
        echo ""
        echo "[INFO] Pushing image: ${FULL_IMAGE} ..."
        docker push "${FULL_IMAGE}"
        echo "[INFO] Push succeeded: ${FULL_IMAGE}"
    fi
fi

echo ""
echo "[INFO] Done."
echo ""
echo "To run the server locally (checkpoint must exist at the path below):"
echo ""
echo "  docker run --rm --gpus all \\"
echo "    -p 8000:8000 \\"
echo "    -v /full-path-to/checkpoint.pt:/app/model_ckpt/model.pt:ro \\"
echo "    ${FULL_IMAGE}"
echo ""
