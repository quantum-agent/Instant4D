#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/smoke/prepared-scene-smoke.yaml}"
SOURCE_PATH="${SOURCE_PATH:-}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/output/smoke-test}"
INSTANT4D_ENV="${INSTANT4D_ENV:-instant4d}"
GPU_DEVICE="${GPU_DEVICE:-}"
WEBSOCKET_HOST="${WEBSOCKET_HOST:-127.0.0.1}"
WEBSOCKET_PORT="${WEBSOCKET_PORT:-6119}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") --source-path <prepared-scene-dir> [options] [-- extra-args]

Run a lightweight optimization smoke test against a prepared Instant4D scene bundle.

Expected source bundle contents:
- filtered_cvd.npz
- transforms_train.json
- transforms_test.json
- frame images referenced by the transforms JSON

Options:
  --source-path <path>   Prepared scene bundle to optimize (required)
  --model-path <path>    Output directory for the smoke-test run (default: ${MODEL_PATH})
  --config <path>        Smoke-test config (default: ${CONFIG})
  --env <name>           Conda env to use (default: ${INSTANT4D_ENV})
  --gpu <id>             Optional CUDA_VISIBLE_DEVICES value
  --websocket-host <h>   Websocket host for the viewer (default: ${WEBSOCKET_HOST})
  --websocket-port <p>   Websocket port for the viewer (default: ${WEBSOCKET_PORT})
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-path)
      SOURCE_PATH="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --env)
      INSTANT4D_ENV="$2"
      shift 2
      ;;
    --gpu)
      GPU_DEVICE="$2"
      shift 2
      ;;
    --websocket-host)
      WEBSOCKET_HOST="$2"
      shift 2
      ;;
    --websocket-port)
      WEBSOCKET_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOURCE_PATH" ]]; then
  echo "--source-path is required" >&2
  usage >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required for this script unless you inline your own environment handling." >&2
  exit 1
fi

mkdir -p "$MODEL_PATH"

CMD=(python "${REPO_ROOT}/script/optimize.py" \
  --config "$CONFIG" \
  --source_path "$SOURCE_PATH" \
  --model_path "$MODEL_PATH" \
  --test_iterations 250 \
  --save_iterations 250 \
  --websocket_host "$WEBSOCKET_HOST" \
  --websocket_port "$WEBSOCKET_PORT" \
  "${EXTRA_ARGS[@]}")

if [[ -n "$GPU_DEVICE" ]]; then
  conda run --no-capture-output -n "$INSTANT4D_ENV" env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "${CMD[@]}"
else
  conda run --no-capture-output -n "$INSTANT4D_ENV" "${CMD[@]}"
fi
