#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCENES_CSV="${SCENES:-panda}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/example}"
DEPTH_DIR="${DEPTH_DIR:-${REPO_ROOT}/SLAM/medium}"
MEGASAM_CKPT="${MEGASAM_CKPT:-${REPO_ROOT}/checkpoints/megasam_final.pth}"
DEPTH_ANYTHING_CKPT="${DEPTH_ANYTHING_CKPT:-${REPO_ROOT}/checkpoints/depth_anything_vitl14.pth}"
RAFT_CKPT="${RAFT_CKPT:-${REPO_ROOT}/checkpoints/raft-things.pth}"
INSTANT4D_ENV="${INSTANT4D_ENV:-instant4d}"
UNIDEPTH_ENV="${UNIDEPTH_ENV:-unidepth}"
GPU_DEVICE="${GPU_DEVICE:-}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra-args-for-camera-tracking]

Run the Mega-SAM / depth / flow / CVD preprocessing stages for one or more scenes.

Options:
  --scenes <csv>                 Comma-separated scene names (default: ${SCENES_CSV})
  --data-dir <path>              Root directory containing per-scene frame folders
  --depth-dir <path>             Root output directory for depth/intermediate outputs
  --megasam-ckpt <path>          Mega-SAM checkpoint path
  --depth-anything-ckpt <path>   Depth Anything checkpoint path
  --raft-ckpt <path>             RAFT checkpoint path
  --instant4d-env <name>         Conda env for most stages (default: ${INSTANT4D_ENV})
  --unidepth-env <name>          Conda env for UniDepth stage (default: ${UNIDEPTH_ENV})
  --gpu <id>                     Optional CUDA_VISIBLE_DEVICES value; omit to inherit current env
  -h, --help                     Show this help

Environment overrides:
  SCENES, DATA_DIR, DEPTH_DIR, MEGASAM_CKPT, DEPTH_ANYTHING_CKPT, RAFT_CKPT,
  INSTANT4D_ENV, UNIDEPTH_ENV, GPU_DEVICE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenes)
      SCENES_CSV="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --depth-dir)
      DEPTH_DIR="$2"
      shift 2
      ;;
    --megasam-ckpt)
      MEGASAM_CKPT="$2"
      shift 2
      ;;
    --depth-anything-ckpt)
      DEPTH_ANYTHING_CKPT="$2"
      shift 2
      ;;
    --raft-ckpt)
      RAFT_CKPT="$2"
      shift 2
      ;;
    --instant4d-env)
      INSTANT4D_ENV="$2"
      shift 2
      ;;
    --unidepth-env)
      UNIDEPTH_ENV="$2"
      shift 2
      ;;
    --gpu)
      GPU_DEVICE="$2"
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

IFS=',' read -r -a SCENES <<< "$SCENES_CSV"

run_in_env() {
  local env_name="$1"
  shift

  if [[ -n "$GPU_DEVICE" ]]; then
    if [[ -n "$env_name" ]]; then
      conda run --no-capture-output -n "$env_name" env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$@"
    else
      env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$@"
    fi
  else
    if [[ -n "$env_name" ]]; then
      conda run --no-capture-output -n "$env_name" "$@"
    else
      "$@"
    fi
  fi
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required for this script unless you inline your own environment handling." >&2
  exit 1
fi

mkdir -p "$DEPTH_DIR/UniDepth" "$DEPTH_DIR/Depth-Anything"

MEGASAM_DIR="${REPO_ROOT}/SLAM/mega-sam"
if [[ ! -d "$MEGASAM_DIR" ]]; then
  echo "Expected Mega-SAM checkout at $MEGASAM_DIR" >&2
  exit 1
fi

for scene in "${SCENES[@]}"; do
  DATA_PATH="${DATA_DIR}/${scene}"
  if [[ ! -d "$DATA_PATH" ]]; then
    echo "Scene directory not found: $DATA_PATH" >&2
    exit 1
  fi

  echo "==> [${scene}] UniDepth"
  (
    cd "$MEGASAM_DIR"
    run_in_env "$UNIDEPTH_ENV" env PYTHONPATH="${PYTHONPATH:-}:$(pwd)/UniDepth" \
      python UniDepth/scripts/demo_mega-sam.py \
      --scene-name "$scene" \
      --img-path "$DATA_PATH" \
      --outdir "$DEPTH_DIR/UniDepth/"
  )

  echo "==> [${scene}] Depth Anything"
  (
    cd "$MEGASAM_DIR"
    run_in_env "$INSTANT4D_ENV" python Depth-Anything/run_videos.py --encoder vitl \
      --load-from "$DEPTH_ANYTHING_CKPT" \
      --img-path "$DATA_PATH/" \
      --outdir "$DEPTH_DIR/Depth-Anything/$scene"
  )

  echo "==> [${scene}] Camera tracking"
  (
    cd "$MEGASAM_DIR"
    run_in_env "$INSTANT4D_ENV" python3 camera_tracking_scripts/test_demo.py \
      --datapath="$DATA_PATH" \
      --weights="$MEGASAM_CKPT" \
      --scene_name "$scene" \
      --mono_depth_path "$DEPTH_DIR/Depth-Anything" \
      --metric_depth_path "$DEPTH_DIR/UniDepth" \
      --disable_vis "${EXTRA_ARGS[@]}"
  )

  echo "==> [${scene}] RAFT optical flow"
  (
    cd "$MEGASAM_DIR"
    run_in_env "$INSTANT4D_ENV" python3 cvd_opt/preprocess_flow.py \
      --datapath="$DATA_PATH" \
      --model="$RAFT_CKPT" \
      --scene_name "$scene" --mixed_precision
  )

  echo "==> [${scene}] CVD optimisation"
  (
    cd "$MEGASAM_DIR"
    run_in_env "$INSTANT4D_ENV" python3 cvd_opt/cvd_opt.py \
      --scene_name "$scene" \
      --w_grad 2.0 --w_normal 5.0
  )
done
