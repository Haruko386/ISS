#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPUS="${GPUS:-0,1}"
CONFIG="configs/sd15.yaml"
DATA=""
OUTPUT=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash script/train_multi_gpu.sh --gpus 0,3 [options] [-- extra train arguments]

Options:
  --gpus IDS       Physical GPU IDs, comma-separated (default: 0,1 or $GPUS)
  --config PATH    Training YAML (default: configs/sd15.yaml)
  --data PATH      Prepared dataset root (optional; otherwise use YAML value)
  --output PATH    Output directory (optional; otherwise use YAML value)
  -h, --help       Show this help

Examples:
  bash script/train_multi_gpu.sh --gpus 0,3 --data dataset/prepared/panoramas
  bash script/train_multi_gpu.sh --gpus 0,1,3 --output outputs/sd15-3gpu -- --steps 20000
EOF
}

while (($#)); do
  case "$1" in
    --gpus)
      [[ $# -ge 2 ]] || { echo "error: --gpus requires a value" >&2; exit 2; }
      GPUS="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "error: --config requires a value" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --data)
      [[ $# -ge 2 ]] || { echo "error: --data requires a value" >&2; exit 2; }
      DATA="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "error: --output requires a value" >&2; exit 2; }
      OUTPUT="$2"
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
      echo "error: unknown option '$1' (put iss train options after --)" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Spaces are harmless for human input but invalid in CUDA_VISIBLE_DEVICES.
GPUS="${GPUS//[[:space:]]/}"
if [[ ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "error: --gpus must be comma-separated non-negative GPU IDs, e.g. 0,3" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "error: duplicate GPU ID '$gpu'" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done

TRAIN_ARGS=(--config "$CONFIG" --device cuda)
[[ -z "$DATA" ]] || TRAIN_ARGS+=(--data "$DATA")
[[ -z "$OUTPUT" ]] || TRAIN_ARGS+=(--output "$OUTPUT")
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

echo "Launching ISS on physical GPUs: $GPUS (${#GPU_IDS[@]} processes)"
CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${#GPU_IDS[@]}" \
  main.py train "${TRAIN_ARGS[@]}"
