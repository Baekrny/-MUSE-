#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs ckpt

DENSE_CKPT="ckpt/muse_warmup_full_seed2026_dense.ckpt"
SPARSE_CKPT="ckpt/muse_warmup_full_seed2026_sparse.ckpt"

usage() {
  echo "Usage: $0 [validate|warmup|control|staged|all]" >&2
}

validate_data() {
  "$PYTHON_BIN" scripts/validate_dataset.py --root taobao-mm
}

require_warmup_pair() {
  if [[ ! -f "$DENSE_CKPT" || ! -f "$SPARSE_CKPT" ]]; then
    echo "Missing the full-data warm-up checkpoint pair under ./ckpt." >&2
    echo "Run '$0 warmup' first." >&2
    exit 1
  fi
}

run_experiment() {
  local name="$1"
  local config="$2"
  local log_path="logs/${name}_${RUN_ID}.log"

  if [[ -e "$log_path" ]]; then
    echo "Refusing to overwrite existing log: $log_path" >&2
    exit 1
  fi

  echo "Starting $name; log=$log_path"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" main.py \
      --config "$config" 2>&1 | tee "$log_path"
}

run_warmup() {
  if [[ -e "$DENSE_CKPT" || -e "$SPARSE_CKPT" ]]; then
    echo "Refusing to overwrite an existing warm-up checkpoint." >&2
    echo "Expected pair: $DENSE_CKPT and $SPARSE_CKPT" >&2
    exit 1
  fi
  run_experiment "muse_warmup_full_seed2026" \
    "config/muse_warmup_full_seed2026.json"
}

case "$MODE" in
  validate)
    validate_data
    ;;
  warmup)
    validate_data
    run_warmup
    ;;
  control)
    validate_data
    require_warmup_pair
    run_experiment "muse_low_lr_300_full_seed2026" \
      "config/muse_low_lr_300_full_seed2026.json"
    ;;
  staged)
    validate_data
    require_warmup_pair
    run_experiment "global_token_staged_300_full_seed2026" \
      "config/global_token_staged_300_full_seed2026.json"
    ;;
  all)
    validate_data
    if [[ -f "$DENSE_CKPT" && -f "$SPARSE_CKPT" ]]; then
      echo "Reusing the existing warm-up checkpoint pair."
    elif [[ -e "$DENSE_CKPT" || -e "$SPARSE_CKPT" ]]; then
      echo "Only one warm-up checkpoint exists; refusing an inconsistent run." >&2
      exit 1
    else
      run_warmup
    fi
    run_experiment "muse_low_lr_300_full_seed2026" \
      "config/muse_low_lr_300_full_seed2026.json"
    run_experiment "global_token_staged_300_full_seed2026" \
      "config/global_token_staged_300_full_seed2026.json"
    ;;
  *)
    usage
    exit 2
    ;;
esac
