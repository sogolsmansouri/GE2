#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NSYS="${NSYS:-/home/smansou2/miniconda/nsight-compute/2024.1.1/host/target-linux-x64/nsys}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG="${1:-gege/configs/fb15k.yaml}"
TAG="${2:-fb15k}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p profiles

REP_BASE="profiles/${TAG}_${TS}_nsys"
TRAIN_LOG="profiles/${TAG}_${TS}_train.log"
STATS_LOG="profiles/${TAG}_${TS}_nsys_stats.txt"

echo "=== Profiling config: ${CONFIG} ==="
echo "=== Report: ${REP_BASE}.nsys-rep ==="
echo "=== Train log: ${TRAIN_LOG} ==="
echo "=== Stats log: ${STATS_LOG} ==="

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${NSYS}" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output "${REP_BASE}" \
  gege_train "${CONFIG}" 2>&1 | tee "${TRAIN_LOG}"

{
  echo "=== nvtx_sum ==="
  "${NSYS}" stats --force-export true --report nvtx_sum "${REP_BASE}.nsys-rep"
  echo
  echo "=== nvtx_kern_sum:base ==="
  "${NSYS}" stats --force-export true --report nvtx_kern_sum:base "${REP_BASE}.nsys-rep"
  echo
  echo "=== cuda_api_sum ==="
  "${NSYS}" stats --force-export true --report cuda_api_sum "${REP_BASE}.nsys-rep"
  echo
  echo "=== cuda_gpu_kern_sum ==="
  "${NSYS}" stats --force-export true --report cuda_gpu_kern_sum "${REP_BASE}.nsys-rep"
} | tee "${STATS_LOG}"

echo "Done. Logs written to:"
echo "  ${TRAIN_LOG}"
echo "  ${STATS_LOG}"
