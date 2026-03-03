#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NSYS="${NSYS:-/home/smansou2/miniconda/nsight-compute/2024.1.1/host/target-linux-x64/nsys}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIGS=(
  "gege/configs/livejournal.yaml"
  "gege/configs/livejournal_sweep_n512_c8.yaml"
  "gege/configs/livejournal_sweep_n256_c8.yaml"
  "gege/configs/livejournal_sweep_n256_c4.yaml"
)

mkdir -p profiles

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  rep="profiles/${name}_nsys"
  txt="profiles/${name}_nsys_stats.txt"

  echo "=== Profiling ${cfg} -> ${rep}.nsys-rep ==="
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$NSYS" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output "$rep" \
    gege_train "$cfg"

  echo "=== Writing stats -> ${txt} ==="
  "$NSYS" stats \
    --force-export true \
    --report nvtx_sum,osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    "${rep}.nsys-rep" > "$txt"
done

echo "Sweep complete. Reports are in ${ROOT_DIR}/profiles"
