#!/usr/bin/env bash
#SBATCH --job-name=gege-fb15k-prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail

# Usage:
#   sbatch gege/scripts/arc_profile_fb15k_4gpu.sbatch [method]
# Example:
#   sbatch gege/scripts/arc_profile_fb15k_4gpu.sbatch nsys
#
# Optional environment variables:
#   REPO=/mnt/beegfs/smansou2/GE2/dandelion-dev
#   ENV_DIR=/mnt/beegfs/smansou2/conda_envs/distkge-py38-clean
#   CFG=/mnt/beegfs/smansou2/GE2/dandelion-dev/gege/run_configs/fb15k_16p_4gpu_swap_on_eval.yaml
#   NSYS_BIN=/path/to/nsys
#   NCU_BIN=/path/to/ncu
#   NCU_KERNEL=.*scatter.*

METHOD="${1:-nsys}"
REPO="${REPO:-/mnt/beegfs/smansou2/GE2/dandelion-dev}"
ENV_DIR="${ENV_DIR:-/mnt/beegfs/smansou2/conda_envs/distkge-py38-clean}"
CFG="${CFG:-${REPO}/gege/run_configs/fb15k_16p_4gpu_swap_on_eval.yaml}"
OUT_ROOT="${REPO}/gege/profiles"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="${OUT_ROOT}/fb15k_4gpu_${METHOD}_${JOB_ID}"
BIN="${REPO}/gege/build/gege_train"

mkdir -p "${OUT_ROOT}" "${REPO}/logs"

if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Conda env not found: ${ENV_DIR}" >&2
    exit 1
fi

if [[ ! -f "${CFG}" ]]; then
    echo "Config not found: ${CFG}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"
cd "${REPO}"

# Build only when the binary is missing.
if [[ ! -x "${BIN}" ]]; then
    cmake -S . -B gege/build -DGEGE_ENABLE_PROFILING=ON -DGEGE_PROFILING_BACKEND=perf
    cmake --build gege/build --target gege_train -j
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

cmd=("./gege/scripts/profile_gege.sh" "--method" "${METHOD}" "--output-dir" "${OUT_DIR}")

if [[ "${METHOD}" == "nsys" && -n "${NSYS_BIN:-}" ]]; then
    cmd+=("--nsys-bin" "${NSYS_BIN}")
fi
if [[ "${METHOD}" == "ncu" ]]; then
    if [[ -n "${NCU_BIN:-}" ]]; then
        cmd+=("--ncu-bin" "${NCU_BIN}")
    fi
    if [[ -n "${NCU_KERNEL:-}" ]]; then
        cmd+=("--ncu-kernel" "${NCU_KERNEL}")
    fi
fi

cmd+=("--" "${BIN}" "${CFG}")

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Done. Profiling output: ${OUT_DIR}"
