#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  profile_gege.sh [options] -- <command> [args...]

Options:
  --method <perf|gprof|nsys|ncu> Profiling backend (default: perf)
  --output-dir <path>     Output directory for profiling artifacts
  --freq <hz>             Sampling frequency for perf (default: 199)
  --nsys-trace <list>     nsys trace domains (default: cuda,nvtx,osrt)
  --nsys-bin <path>       nsys executable (default: nsys)
  --ncu-set <name>        ncu metric set (default: speedOfLight)
  --ncu-kernel <regex>    ncu kernel filter regex (optional)
  --ncu-bin <path>        ncu executable (default: ncu)
  --no-gege-timing        Do not set GEGE_PROFILE_TIMING=1
  -h, --help              Show this help message

Examples:
  ./gege/scripts/profile_gege.sh --method perf -- ./gege/build/gege_train gege/configs/fb15k.yaml
  ./gege/scripts/profile_gege.sh --method gprof -- ./gege/build/gege_train gege/configs/fb15k.yaml
  ./gege/scripts/profile_gege.sh --method nsys -- ./gege/build/gege_train gege/configs/fb15k.yaml
  ./gege/scripts/profile_gege.sh --method ncu -- ./gege/build/gege_train gege/configs/fb15k.yaml
EOF
}

method="perf"
output_dir=""
freq="199"
nsys_trace="cuda,nvtx,osrt"
nsys_bin="nsys"
ncu_set="speedOfLight"
ncu_kernel=""
ncu_bin="ncu"
enable_gege_timing="1"

tool_exists() {
    local tool="$1"
    if [[ "${tool}" == */* ]]; then
        [[ -x "${tool}" ]]
    else
        command -v "${tool}" >/dev/null 2>&1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            method="${2:-}"
            shift 2
            ;;
        --output-dir)
            output_dir="${2:-}"
            shift 2
            ;;
        --freq)
            freq="${2:-}"
            shift 2
            ;;
        --nsys-trace)
            nsys_trace="${2:-}"
            shift 2
            ;;
        --nsys-bin)
            nsys_bin="${2:-}"
            shift 2
            ;;
        --ncu-set)
            ncu_set="${2:-}"
            shift 2
            ;;
        --ncu-kernel)
            ncu_kernel="${2:-}"
            shift 2
            ;;
        --ncu-bin)
            ncu_bin="${2:-}"
            shift 2
            ;;
        --no-gege-timing)
            enable_gege_timing="0"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "A command is required after --." >&2
    usage
    exit 1
fi

if [[ "${method}" != "perf" && "${method}" != "gprof" && "${method}" != "nsys" && "${method}" != "ncu" ]]; then
    echo "Invalid --method '${method}'. Expected one of: perf, gprof, nsys, ncu." >&2
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${output_dir}" ]]; then
    output_dir="$(pwd)/gege_profiles/${method}_${timestamp}"
fi
mkdir -p "${output_dir}"

if [[ "${enable_gege_timing}" == "1" ]]; then
    export GEGE_PROFILE_TIMING=1
fi

cmd=("$@")
echo "Profiling command: ${cmd[*]}"
echo "Output directory: ${output_dir}"

if [[ "${method}" == "perf" ]]; then
    if ! command -v perf >/dev/null 2>&1; then
        echo "'perf' is not installed or not available in PATH." >&2
        exit 1
    fi

    perf_data="${output_dir}/perf.data"
    perf_report="${output_dir}/perf.report.txt"
    perf_script="${output_dir}/perf.script.txt"
    perf_folded="${output_dir}/perf.folded.txt"
    perf_flamegraph="${output_dir}/perf.flamegraph.svg"

    if ! perf record -F "${freq}" -g --call-graph dwarf -o "${perf_data}" -- "${cmd[@]}"; then
        cat <<'EOF' >&2
perf record failed.
If this is a permissions issue, check:
  sudo sysctl kernel.perf_event_paranoid=1
  sudo sysctl kernel.kptr_restrict=0
EOF
        exit 1
    fi

    perf report --stdio -i "${perf_data}" > "${perf_report}"
    perf script -i "${perf_data}" > "${perf_script}"

    if command -v stackcollapse-perf.pl >/dev/null 2>&1 && command -v flamegraph.pl >/dev/null 2>&1; then
        stackcollapse-perf.pl "${perf_script}" > "${perf_folded}"
        flamegraph.pl "${perf_folded}" > "${perf_flamegraph}"
    fi

    echo "Generated:"
    echo "  ${perf_data}"
    echo "  ${perf_report}"
    echo "  ${perf_script}"
    if [[ -f "${perf_flamegraph}" ]]; then
        echo "  ${perf_folded}"
        echo "  ${perf_flamegraph}"
    fi
elif [[ "${method}" == "gprof" ]]; then
    if ! command -v gprof >/dev/null 2>&1; then
        echo "'gprof' is not installed or not available in PATH." >&2
        exit 1
    fi

    binary="${cmd[0]}"
    if [[ "${binary}" != /* ]]; then
        resolved_bin="$(command -v "${binary}" || true)"
        if [[ -n "${resolved_bin}" ]]; then
            binary="${resolved_bin}"
        fi
    fi

    if [[ ! -x "${binary}" ]]; then
        echo "Cannot resolve executable for gprof: ${cmd[0]}" >&2
        exit 1
    fi

    gmon_prefix="${output_dir}/gmon"
    gprof_report="${output_dir}/gprof.report.txt"
    GMON_OUT_PREFIX="${gmon_prefix}" "${cmd[@]}"

    gmon_file="$(ls -1t "${gmon_prefix}"* 2>/dev/null | head -n1 || true)"
    if [[ -z "${gmon_file}" ]]; then
        echo "No gmon output found. Rebuild with -DGEGE_ENABLE_PROFILING=ON -DGEGE_PROFILING_BACKEND=gprof." >&2
        exit 1
    fi

    gprof "${binary}" "${gmon_file}" > "${gprof_report}"
    echo "Generated:"
    echo "  ${gmon_file}"
    echo "  ${gprof_report}"
elif [[ "${method}" == "nsys" ]]; then
    if ! tool_exists "${nsys_bin}"; then
        echo "'${nsys_bin}' is not installed or not executable." >&2
        exit 1
    fi
    if ! "${nsys_bin}" --version >/dev/null 2>&1; then
        echo "'${nsys_bin}' was found but is not usable in this environment." >&2
        echo "Install Nsight Systems (or set --nsys-bin to a working binary), then retry." >&2
        exit 1
    fi

    nsys_prefix="${output_dir}/nsys_profile"
    nsys_stats="${output_dir}/nsys.stats.txt"
    "${nsys_bin}" profile --force-overwrite true --trace "${nsys_trace}" -o "${nsys_prefix}" -- "${cmd[@]}"

    nsys_report="$(ls -1 "${nsys_prefix}.nsys-rep" "${nsys_prefix}.qdrep" 2>/dev/null | head -n1 || true)"
    if [[ -n "${nsys_report}" ]]; then
        "${nsys_bin}" stats --report cuda_api_sum,gpu_kern_sum,gpu_mem_time "${nsys_report}" > "${nsys_stats}" || true
    fi

    echo "Generated:"
    ls -1 "${nsys_prefix}".* 2>/dev/null | sed 's/^/  /' || true
    if [[ -f "${nsys_stats}" ]]; then
        echo "  ${nsys_stats}"
    fi
else
    if ! tool_exists "${ncu_bin}"; then
        echo "'${ncu_bin}' is not installed or not executable." >&2
        exit 1
    fi
    if ! "${ncu_bin}" --version >/dev/null 2>&1; then
        echo "'${ncu_bin}' was found but is not usable in this environment." >&2
        echo "Install Nsight Compute (or set --ncu-bin to a working binary), then retry." >&2
        exit 1
    fi

    ncu_prefix="${output_dir}/ncu_profile"
    ncu_report="${ncu_prefix}.ncu-rep"
    ncu_cmd=("${ncu_bin}" --set "${ncu_set}" --target-processes all --force-overwrite --export "${ncu_prefix}")
    if [[ -n "${ncu_kernel}" ]]; then
        ncu_cmd+=(--kernel-name "${ncu_kernel}")
    fi
    ncu_cmd+=(-- "${cmd[@]}")
    "${ncu_cmd[@]}"

    echo "Generated:"
    echo "  ${ncu_report}"
fi
