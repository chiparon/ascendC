#!/bin/bash
set -e
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

BUILD_DIR="build"
CLEAN_BUILD=1
BUILD_ONLY=0
RUN_ONLY=0
MATMUL_M=1024
MATMUL_N=640
MATMUL_K=256
REPEAT=1
KERNEL_MSPROF=0
MSPROF_REPEAT=1
MSPROF_OUTPUT_DIR=""

SHORT=i:,m:,n:,k:,t:,Q:,O:,d:,B,R,P
LONG=install-path:,m:,n:,k:,repeat:,msprof-repeat:,msprof-output:,build-dir:,build-only,run-only,kernel-msprof,no-clean
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -m | --m)
        MATMUL_M="$2"
        shift 2
        ;;
    -n | --n)
        MATMUL_N="$2"
        shift 2
        ;;
    -k | --k)
        MATMUL_K="$2"
        shift 2
        ;;
    -t | --repeat)
        REPEAT="$2"
        shift 2
        ;;
    -Q | --msprof-repeat)
        MSPROF_REPEAT="$2"
        shift 2
        ;;
    -O | --msprof-output)
        MSPROF_OUTPUT_DIR="$2"
        shift 2
        ;;
    -d | --build-dir)
        BUILD_DIR="$2"
        shift 2
        ;;
    -B | --build-only)
        BUILD_ONLY=1
        shift 1
        ;;
    -R | --run-only)
        RUN_ONLY=1
        shift 1
        ;;
    -P | --kernel-msprof)
        KERNEL_MSPROF=1
        shift 1
        ;;
    --no-clean)
        CLEAN_BUILD=0
        shift 1
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR]: Unexpected option: $1"
        exit -1
        ;;
    esac
done

if [[ "${BUILD_ONLY}" -eq 1 && "${RUN_ONLY}" -eq 1 ]]; then
    echo "[ERROR]: --build-only and --run-only cannot be used together"
    exit -1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
export DDK_PATH=${_ASCEND_INSTALL_PATH}
export NPU_HOST_LIB=${_ASCEND_INSTALL_PATH}/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64
export MATMUL_M MATMUL_N MATMUL_K
export REPEAT

echo "[INFO]: Shape M=${MATMUL_M}, N=${MATMUL_N}, K=${MATMUL_K}, repeat=${REPEAT}"
echo "[INFO]: Build dir=${BUILD_DIR}"
if [[ "${KERNEL_MSPROF}" -eq 1 ]]; then
    echo "[INFO]: Kernel msprof enabled, msprof_repeat=${MSPROF_REPEAT}, msprof_output=${MSPROF_OUTPUT_DIR:-auto}"
fi

cd ${CURRENT_DIR}

if [[ "${RUN_ONLY}" -ne 1 ]]; then
    rm -rf $HOME/ascend/log/*
    rm -f ./input/*.bin ./output/*.bin

    python3 scripts/gen_data.py

    if [[ "${CLEAN_BUILD}" -eq 1 ]]; then
        rm -rf "${BUILD_DIR}"
    fi
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake ../src -DCMAKE_SKIP_RPATH=TRUE
    make -j
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
    echo "[INFO]: Build finished. Skip execution because --build-only is set."
    exit 0
fi

export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
cd ${CURRENT_DIR}/output
if [[ ! -x "./execute_matmul_leakyrelu_op" ]]; then
    echo "[ERROR]: Missing executable ./execute_matmul_leakyrelu_op. Please build first."
    exit -1
fi

python3 - << 'PY'
import math
import os
import subprocess
import time

repeat = int(os.getenv("REPEAT", "1"))
if repeat < 1:
    repeat = 1

times = []
for _ in range(repeat):
    t0 = time.perf_counter()
    subprocess.run(["./execute_matmul_leakyrelu_op"], check=True)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000.0)

times_sorted = sorted(times)

def percentile(vals, p):
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] + (vals[c] - vals[f]) * (k - f)

avg = sum(times) / len(times)
p50 = percentile(times_sorted, 0.50)
p90 = percentile(times_sorted, 0.90)

print("[PERF] TIMES_MS=" + ",".join(f"{t:.3f}" for t in times))
print(f"[PERF] AVG_MS={avg:.3f}")
print(f"[PERF] P50_MS={p50:.3f}")
print(f"[PERF] P90_MS={p90:.3f}")
PY

if [[ "${KERNEL_MSPROF}" -eq 1 ]]; then
    if ! command -v msprof >/dev/null 2>&1; then
        echo "[ERROR]: msprof not found in PATH."
        exit -1
    fi
    if [[ "${MSPROF_REPEAT}" -lt 1 ]]; then
        echo "[ERROR]: --msprof-repeat must be >= 1"
        exit -1
    fi
    TS=$(date +%Y%m%d_%H%M%S)
    PROF_ROOT="${MSPROF_OUTPUT_DIR:-${CURRENT_DIR}/msprof_kernel_${TS}}"
    mkdir -p "${PROF_ROOT}"
    echo "[MSPROF] Start kernel profiling. output_root=${PROF_ROOT}"
    for ((i = 1; i <= MSPROF_REPEAT; i++)); do
        RUN_OUT="${PROF_ROOT}/run_${i}"
        mkdir -p "${RUN_OUT}"
        echo "[MSPROF] run ${i}/${MSPROF_REPEAT}"
        msprof op --output="${RUN_OUT}" --application=./execute_matmul_leakyrelu_op | tee "${RUN_OUT}/msprof_stdout.log"
    done
    echo "[MSPROF] Done. Please check ${PROF_ROOT}"
fi

cd ${CURRENT_DIR}
python3 scripts/verify_result.py output/output_z.bin output/golden.bin
