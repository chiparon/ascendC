#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"
BUILD_DIR="build"
MATMUL_M=1024
MATMUL_N=640
MATMUL_K=256
REPEAT=1
FORCE_CORE=0
FORCE_BASE_M=0
FORCE_BASE_N=0
BUILD_ONLY=0
RUN_ONLY=0
CLEAN_BUILD=1
KERNEL_MSPROF=0
MSPROF_REPEAT=1
MSPROF_OUTPUT_DIR=""

SHORT=r:,v:,i:,b:,p:,d:,m:,n:,k:,t:,c:,M:,N:,Q:,O:,B,R,P
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,build-dir:,m:,n:,k:,repeat:,force-core:,force-base-m:,force-base-n:,msprof-repeat:,msprof-output:,build-only,run-only,kernel-msprof,no-clean
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
# Default to 910B3 as the optimization target platform.
SOC_VERSION="${SOC_VERSION:-Ascend910B3}"

while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    -d | --build-dir)
        BUILD_DIR="$2"
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
    -c | --force-core)
        FORCE_CORE="$2"
        shift 2
        ;;
    -M | --force-base-m)
        FORCE_BASE_M="$2"
        shift 2
        ;;
    -N | --force-base-n)
        FORCE_BASE_N="$2"
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
        break
        ;;
    esac
done

RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "[ERROR]: RUN_MODE error, This sample only support specify cpu, sim or npu!"
    exit -1
fi

VERSION_LIST="Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "[ERROR]: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

if [[ "${BUILD_ONLY}" -eq 1 && "${RUN_ONLY}" -eq 1 ]]; then
    echo "[ERROR]: --build-only and --run-only cannot be used together"
    exit -1
fi

if [[ "${KERNEL_MSPROF}" -eq 1 && "${RUN_MODE}" != "npu" ]]; then
    echo "[ERROR]: --kernel-msprof only supports -r npu"
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

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
export MATMUL_M MATMUL_N MATMUL_K
export MATMUL_FORCE_CORE_NUM=${FORCE_CORE}
export MATMUL_FORCE_BASE_M=${FORCE_BASE_M}
export MATMUL_FORCE_BASE_N=${FORCE_BASE_N}
export REPEAT
echo "[INFO]: Current compile soc version is ${SOC_VERSION}"
echo "[INFO]: Shape M=${MATMUL_M}, N=${MATMUL_N}, K=${MATMUL_K}, repeat=${REPEAT}, force_core=${FORCE_CORE}"
echo "[INFO]: force_base_m=${FORCE_BASE_M}, force_base_n=${FORCE_BASE_N}"
echo "[INFO]: Build dir=${BUILD_DIR}, install dir=${INSTALL_PREFIX}"
if [[ "${KERNEL_MSPROF}" -eq 1 ]]; then
    echo "[INFO]: Kernel msprof enabled, msprof_repeat=${MSPROF_REPEAT}, msprof_output=${MSPROF_OUTPUT_DIR:-auto}"
fi
source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
if [ "${RUN_MODE}" = "sim" ]; then
    # in case of running op in simulator, use stub .so instead
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

set -e
cd "${CURRENT_DIR}"
if [[ "${RUN_ONLY}" -ne 1 ]]; then
    if [[ "${CLEAN_BUILD}" -eq 1 ]]; then
        rm -rf "${BUILD_DIR}" "${INSTALL_PREFIX}"
    fi
    mkdir -p "${BUILD_DIR}"
    cmake -S "${CURRENT_DIR}" -B "${BUILD_DIR}" \
        -DRUN_MODE=${RUN_MODE} \
        -DSOC_VERSION=${SOC_VERSION} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
    cmake --build "${BUILD_DIR}" -j
    cmake --install "${BUILD_DIR}"
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
    echo "[INFO]: Build finished. Skip execution because --build-only is set."
    exit 0
fi

rm -f ascendc_kernels_bbit
BIN_PATH="${INSTALL_PREFIX}/bin/ascendc_kernels_bbit"
LIB_PATH_1="${INSTALL_PREFIX}/lib"
LIB_PATH_2="${INSTALL_PREFIX}/lib64"
if [[ ! -x "${BIN_PATH}" ]]; then
    echo "[ERROR]: Missing executable ${BIN_PATH}. Please build first."
    exit -1
fi
cp "${BIN_PATH}" ./
rm -rf input output
mkdir -p input output
python3 scripts/gen_data.py
(
    export LD_LIBRARY_PATH=${LIB_PATH_1}:${LIB_PATH_2}:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
    if [[ "${RUN_WITH_TOOLCHAIN:-0}" -eq 1 ]]; then
        if [ "${RUN_MODE}" = "npu" ]; then
            msprof op --application=./ascendc_kernels_bbit
        elif [ "${RUN_MODE}" = "sim" ]; then
            msprof op simulator --application=./ascendc_kernels_bbit
        elif [ "${RUN_MODE}" = "cpu" ]; then
            ./ascendc_kernels_bbit
        fi
    else
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
    subprocess.run(["./ascendc_kernels_bbit"], check=True)
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
                msprof op --output="${RUN_OUT}" --application=./ascendc_kernels_bbit | tee "${RUN_OUT}/msprof_stdout.log"
            done
            echo "[MSPROF] Done. Please check ${PROF_ROOT}"
        fi
    fi
)
# tidy folder by delete log files
if [ "${RUN_MODE}" = "sim" ]; then
    rm -f *.log *.dump *.vcd *.toml *_log
fi
md5sum output/*.bin
python3 scripts/verify_result.py output/output.bin output/golden.bin
