#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)
PROJECT_DIR=$(
    cd "${SCRIPT_DIR}/.."
    pwd
)

RUN_MODE="${RUN_MODE:-npu}"
SOC_VERSION="${SOC_VERSION:-Ascend910B3}"
BUILD_DIR="${BUILD_DIR:-/tmp/matmul_v2_tune_build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/tmp/matmul_v2_tune_out}"
DO_BUILD="${DO_BUILD:-1}"
REPEAT="${REPEAT:-1}"
MSPROF_REPEAT="${MSPROF_REPEAT:-1}"
FORCE_CORE_LIST="${FORCE_CORE_LIST:-2 4}"
BASE_M_LIST="${BASE_M_LIST:-128 256}"
BASE_N_LIST="${BASE_N_LIST:-128 256}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_DIR}/tune_logs_${TS}"
SUMMARY_CSV="${LOG_DIR}/summary.csv"
SUMMARY_MD="${LOG_DIR}/summary.md"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

if [[ "${RUN_MODE}" != "npu" ]]; then
    echo "[ERROR] run_kernel_tune.sh requires RUN_MODE=npu to use msprof."
    exit 1
fi

if [[ "${DO_BUILD}" == "1" ]]; then
    echo "[INFO] build once for ${RUN_MODE}/${SOC_VERSION}"
    bash run.sh -r "${RUN_MODE}" -v "${SOC_VERSION}" -d "${BUILD_DIR}" -p "${INSTALL_PREFIX}" --build-only
fi

echo "shape,force_core,base_m,base_n,repeat,msprof_repeat,avg_ms,p50_ms,p90_ms,error_ratio,pass,msprof_dir,log_file" > "${SUMMARY_CSV}"
cat > "${SUMMARY_MD}" << 'EOF'
| Shape | core | baseM | baseN | repeat | msprof_repeat | AVG(ms) | P50(ms) | P90(ms) | error ratio | pass/fail | msprof dir | log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
EOF

run_case() {
    local shape="$1"
    local m="$2"
    local n="$3"
    local k="$4"
    local force_core="$5"
    local base_m="$6"
    local base_n="$7"

    local tag
    tag="$(echo "${shape}_c${force_core}_m${base_m}_n${base_n}" | tr '(), ' '____')"
    local log_file="${LOG_DIR}/${tag}.log"
    local msprof_dir="${LOG_DIR}/msprof_${tag}"

    echo "[INFO] ${shape} core=${force_core} baseM=${base_m} baseN=${base_n}"
    bash run.sh -r "${RUN_MODE}" -v "${SOC_VERSION}" \
        -d "${BUILD_DIR}" -p "${INSTALL_PREFIX}" \
        --m "${m}" --n "${n}" --k "${k}" \
        --repeat "${REPEAT}" --force-core "${force_core}" \
        --force-base-m "${base_m}" --force-base-n "${base_n}" \
        --kernel-msprof --msprof-repeat "${MSPROF_REPEAT}" --msprof-output "${msprof_dir}" \
        --run-only \
        | tee "${log_file}"

    local avg p50 p90 error_ratio passed
    avg="$(grep -Eo '\[PERF\] AVG_MS=[0-9.]+' "${log_file}" | tail -n1 | cut -d= -f2)"
    p50="$(grep -Eo '\[PERF\] P50_MS=[0-9.]+' "${log_file}" | tail -n1 | cut -d= -f2)"
    p90="$(grep -Eo '\[PERF\] P90_MS=[0-9.]+' "${log_file}" | tail -n1 | cut -d= -f2)"
    error_ratio="$(grep -Eo 'error ratio: [0-9.]+' "${log_file}" | tail -n1 | awk '{print $3}')"
    if grep -q "test pass" "${log_file}"; then
        passed="pass"
    else
        passed="fail"
    fi

    echo "${shape},${force_core},${base_m},${base_n},${REPEAT},${MSPROF_REPEAT},${avg:-NA},${p50:-NA},${p90:-NA},${error_ratio:-NA},${passed},${msprof_dir},${log_file}" >> "${SUMMARY_CSV}"
    echo "| ${shape} | ${force_core} | ${base_m} | ${base_n} | ${REPEAT} | ${MSPROF_REPEAT} | ${avg:-NA} | ${p50:-NA} | ${p90:-NA} | ${error_ratio:-NA} | ${passed} | ${msprof_dir} | ${log_file} |" >> "${SUMMARY_MD}"
}

# High-priority shapes first (M2 gating targets).
for core in ${FORCE_CORE_LIST}; do
    for bm in ${BASE_M_LIST}; do
        for bn in ${BASE_N_LIST}; do
            run_case "S2(4096,1024,4096)" 4096 1024 4096 "${core}" "${bm}" "${bn}"
            run_case "S1(2048,2048,2048)" 2048 2048 2048 "${core}" "${bm}" "${bn}"
            run_case "S3(1024,512,1024)" 1024 512 1024 "${core}" "${bm}" "${bn}"
        done
    done
done

echo "[INFO] done"
echo "[INFO] summary csv: ${SUMMARY_CSV}"
echo "[INFO] summary md : ${SUMMARY_MD}"
