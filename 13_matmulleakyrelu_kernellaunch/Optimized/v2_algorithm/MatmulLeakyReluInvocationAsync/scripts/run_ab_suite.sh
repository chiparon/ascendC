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
BUILD_DIR="${BUILD_DIR:-/tmp/matmul_v2_ab_build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/tmp/matmul_v2_ab_out}"
MATMUL_ADAPTIVE_MAX_CORE="${MATMUL_ADAPTIVE_MAX_CORE:-4}"
DO_BUILD="${DO_BUILD:-1}"

S3_REPEAT="${S3_REPEAT:-5}"
S1_REPEAT="${S1_REPEAT:-3}"
S2_REPEAT="${S2_REPEAT:-3}"
S4_REPEAT="${S4_REPEAT:-3}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_DIR}/ab_logs_${TS}"
SUMMARY_CSV="${LOG_DIR}/summary.csv"
SUMMARY_MD="${LOG_DIR}/summary.md"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

if [[ "${DO_BUILD}" == "1" ]]; then
    echo "[INFO] build once for ${RUN_MODE}/${SOC_VERSION}"
    bash run.sh -r "${RUN_MODE}" -v "${SOC_VERSION}" -d "${BUILD_DIR}" -p "${INSTALL_PREFIX}" --build-only
fi

echo "shape,group,force_core,repeat,avg_ms,p50_ms,p90_ms,error_ratio,passed,log_file" > "${SUMMARY_CSV}"
cat > "${SUMMARY_MD}" << 'EOF'
| Shape | Group | force-core | repeat | AVG(ms) | P50(ms) | P90(ms) | error ratio | pass/fail | log |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
EOF

run_case() {
    local shape="$1"
    local m="$2"
    local n="$3"
    local k="$4"
    local repeat="$5"
    local group="$6"
    local force_core="$7"
    local tag
    tag="$(echo "${shape}_${group}" | tr '(), ' '____')"
    local log_file="${LOG_DIR}/${tag}.log"

    echo "[INFO] ${shape} group=${group} core=${force_core} repeat=${repeat}"
    MATMUL_ADAPTIVE_MAX_CORE="${MATMUL_ADAPTIVE_MAX_CORE}" \
        bash run.sh -r "${RUN_MODE}" -v "${SOC_VERSION}" \
        -d "${BUILD_DIR}" -p "${INSTALL_PREFIX}" \
        --m "${m}" --n "${n}" --k "${k}" \
        --repeat "${repeat}" --force-core "${force_core}" --run-only \
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

    echo "${shape},${group},${force_core},${repeat},${avg:-NA},${p50:-NA},${p90:-NA},${error_ratio:-NA},${passed},${log_file}" >> "${SUMMARY_CSV}"
    echo "| ${shape} | ${group} | ${force_core} | ${repeat} | ${avg:-NA} | ${p50:-NA} | ${p90:-NA} | ${error_ratio:-NA} | ${passed} | ${log_file} |" >> "${SUMMARY_MD}"
}

# S3: A/B
run_case "S3(1024,512,1024)" 1024 512 1024 "${S3_REPEAT}" "A" 2
run_case "S3(1024,512,1024)" 1024 512 1024 "${S3_REPEAT}" "B" 0

# S1: A/B
run_case "S1(2048,2048,2048)" 2048 2048 2048 "${S1_REPEAT}" "A" 2
run_case "S1(2048,2048,2048)" 2048 2048 2048 "${S1_REPEAT}" "B" 0

# S2: A/B
run_case "S2(4096,1024,4096)" 4096 1024 4096 "${S2_REPEAT}" "A" 2
run_case "S2(4096,1024,4096)" 4096 1024 4096 "${S2_REPEAT}" "B" 0

# S4: B only regression sanity
run_case "S4(512,128,512)" 512 128 512 "${S4_REPEAT}" "B" 0

echo "[INFO] done"
echo "[INFO] summary csv: ${SUMMARY_CSV}"
echo "[INFO] summary md : ${SUMMARY_MD}"
