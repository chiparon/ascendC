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

BUILD_DIR="${BUILD_DIR:-build}"
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

echo "shape,force_core,base_m,base_n,repeat,msprof_repeat,avg_ms,p50_ms,p90_ms,error_ratio,pass,msprof_dir,log_file" > "${SUMMARY_CSV}"
cat > "${SUMMARY_MD}" << 'MD'
| Shape | core | baseM | baseN | repeat | msprof_repeat | AVG(ms) | P50(ms) | P90(ms) | error ratio | pass/fail | msprof dir | log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
MD

if [[ "${DO_BUILD}" == "1" ]]; then
    bash run.sh -d "${BUILD_DIR}" --build-only
fi

run_case() {
    local shape="$1"
    local m="$2"
    local n="$3"
    local k="$4"
    local core="$5"
    local bm="$6"
    local bn="$7"

    local tag
    tag="$(echo "${shape}_c${core}_m${bm}_n${bn}" | tr '(), ' '____')"
    local log_file="${LOG_DIR}/${tag}.log"
    local msprof_dir="${LOG_DIR}/msprof_${tag}"

    MATMUL_FORCE_CORE_NUM="${core}" MATMUL_FORCE_BASE_M="${bm}" MATMUL_FORCE_BASE_N="${bn}" \
        bash run.sh -d "${BUILD_DIR}" -R \
        -m "${m}" -n "${n}" -k "${k}" -t "${REPEAT}" \
        -P -Q "${MSPROF_REPEAT}" -O "${msprof_dir}" | tee "${log_file}"

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

    echo "${shape},${core},${bm},${bn},${REPEAT},${MSPROF_REPEAT},${avg:-NA},${p50:-NA},${p90:-NA},${error_ratio:-NA},${passed},${msprof_dir},${log_file}" >> "${SUMMARY_CSV}"
    echo "| ${shape} | ${core} | ${bm} | ${bn} | ${REPEAT} | ${MSPROF_REPEAT} | ${avg:-NA} | ${p50:-NA} | ${p90:-NA} | ${error_ratio:-NA} | ${passed} | ${msprof_dir} | ${log_file} |" >> "${SUMMARY_MD}"
}

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
