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
KERNEL_PATTERN="${KERNEL_PATTERN:-matmul|leaky|custom}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_DIR}/tune_logs_${TS}"
SUMMARY_CSV="${LOG_DIR}/summary.csv"
SUMMARY_MD="${LOG_DIR}/summary.md"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

echo "shape,force_core,base_m,base_n,repeat,msprof_repeat,avg_ms,p50_ms,p90_ms,kernel_avg_ms,kernel_p50_ms,kernel_p90_ms,kernel_samples,kernel_source,error_ratio,pass,msprof_dir,log_file" > "${SUMMARY_CSV}"
cat > "${SUMMARY_MD}" << 'MD'
| Shape | core | baseM | baseN | repeat | msprof_repeat | AVG(ms) | P50(ms) | P90(ms) | kernel AVG(ms) | kernel P50(ms) | kernel P90(ms) | kernel samples | error ratio | pass/fail | msprof dir | log |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
MD

if [[ "${DO_BUILD}" == "1" ]]; then
    bash run.sh -d "${BUILD_DIR}" --build-only
fi

run_case() {
    local shape="$1"
    local m="$2"
    local n="$3"
    local k="$4"

    local tag
    tag="$(echo "${shape}" | tr '(), ' '____')"
    local log_file="${LOG_DIR}/${tag}.log"
    local msprof_dir="${LOG_DIR}/msprof_${tag}"

    bash run.sh -d "${BUILD_DIR}" -R \
        -m "${m}" -n "${n}" -k "${k}" -t "${REPEAT}" \
        -P -Q "${MSPROF_REPEAT}" -O "${msprof_dir}" | tee "${log_file}"

    local avg p50 p90 kernel_avg kernel_p50 kernel_p90 kernel_samples kernel_source error_ratio passed
    avg="$(grep -Eo '\[PERF\] AVG_MS=[0-9.]+' "${log_file}" | tail -n1 | cut -d= -f2)"
    p50="$(grep -Eo '\[PERF\] P50_MS=[0-9.]+' "${log_file}" | tail -n1 | cut -d= -f2)"
    p90="$(grep -Eo '\[PERF\] P90_MS=[0-9.]+' "${log_file}" | tail -n1 | cut -d= -f2)"
    local kernel_report_file="${msprof_dir}/kernel_summary.log"
    python3 scripts/extract_msprof_kernel_ms.py --msprof-root "${msprof_dir}" --pattern "${KERNEL_PATTERN}" \
        | tee "${kernel_report_file}" | tee -a "${log_file}" >/dev/null
    kernel_avg="$(grep -Eo '\[KERNEL\] AVG_MS=[^[:space:]]+' "${kernel_report_file}" | tail -n1 | cut -d= -f2)"
    kernel_p50="$(grep -Eo '\[KERNEL\] P50_MS=[^[:space:]]+' "${kernel_report_file}" | tail -n1 | cut -d= -f2)"
    kernel_p90="$(grep -Eo '\[KERNEL\] P90_MS=[^[:space:]]+' "${kernel_report_file}" | tail -n1 | cut -d= -f2)"
    kernel_samples="$(grep -Eo '\[KERNEL\] SAMPLES=[^[:space:]]+' "${kernel_report_file}" | tail -n1 | cut -d= -f2)"
    kernel_source="$(grep -Eo '\[KERNEL\] SOURCE=.*' "${kernel_report_file}" | tail -n1 | cut -d= -f2-)"
    kernel_source="${kernel_source:-NA}"
    if [[ "${kernel_source}" == "NA" ]]; then
        kernel_source="NA"
    else
        kernel_source="${kernel_source//,/;}"
        kernel_source="${kernel_source//\"/\"\"}"
    fi
    error_ratio="$(grep -Eo 'error ratio: [0-9.]+' "${log_file}" | tail -n1 | awk '{print $3}')"
    if grep -q "test pass" "${log_file}"; then
        passed="pass"
    else
        passed="fail"
    fi

    local csv_shape="${shape//\"/\"\"}"
    echo "\"${csv_shape}\",baseline,baseline,baseline,${REPEAT},${MSPROF_REPEAT},${avg:-NA},${p50:-NA},${p90:-NA},${kernel_avg:-NA},${kernel_p50:-NA},${kernel_p90:-NA},${kernel_samples:-0},\"${kernel_source}\",${error_ratio:-NA},${passed},${msprof_dir},${log_file}" >> "${SUMMARY_CSV}"
    echo "| ${shape} | baseline | baseline | baseline | ${REPEAT} | ${MSPROF_REPEAT} | ${avg:-NA} | ${p50:-NA} | ${p90:-NA} | ${kernel_avg:-NA} | ${kernel_p50:-NA} | ${kernel_p90:-NA} | ${kernel_samples:-0} | ${error_ratio:-NA} | ${passed} | ${msprof_dir} | ${log_file} |" >> "${SUMMARY_MD}"
}

run_case "S2(4096,1024,4096)" 4096 1024 4096
run_case "S1(2048,2048,2048)" 2048 2048 2048
run_case "S3(1024,512,1024)" 1024 512 1024

echo "[INFO] done"
echo "[INFO] summary csv: ${SUMMARY_CSV}"
echo "[INFO] summary md : ${SUMMARY_MD}"
