#!/bin/bash
SHORT=v:,i:,
LONG=soc-version:,install-path:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
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

VERSION_LIST="Ascend910A Ascend910B Ascend310B1 Ascend310B2 Ascend310B3 Ascend310B4 Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"

detect_soc_version() {
    local detected
    detected=$(npu-smi info 2>/dev/null | grep -Eo '910B[0-9]*|910A|310P[0-9]*|310B[0-9]*' | head -n 1)
    if [ -n "$detected" ]; then
        SOC_VERSION="Ascend${detected}"
        echo "[INFO]: Auto detected SOC_VERSION=${SOC_VERSION}"
    fi
}

normalize_soc_version() {
    local raw_soc="$1"
    if [ -z "$raw_soc" ]; then
        echo ""
        return
    fi
    raw_soc="${raw_soc#Ascend}"
    echo "Ascend${raw_soc}"
}

resolve_install_path() {
    local candidates=(
        "$ASCEND_INSTALL_PATH"
        "$ASCEND_HOME_PATH"
        "$ASCEND_TOOLKIT_HOME"
        "$ASCEND_CANN_PACKAGE_PATH"
        "$HOME/Ascend/ascend-toolkit/latest"
        "/usr/local/Ascend/ascend-toolkit/latest"
        "/usr/local/Ascend/latest"
    )
    local candidate=""
    for candidate in "${candidates[@]}"; do
        if [ -n "$candidate" ] && [ -f "$candidate/bin/setenv.bash" ]; then
            echo "$candidate"
            return
        fi
    done
    echo ""
}

map_compute_unit() {
    case "$1" in
        Ascend910B|Ascend910B1|Ascend910B2|Ascend910B3|Ascend910B4) echo "ascend910b" ;;
        Ascend910A) echo "ascend910" ;;
        Ascend310P1|Ascend310P3) echo "ascend310p" ;;
        Ascend310B1|Ascend310B2|Ascend310B3|Ascend310B4) echo "ascend310b" ;;
        *) echo "$(echo "$1" | tr '[:upper:]' '[:lower:]')" ;;
    esac
}

SOC_VERSION=$(normalize_soc_version "$SOC_VERSION")
if [ -z "$SOC_VERSION" ]; then
    detect_soc_version
fi

if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "[ERROR]: SOC_VERSION should be in [$VERSION_LIST], current is [${SOC_VERSION}]"
    exit -1
fi

_ASCEND_INSTALL_PATH=$(resolve_install_path)
if [ -z "$_ASCEND_INSTALL_PATH" ]; then
    echo "[ERROR]: Cannot resolve valid CANN install path, please set --install-path or ASCEND_INSTALL_PATH."
    exit -1
fi
source "$_ASCEND_INSTALL_PATH/bin/setenv.bash"
export ASCEND_HOME_PATH="$_ASCEND_INSTALL_PATH"
export ASCEND_INSTALL_PATH="$_ASCEND_INSTALL_PATH"
export ASCEND_COMPUTE_UNIT=$(map_compute_unit "$SOC_VERSION")
echo "[INFO]: Use ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}"
echo "[INFO]: Use ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}"

OP_NAME=MatmulLeakyReluCustom
rm -rf CustomOp
# Generate the op framework
msopgen gen -i "$OP_NAME.json" -c "ai_core-${SOC_VERSION}" -lan cpp -out CustomOp
if [ $? -ne 0 ]; then
    echo "[ERROR]: msopgen failed"
    exit -1
fi
# Copy op implementation files to CustomOp
cp -rf "$OP_NAME"/* CustomOp
# Build CustomOp project
(cd CustomOp && bash build.sh)
