#!/bin/bash
set -e
SHORT=v:,i:
LONG=soc-version:,install-path:
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

SOC_VERSION="${SOC_VERSION:-Ascend910B3}"
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
        exit -1
        ;;
    esac
done

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
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

OP_NAME=MatmulLeakyReluCustom
rm -rf CustomOp

msopgen gen -i ${OP_NAME}.json -c ai_core-${SOC_VERSION} -lan cpp -out CustomOp
cp -rf ${OP_NAME}/* CustomOp
(cd CustomOp && bash build.sh)

echo "[INFO]: install build done. SOC_VERSION=${SOC_VERSION}, ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}"
