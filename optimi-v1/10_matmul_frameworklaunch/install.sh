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
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "[ERROR]: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

_ASCEND_INSTALL_PATH=/home/service/miniconda3/Ascend/cann-8.5.0

source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export ASCEND_HOME_PATH=$_ASCEND_INSTALL_PATH

OP_NAME=MatmulCustom
rm -rf CustomOp
# Generate the op framework
msopgen gen -i $OP_NAME.json -c ai_core-${SOC_VERSION} -lan cpp -out CustomOp
# Copy op implementation files to CustomOp, select one of the following two options
cp -rf MatmulCustomSingleCore/* CustomOp # cp -rf MatmulCustomMultiCore/* CustomOp
# Build CustomOp project
(cd CustomOp && bash build.sh)
