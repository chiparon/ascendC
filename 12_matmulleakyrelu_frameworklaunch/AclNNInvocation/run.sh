#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

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

resolve_custom_opp_root() {
    local path_candidate="$CUSTOM_OPP_PATH"
    if [ -z "$path_candidate" ] && [ -n "$ASCEND_CUSTOM_OPP_PATH" ]; then
        path_candidate="${ASCEND_CUSTOM_OPP_PATH%%:*}"
    fi

    if [ -z "$path_candidate" ]; then
        echo ""
        return
    fi

    if [[ "$path_candidate" == */op_api ]]; then
        dirname "$path_candidate"
        return
    fi
    if [[ "$path_candidate" == */vendors/customize ]]; then
        echo "$path_candidate"
        return
    fi
    if [ -d "$path_candidate/vendors/customize/op_api" ]; then
        echo "$path_candidate/vendors/customize"
        return
    fi
    echo "$path_candidate"
}

_ASCEND_INSTALL_PATH=$(resolve_install_path)
if [ -z "$_ASCEND_INSTALL_PATH" ]; then
    echo "[ERROR]: Cannot resolve valid CANN install path, please set ASCEND_INSTALL_PATH."
    exit 1
fi
source "$_ASCEND_INSTALL_PATH/bin/setenv.bash"
export ASCEND_INSTALL_PATH="$_ASCEND_INSTALL_PATH"
export ASCEND_HOME_PATH="$_ASCEND_INSTALL_PATH"
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64
CUSTOM_OPP_ROOT=$(resolve_custom_opp_root)
if [ -n "$CUSTOM_OPP_ROOT" ]; then
    export CUSTOM_OPP_PATH="$CUSTOM_OPP_ROOT"
    export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_ROOT"
else
    export CUSTOM_OPP_PATH="$_ASCEND_INSTALL_PATH/opp/vendors/customize"
fi

function main {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm ./input/*.bin
    rm ./output/*.bin

    # 2. 生成输入数据和真值数据
    cd $CURRENT_DIR
    python3 scripts/gen_data.py
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Generate input data failed!"
        return 1
    fi
    echo "[INFO]: Generate input data success!"

    # 3. 编译acl可执行文件
    cd $CURRENT_DIR
    rm -rf build
    mkdir -p build
    cd build
    cmake ../src -DCMAKE_SKIP_RPATH=TRUE -DCUSTOM_OPP_PATH="$CUSTOM_OPP_PATH"
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Cmake failed!"
        return 1
    fi
    echo "[INFO]: Cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Make failed!"
        return 1
    fi
    echo "[INFO]: Make success!"

    # 4. 运行可执行文件
    export LD_LIBRARY_PATH="$CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
    cd $CURRENT_DIR/output
    echo "[INFO]: Execute op!"
    ./execute_matmul_leakyrelu_op
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Acl executable run failed! please check your project!"
        return 1
    fi
    echo "[INFO]: Acl executable run success!"

    # 5. 比较真值文件
    cd $CURRENT_DIR
    python3 scripts/verify_result.py output/output_z.bin output/golden.bin
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Verify result failed!"
        return 1
    fi
}

main
