#!/bin/bash
_ASCEND_INSTALL_PATH=/home/service/miniconda3/Ascend/cann-8.5.0

source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64

set -e
rm -rf build
mkdir -p build
cmake -B build -DCMAKE_SKIP_RPATH=TRUE
cmake --build build -j
(
    cd build
    export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
    ./execute_whole_reduce_sum_op
)
