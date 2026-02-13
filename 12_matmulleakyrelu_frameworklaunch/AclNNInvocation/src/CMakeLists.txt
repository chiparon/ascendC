# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_execute_matmul_leakyrelu)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../output")

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    if (DEFINED ENV{ASCEND_INSTALL_PATH})
        set(INC_PATH "$ENV{ASCEND_INSTALL_PATH}")
    elseif (DEFINED ENV{ASCEND_HOME_PATH})
        set(INC_PATH "$ENV{ASCEND_HOME_PATH}")
    else ()
        set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif ()
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(CUSTOM_OPP_ROOT "")
if (DEFINED CUSTOM_OPP_PATH)
    set(CUSTOM_OPP_ROOT "${CUSTOM_OPP_PATH}")
elseif (DEFINED ENV{CUSTOM_OPP_PATH})
    set(CUSTOM_OPP_ROOT "$ENV{CUSTOM_OPP_PATH}")
elseif (DEFINED ENV{ASCEND_CUSTOM_OPP_PATH})
    string(REPLACE ":" ";" CUSTOM_OPP_PATH_LIST "$ENV{ASCEND_CUSTOM_OPP_PATH}")
    list(GET CUSTOM_OPP_PATH_LIST 0 CUSTOM_OPP_ROOT)
endif ()

if (CUSTOM_OPP_ROOT STREQUAL "")
    set(CUSTOM_OPP_ROOT "${INC_PATH}/opp/vendors/customize")
endif ()

if (CUSTOM_OPP_ROOT MATCHES "/op_api$")
    set(CUST_PKG_PATH "${CUSTOM_OPP_ROOT}")
else ()
    set(CUST_PKG_PATH "${CUSTOM_OPP_ROOT}/op_api")
endif ()
message(STATUS "CUSTOM OPP PATH: ${CUSTOM_OPP_ROOT}")
message(STATUS "CUST_PKG_PATH: ${CUST_PKG_PATH}")

set(LIB_PATH $ENV{NPU_HOST_LIB})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/lib64")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ../inc
    ${INC_PATH}/include
    ${CUST_PKG_PATH}/include
)

# add host lib path
link_directories(
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_matmul_leakyrelu_op
    operator_desc.cpp
    op_runner.cpp
    main.cpp
    op_runner.cpp
    common.cpp
)

target_link_libraries(execute_matmul_leakyrelu_op
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS execute_matmul_leakyrelu_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
