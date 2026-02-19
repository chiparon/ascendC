/**
 * @file matmul_custom_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef MATMUL_CUSTOM_TILING_H
#define MATMUL_CUSTOM_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MatmulCustomTilingData {
    uint64_t localMemSize;
    AscendC::tiling::TCubeTiling cubeTilingData;
};

#endif  // MATMUL_CUSTOM_TILING_H