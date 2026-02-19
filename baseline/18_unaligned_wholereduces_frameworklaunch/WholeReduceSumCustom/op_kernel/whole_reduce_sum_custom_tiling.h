/**
 * @file whole_reduce_sum_custom_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef WHOLE_REDUCE_SUM_CUSTOM_TILING_H
#define WHOLE_REDUCE_SUM_CUSTOM_TILING_H
#include <cstdint>

struct WholeReduceSumCustomTilingData {
    uint32_t totalLength;
    uint32_t rows;
    uint32_t cols;
};
#endif // WHOLE_REDUCE_SUM_CUSTOM_TILING_H