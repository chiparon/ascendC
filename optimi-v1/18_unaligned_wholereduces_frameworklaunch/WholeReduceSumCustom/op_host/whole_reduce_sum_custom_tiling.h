/**
 * @file whole_reduce_sum_custom_tiling.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef WHOLE_REDUCE_SUM_CUSTOM_TILING_H
#define WHOLE_REDUCE_SUM_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, cols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WholeReduceSumCustom, TilingData)
} // namespace optiling
#endif // WHOLE_REDUCE_SUM_CUSTOM_TILING_H
