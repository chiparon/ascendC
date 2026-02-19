/**
 * @file broadcast_custom_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef BROADCAST_CUSTOM_TILING_H
#define BROADCAST_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BroadcastTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tilenum);
TILING_DATA_FIELD_DEF(uint32_t, tmpSize);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, isReuseSource);
TILING_DATA_FIELD_DEF(uint32_t, axis);
TILING_DATA_FIELD_DEF(uint32_t, num);
TILING_DATA_FIELD_DEF(uint32_t, bLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BroadcastCustom, BroadcastTilingData)
} // namespace optiling
#endif  // BROADCAST_CUSTOM_TILING_H