/**
 * @file whole_reduce_sum_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "whole_reduce_sum_custom_tiling.h"
#include <algorithm>
#include <cstdlib>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {

uint32_t GetEnvU32(const char *name, uint32_t defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return defaultValue;
    }
    char *end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0') {
        return defaultValue;
    }
    return static_cast<uint32_t>(parsed);
}

} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    auto const shape = context->GetInputShape(0)->GetOriginShape();
    if (shape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    uint32_t rows = shape.GetDim(0);
    uint32_t cols = shape.GetDim(1);
    if (rows == 0 || cols == 0) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = std::max<uint32_t>(1U, ascendcPlatform.GetCoreNumAiv());
    const uint32_t targetElemsPerCore = 4096U;
    uint32_t suggestedBlockDim = std::max<uint32_t>(1U, (rows * cols + targetElemsPerCore - 1U) / targetElemsPerCore);
    uint32_t blockDim = std::max<uint32_t>(1U, std::min<uint32_t>(rows, std::min<uint32_t>(coreNum, suggestedBlockDim)));
    const uint32_t maxRowsPerCore = 65535U;
    const uint32_t minBlockByRowBound = std::max<uint32_t>(1U, (rows + maxRowsPerCore - 1U) / maxRowsPerCore);
    blockDim = std::max<uint32_t>(blockDim, minBlockByRowBound);
    uint32_t forceBlockDim = GetEnvU32("WRS_FORCE_BLOCKDIM", 0U);
    if (forceBlockDim > 0U) {
        blockDim = std::max<uint32_t>(minBlockByRowBound, forceBlockDim);
    }
    blockDim = std::min<uint32_t>(rows, std::min<uint32_t>(coreNum, blockDim));
    context->SetBlockDim(blockDim);
    tiling.set_totalLength(totalLength);
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    if (xShape->GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = {xShape->GetDim(0), 1};
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class WholeReduceSumCustom : public OpDef {
public:
    explicit WholeReduceSumCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(WholeReduceSumCustom);
} // namespace ops
