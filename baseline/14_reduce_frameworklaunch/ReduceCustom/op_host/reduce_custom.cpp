/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "reduce_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t REDUCE_TILING_1 = 1;
constexpr uint32_t REDUCE_TILING_2 = 2;
constexpr uint32_t REDUCE_TILING_3 = 3;
constexpr uint32_t REDUCE_TILING_4 = 4;
constexpr uint32_t REDUCE_TILING_5 = 5;

constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t ONE_REPEAT_LEN = 256;
constexpr uint32_t ONE_BLOCK_LEN = 32;
constexpr uint32_t OUT_SHAPE = 32;
constexpr uint32_t HALF_THRESHOLD0 = ONE_REPEAT_LEN / sizeof(uint16_t); // 128
constexpr uint32_t FLOAT_THRESHOLD0 = ONE_REPEAT_LEN / sizeof(float); // 64
constexpr uint32_t HALF_THRESHOLD1 = ONE_REPEAT_LEN / sizeof(uint16_t) * ONE_BLOCK_LEN / sizeof(uint16_t); // 2048
constexpr uint32_t FLOAT_THRESHOLD1 = ONE_REPEAT_LEN / sizeof(float) * ONE_BLOCK_LEN / sizeof(float); //512
constexpr uint32_t HALF_THRESHOLD2 = ONE_REPEAT_LEN / sizeof(uint16_t) * ONE_REPEAT_LEN / sizeof(uint16_t); // 16384
constexpr uint32_t FLOAT_THRESHOLD2 = ONE_REPEAT_LEN / sizeof(float) * ONE_REPEAT_LEN / sizeof(float); // 4096
constexpr uint32_t WHOLEREDUCESUM_SIGLE_MODE = 10000;
constexpr uint32_t BINARYREDUCESUM_SIGLE_MODE = 20000;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    auto inputDtype = context->GetInputTensor(0)->GetDataType();
    // Only WholeReduceSum is used under 256B.
    if ((totalLength <= HALF_THRESHOLD0 && inputDtype == ge::DT_FLOAT16) ||
        (totalLength <= FLOAT_THRESHOLD0 && inputDtype == ge::DT_FLOAT)) {
        context->SetTilingKey(REDUCE_TILING_1);
    // One WholeReduceSum and one BlockReduceSum are used in (256B,2KB](for float input) and (256B,4KB](for half input).
    } else if ((totalLength <= HALF_THRESHOLD1 && inputDtype == ge::DT_FLOAT16) ||
        (totalLength <= FLOAT_THRESHOLD1 && inputDtype == ge::DT_FLOAT)) {
        context->SetTilingKey(REDUCE_TILING_2);
    // Two WholeReduceSum are used in (2KB,16KB](for float input) and (4KB,32KB](for half input).
    } else if ((totalLength <= HALF_THRESHOLD2 && inputDtype == ge::DT_FLOAT16) ||
        (totalLength <= FLOAT_THRESHOLD2 && inputDtype == ge::DT_FLOAT)) {
        context->SetTilingKey(REDUCE_TILING_3);
    } else if (totalLength == WHOLEREDUCESUM_SIGLE_MODE) {
        context->SetTilingKey(REDUCE_TILING_4);
    } else if (totalLength == BINARYREDUCESUM_SIGLE_MODE) {
        context->SetTilingKey(REDUCE_TILING_5);
    }
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_outLength(OUT_SHAPE);
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
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = {optiling::OUT_SHAPE};
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ReduceCustom : public OpDef {
public:
    explicit ReduceCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p")
            .AddConfig("ascend910b");
    }
};
OP_ADD(ReduceCustom);
} // namespace ops
