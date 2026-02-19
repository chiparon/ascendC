/**
 * @file broadcast_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "broadcast_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    BroadcastTilingData tiling;
    const gert::RuntimeAttrs *broadcastattrs = context->GetAttrs();
    const uint32_t bufferMode = *(broadcastattrs->GetAttrPointer<uint32_t>(0));
    const uint32_t dim = *(broadcastattrs->GetAttrPointer<uint32_t>(1));
    const uint32_t isReuseSource = *(broadcastattrs->GetAttrPointer<uint32_t>(2));
    const uint32_t axis = *(broadcastattrs->GetAttrPointer<uint32_t>(3));
    const uint32_t num = *(broadcastattrs->GetAttrPointer<uint32_t>(4));
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    if (totalLength == 0 || num == 0 || (dim != 1 && dim != 2)) {
        return ge::GRAPH_FAILED;
    }
    if (dim == 2 && axis > 1) {
        return ge::GRAPH_FAILED;
    }
    auto dt = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypesize;
    if (dt == ge::DT_FLOAT16) {
        dtypesize = 2;
    } else if (dt == ge::DT_FLOAT) {
        dtypesize = 4;
    } else if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        dtypesize = 1;
    } else {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *src_shape = context->GetInputShape(0);
    uint32_t bLength;
    uint32_t sLength;
    ge::Shape inputShape;
    ge::Shape outputShape;
    uint32_t tilenum = 1;
    if (dim == 1) {
        bLength = src_shape->GetStorageShape().GetDim(0);
        std::vector<int64_t> inputShapeDim = {1};
        inputShape = ge::Shape(inputShapeDim);
        std::vector<int64_t> outputShapeDim = {num};
        outputShape = ge::Shape(outputShapeDim);
        tilenum = totalLength >= 4U ? 4U : totalLength;
        while (tilenum > 1U && (totalLength % tilenum != 0U)) {
            --tilenum;
        }
    } else {
        bLength = src_shape->GetStorageShape().GetDim(0);
        sLength = src_shape->GetStorageShape().GetDim(1);
        if (bLength == 0 || sLength == 0) {
            return ge::GRAPH_FAILED;
        }
        std::vector<int64_t> inputShapeDim = {bLength, sLength};
        inputShape = ge::Shape(inputShapeDim);
        if (axis == 0) {
            std::vector<int64_t> outputShapeDim = {bLength * num, sLength};
            outputShape = ge::Shape(outputShapeDim);
            tilenum = 1;
        } else {
            std::vector<int64_t> outputShapeDim = {bLength, sLength * num};
            outputShape = ge::Shape(outputShapeDim);
            tilenum = sLength >= 4U ? 4U : sLength;
            while (tilenum > 1U && (sLength % tilenum != 0U)) {
                --tilenum;
            }
        }
    }
    uint32_t tmpSize;
    uint32_t maxsize = 0;
    uint32_t minsize = 0;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (isReuseSource == 0) {
        AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, inputShape, outputShape, dtypesize, false, maxsize,
                                           minsize);
    } else {
        AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, inputShape, outputShape, dtypesize, true, maxsize, minsize);
    }
    if (bufferMode == 0) {
        tmpSize = 0;
    } else if (bufferMode == 1) {
        tmpSize = minsize;
    } else if (bufferMode == 2) {
        tmpSize = maxsize;
    } else {
        tmpSize = (maxsize + minsize) / 2;
    }
    tiling.set_tmpSize(tmpSize);

    context->SetBlockDim(1);
    tiling.set_totalLength(totalLength);
    tiling.set_tilenum(tilenum);
    tiling.set_isReuseSource(isReuseSource);
    tiling.set_axis(axis);
    tiling.set_num(num);
    tiling.set_bLength(bLength);
    tiling.set_dim(dim);

    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkSpace = context->GetWorkspaceSizes(1);
    currentWorkSpace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
namespace ops {
class BroadcastCustom : public OpDef {
public:
    BroadcastCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("bufferMode").AttrType(REQUIRED).Int(0);
        this->Attr("dim").AttrType(REQUIRED).Int(0);
        this->Attr("isReuseSource").AttrType(REQUIRED).Int(0);
        this->Attr("axis").AttrType(REQUIRED).Int(0);
        this->Attr("num").AttrType(REQUIRED).Int(0);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p")
            .AddConfig("ascend910b");
    }
};

OP_ADD(BroadcastCustom);
} // namespace ops
