/**
 * @file matmul_leakyrelu_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "matmul_leakyrelu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
using namespace matmul_tiling;

namespace optiling {
/**
  * @brief  Generate MatmulLeakyrelu tiling.
  * @param  context: Tiling kernel context.
  * @retval Status of GetTiling (GRAPH_SUCCESS or GRAPH_FAILED).
  */
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shape_a = context->GetInputTensor(0)->GetOriginShape();
    auto shape_b = context->GetInputTensor(1)->GetOriginShape();
    int32_t M = shape_a.GetDim(0);
    int32_t N = shape_b.GetDim(1);
    int32_t K = shape_a.GetDim(1);
    int32_t baseM = 128;
    int32_t baseN = 128;
    int32_t singleCoreM = 512;
    int32_t singleCoreN = 640;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(ascendcPlatform.GetCoreNumAiv()); // Set the number of cores that participate in multi-core computaion is 48.
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetCType(TPosition::VECIN, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        cubeTiling.SetSingleShape(singleCoreM, singleCoreN, -1);  // Set the fixed singleCoreM=512, singleCoreN=640.
        cubeTiling.SetFixSplit(baseM, baseN, -1);  // Set the fixed baseM=128, baseN=128.
    } else {
        int32_t newBaseN = 160;
        cubeTiling.SetFixSplit(-1, newBaseN, -1); // Set the fixed baseN=160.
    }
    cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    MatmulLeakyreluCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) { // Get matmul tiling data.
        return ge::GRAPH_FAILED;
    }
    tiling.set_alpha(0.001); // Set the leakyrelu tiling alpha=0.001.

    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        tiling.cubeTilingData.set_stepM(1); // Set the matmul tiling stepM=1.
        tiling.cubeTilingData.set_stepN(1); // Set the matmul tiling stepN=1.
        context->SetBlockDim(2);
        context->SetTilingKey(2);
    } else {
        /* SetBlockDim here refers to the number of cube cores, so for separated arch(AIC:AIV=1:2), 
           when vector cores number is set like 48 by SetDim, cube core number need to be set 24 here.*/ 
        context->SetBlockDim(ascendcPlatform.GetCoreNumAic());
        context->SetTilingKey(1);
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class MatmulLeakyreluCustom : public OpDef {
public:
    explicit MatmulLeakyreluCustom(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p")
            .AddConfig("ascend910b");
    }
};

OP_ADD(MatmulLeakyreluCustom);
} // namespace ops
