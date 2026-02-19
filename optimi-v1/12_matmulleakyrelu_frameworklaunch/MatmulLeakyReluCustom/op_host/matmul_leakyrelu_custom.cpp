/**
 * @file matmul_leakyrelu_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

#include "matmul_leakyrelu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace {

inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

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

struct SplitConfig {
    int32_t baseM;
    int32_t baseN;
};

bool TryGenerateOnce(const platform_ascendc::PlatformAscendC &platform, TCubeTiling &cubeTilingData, uint32_t M, uint32_t N,
                     uint32_t K, uint32_t usedCoreNum, int32_t baseM, int32_t baseN)
{
    MultiCoreMatmulTiling tilingApi(platform);
    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetCType(TPosition::VECIN, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(true);
    tilingApi.SetTraverse(MatrixTraverse::FIRSTM);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBufferSpace(-1, -1, -1);

    if (tilingApi.GetTiling(cubeTilingData) == -1) {
        return false;
    }
    cubeTilingData.set_stepM(1);
    cubeTilingData.set_stepN(1);

    const bool invalidTileShape = (cubeTilingData.singleCoreM < cubeTilingData.baseM) ||
                                  (cubeTilingData.singleCoreN < cubeTilingData.baseN) ||
                                  (cubeTilingData.singleCoreM % cubeTilingData.baseM != 0U) ||
                                  (cubeTilingData.singleCoreN % cubeTilingData.baseN != 0U);
    return !invalidTileShape;
}

} // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shapeA = context->GetInputTensor(0)->GetOriginShape();
    auto shapeB = context->GetInputTensor(1)->GetOriginShape();
    const uint32_t M = static_cast<uint32_t>(shapeA.GetDim(0));
    const uint32_t K = static_cast<uint32_t>(shapeA.GetDim(1));
    const uint32_t N = static_cast<uint32_t>(shapeB.GetDim(1));

    uint32_t tilingKey = 0U;
    if (M == 512U && N == 128U && K == 512U) {
        tilingKey = 1U;
    } else if (M == 2048U && N == 2048U && K == 2048U) {
        tilingKey = 2U;
    } else if (M == 4096U && N == 1024U && K == 4096U) {
        tilingKey = 3U;
    } else if (M == 1024U && N == 512U && K == 1024U) {
        tilingKey = 4U;
    }

    const int32_t defaultBaseM = (M >= 4096U) ? 128 : 256;
    const int32_t defaultBaseN = (N >= 2048U || (N % 256U == 0U && N >= 1024U)) ? 256 : 128;
    const uint32_t forceBaseM = GetEnvU32("MATMUL_FORCE_BASE_M", 0U);
    const uint32_t forceBaseN = GetEnvU32("MATMUL_FORCE_BASE_N", 0U);
    const uint32_t forceCore = GetEnvU32("MATMUL_FORCE_CORE_NUM", 0U);

    std::vector<SplitConfig> splitCandidates = {
        {defaultBaseM, defaultBaseN},
        {128, 128},
        {256, 128},
        {64, 128},
        {128, 64},
    };
    if (tilingKey == 1U) {
        splitCandidates = {{128, 128}, {256, 128}, {64, 128}, {128, 64}, {defaultBaseM, defaultBaseN}};
    } else if (tilingKey == 2U) {
        splitCandidates = {{128, 256}, {256, 256}, {256, 128}, {128, 128}, {64, 128}};
    } else if (tilingKey == 3U) {
        splitCandidates = {{128, 128}, {256, 128}, {128, 256}, {64, 128}, {128, 64}};
    } else if (tilingKey == 4U) {
        splitCandidates = {{256, 128}, {128, 128}, {128, 64}, {64, 128}, {defaultBaseM, defaultBaseN}};
    }

    if (forceBaseM > 0U && forceBaseN > 0U) {
        splitCandidates.insert(splitCandidates.begin(), {static_cast<int32_t>(forceBaseM), static_cast<int32_t>(forceBaseN)});
    }

    std::vector<SplitConfig> uniqCandidates;
    uniqCandidates.reserve(splitCandidates.size());
    for (const auto &c : splitCandidates) {
        bool seen = false;
        for (const auto &u : uniqCandidates) {
            if (u.baseM == c.baseM && u.baseN == c.baseN) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            uniqCandidates.push_back(c);
        }
    }
    splitCandidates.swap(uniqCandidates);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const bool is310p = (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P);
    const uint32_t maxCoreNum = std::max<uint32_t>(1U, platform.GetCoreNumAiv());
    const uint32_t tileM = CeilDiv(M, 256U);
    const uint32_t tileN = CeilDiv(N, 128U);
    const uint64_t tileCount = static_cast<uint64_t>(tileM) * static_cast<uint64_t>(tileN);
    const uint32_t adaptiveMaxCore = GetEnvU32("MATMUL_ADAPTIVE_MAX_CORE", 4U);
    const uint32_t adaptiveDefault = static_cast<uint32_t>(std::min<uint64_t>(tileCount, std::max<uint32_t>(1U, maxCoreNum / 2U)));
    const uint32_t preferredCore = (forceCore > 0U) ? forceCore : std::max<uint32_t>(1U, std::min<uint32_t>(adaptiveDefault, adaptiveMaxCore));
    const uint32_t preferredCap = std::min<uint32_t>(maxCoreNum, preferredCore == 0U ? maxCoreNum : preferredCore);

    MatmulLeakyreluCustomTilingData tiling;
    bool found = false;
    for (const auto &split : splitCandidates) {
        const uint64_t splitTileCount = static_cast<uint64_t>(CeilDiv(M, static_cast<uint32_t>(split.baseM))) *
                                        static_cast<uint64_t>(CeilDiv(N, static_cast<uint32_t>(split.baseN)));
        if (splitTileCount == 0U) {
            continue;
        }
        uint32_t startCore = std::min<uint32_t>(preferredCap, static_cast<uint32_t>(splitTileCount));
        if (startCore >= 2U) {
            for (uint32_t core = startCore; core >= 2U; --core) {
                if (core > 2U && (core & 1U) != 0U) {
                    continue;
                }
                if (TryGenerateOnce(platform, tiling.cubeTilingData, M, N, K, core, split.baseM, split.baseN)) {
                    found = true;
                    break;
                }
            }
        }
        if (found) {
            break;
        }
    }

    if (!found && is310p) {
        for (const auto &split : splitCandidates) {
            if (TryGenerateOnce(platform, tiling.cubeTilingData, M, N, K, 1U, split.baseM, split.baseN)) {
                found = true;
                break;
            }
        }
    }

    if (!found) {
        std::cout << "gen tiling failed for shape M=" << M << ", N=" << N << ", K=" << K
                  << " on soc=" << static_cast<int32_t>(platform.GetSocVersion()) << std::endl;
        return ge::GRAPH_FAILED;
    }

    if (tiling.cubeTilingData.usedCoreNum == 0U) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_alpha(0.001f);

    if (is310p) {
        context->SetBlockDim(tiling.cubeTilingData.usedCoreNum);
    } else {
        if (tiling.cubeTilingData.usedCoreNum < 2U) {
            std::cout << "unsupported usedCoreNum=" << tiling.cubeTilingData.usedCoreNum << " on 910B" << std::endl;
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim((tiling.cubeTilingData.usedCoreNum + 1U) / 2U);
    }
    context->SetTilingKey(1);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t userWorkspaceSize = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
    size_t systemWorkspaceSize = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = userWorkspaceSize + systemWorkspaceSize;

    std::cout << "select tiling key=" << tilingKey << " usedCore=" << tiling.cubeTilingData.usedCoreNum
              << " baseM=" << tiling.cubeTilingData.baseM << " baseN=" << tiling.cubeTilingData.baseN
              << " blockDim=" << ((tiling.cubeTilingData.usedCoreNum + 1U) / 2U) << std::endl;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class MatmulLeakyreluCustom : public OpDef {
public:
    explicit MatmulLeakyreluCustom(const char *name) : OpDef(name)
    {
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend310p").AddConfig("ascend910b");
    }
};

OP_ADD(MatmulLeakyreluCustom);
} // namespace ops
