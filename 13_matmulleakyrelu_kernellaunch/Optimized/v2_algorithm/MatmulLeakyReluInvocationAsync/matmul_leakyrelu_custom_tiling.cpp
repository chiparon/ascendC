/**
 * @file matmul_leakyrelu_custom_tiling.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "kernel_tiling/kernel_tiling.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;
using namespace std;

namespace {

inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

struct SplitConfig {
    int32_t baseM;
    int32_t baseN;
};

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

bool TryGenerateOnce(const platform_ascendc::PlatformAscendC *platform, uint8_t *tilingBuf, uint32_t M, uint32_t N, uint32_t K,
                     uint32_t usedCoreNum, int32_t baseM, int32_t baseN)
{
    TPosition leftPosition = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    bool isTransA = false;

    TPosition rightPosition = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    bool isTransB = false;

    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT;

    TPosition biasPosition = TPosition::GM;
    CubeFormat biasFormat = CubeFormat::ND;
    DataType biasDtype = DataType::DT_FLOAT;
    bool isBias = true;

    optiling::TCubeTiling tilingData;
    MultiCoreMatmulTiling tilingApi(*platform);
    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);
    tilingApi.SetBiasType(biasPosition, biasFormat, biasDtype);
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(isBias);
    tilingApi.SetTraverse(MatrixTraverse::FIRSTM);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBufferSpace(-1, -1, -1);

    const int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        return false;
    }
    tilingData.set_stepM(1);
    tilingData.set_stepN(1);
    tilingData.SaveToBuffer(tilingBuf, tilingData.GetDataSize());

    // Kernel pipeline assumes per-core region is composed of full baseM x baseN tiles.
    const auto *tiling = reinterpret_cast<const TCubeTiling *>(tilingBuf);
    const bool invalidTileShape = (tiling->singleCoreM < tiling->baseM) || (tiling->singleCoreN < tiling->baseN) ||
                                  (tiling->singleCoreM % tiling->baseM != 0U) || (tiling->singleCoreN % tiling->baseN != 0U);
    if (invalidTileShape) {
        return false;
    }
    return true;
}

} // namespace

/**
  * @brief  Generate matmul tiling.
  * @param  socVersion: Platform socversion.
  * @param  tilingBuf data buffer.
  */
bool GenerateTiling(const char *socVersion, uint8_t *tilingBuf, uint32_t M, uint32_t N, uint32_t K, uint32_t preferredCoreNum)
{
    uint32_t tilingKey = 0U;
    if (M == 512U && N == 128U && K == 512U) {
        tilingKey = 1U;
    } else if (M == 2048U && N == 2048U && K == 2048U) {
        tilingKey = 2U; // S1
    } else if (M == 4096U && N == 1024U && K == 4096U) {
        tilingKey = 3U; // S2
    } else if (M == 1024U && N == 512U && K == 1024U) {
        tilingKey = 4U; // S3
    }

    const int32_t baseM = (M >= 4096U) ? 128 : 256;
    const int32_t baseN = (N >= 2048U || (N % 256U == 0U && N >= 1024U)) ? 256 : 128;
    const uint32_t forceBaseM = GetEnvU32("MATMUL_FORCE_BASE_M", 0U);
    const uint32_t forceBaseN = GetEnvU32("MATMUL_FORCE_BASE_N", 0U);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    const uint32_t maxCoreNum = std::max<uint32_t>(1U, ascendcPlatform->GetCoreNumAiv());
    const uint32_t preferredCap = std::min<uint32_t>(maxCoreNum, preferredCoreNum == 0U ? maxCoreNum : preferredCoreNum);

    // Keep a deterministic split search order. For small-N cases (e.g. N=128), try tighter M split first.
    std::vector<SplitConfig> splitCandidates = {
        {baseM, baseN},
        {128, 128},
        {256, 128},
        {64, 128},
        {128, 64},
    };
    if (tilingKey == 1U) {
        splitCandidates = {
            {128, 128},
            {256, 128},
            {64, 128},
            {128, 64},
            {baseM, baseN},
        };
    } else if (tilingKey == 2U) {
        splitCandidates = {
            {128, 256},
            {256, 256},
            {256, 128},
            {128, 128},
            {64, 128},
        };
    } else if (tilingKey == 3U) {
        splitCandidates = {
            {128, 128},
            {256, 128},
            {128, 256},
            {64, 128},
            {128, 64},
        };
    } else if (tilingKey == 4U) {
        splitCandidates = {
            {256, 128},
            {128, 128},
            {128, 64},
            {64, 128},
            {baseM, baseN},
        };
    }

    if (forceBaseM > 0U && forceBaseN > 0U) {
        splitCandidates.insert(splitCandidates.begin(), {static_cast<int32_t>(forceBaseM), static_cast<int32_t>(forceBaseN)});
    }

    // Deduplicate candidate list while preserving order to keep tuning logs deterministic.
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

    // Prefer multi-core plans. Single-core is kept as a last-resort fallback.
    for (const auto &split : splitCandidates) {
        const uint64_t tileCount = static_cast<uint64_t>(CeilDiv(M, static_cast<uint32_t>(split.baseM))) *
                                   static_cast<uint64_t>(CeilDiv(N, static_cast<uint32_t>(split.baseN)));
        if (tileCount == 0U) {
            continue;
        }
        uint32_t startCoreNum = std::min<uint32_t>(preferredCap, static_cast<uint32_t>(tileCount));
        if (startCoreNum >= 2U) {
            for (uint32_t core = startCoreNum; core >= 2U; --core) {
                if (core > 2U && (core & 1U) != 0U) {
                    continue; // Keep even core plan to match blockDim mapping on 910B.
                }
                if (TryGenerateOnce(ascendcPlatform, tilingBuf, M, N, K, core, split.baseM, split.baseN)) {
                    std::cout << "select tiling key=" << tilingKey << " core=" << core << " baseM=" << split.baseM
                              << " baseN=" << split.baseN << std::endl;
                    return true;
                }
            }
        }
    }

    for (const auto &split : splitCandidates) {
        if (TryGenerateOnce(ascendcPlatform, tilingBuf, M, N, K, 1U, split.baseM, split.baseN)) {
            std::cout << "select tiling key=" << tilingKey << " core=1 baseM=" << split.baseM
                      << " baseN=" << split.baseN << std::endl;
            return true;
        }
    }

    std::cout << "gen tiling failed for shape M=" << M << ", N=" << N << ", K=" << K << std::endl;
    return false;
}
