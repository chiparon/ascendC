/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <algorithm>
#include <cstdlib>
#include <cstdio>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_matmul_leakyrelu_custom.h"
#else
#include "tikicpulib.h"
extern "C" void matmul_leakyrelu_custom(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
#endif

extern bool GenerateTiling(const char *socVersion, uint8_t *tilingBuf, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t preferredCoreNum);

namespace {

uint32_t GetEnvU32(const char *name, uint32_t defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return defaultValue;
    }
    char *end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0UL) {
        return defaultValue;
    }
    return static_cast<uint32_t>(parsed);
}

uint32_t ResolvePreferredCoreNum(const platform_ascendc::PlatformAscendC *platform, uint32_t m, uint32_t n)
{
    const uint32_t forced = GetEnvU32("MATMUL_FORCE_CORE_NUM", 0U);
    if (forced > 0U) {
        return forced;
    }

    const uint32_t maxAiv = std::max<uint32_t>(1U, platform->GetCoreNumAiv());
    const uint32_t tileM = (m + 255U) / 256U;
    const uint32_t tileN = (n + 127U) / 128U;
    const uint64_t tileCount = static_cast<uint64_t>(tileM) * static_cast<uint64_t>(tileN);
    if (tileCount == 0U) {
        return 1U;
    }

    const uint32_t conservative = std::max<uint32_t>(1U, maxAiv / 2U);
    const uint32_t adaptive = static_cast<uint32_t>(std::min<uint64_t>(tileCount, conservative));
    const uint32_t adaptiveMaxCore = GetEnvU32("MATMUL_ADAPTIVE_MAX_CORE", 4U);
    return std::max<uint32_t>(1U, std::min<uint32_t>(adaptive, adaptiveMaxCore));
}

} // namespace

int32_t main(int32_t argc, char *argv[])
{
    (void)argc;
    (void)argv;

    const char *socVersion = SOC_VERSION;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    const uint32_t M = GetEnvU32("MATMUL_M", 1024U);
    const uint32_t N = GetEnvU32("MATMUL_N", 640U);
    const uint32_t K = GetEnvU32("MATMUL_K", 256U);

    size_t aFileSize = static_cast<size_t>(M) * K * sizeof(int16_t);
    size_t bFileSize = static_cast<size_t>(K) * N * sizeof(int16_t);
    size_t cFileSize = static_cast<size_t>(M) * N * sizeof(float);
    size_t biasFileSize = static_cast<size_t>(N) * sizeof(float);
    size_t tilingFileSize = sizeof(TCubeTiling);
    size_t userWorkspaceSize = static_cast<size_t>(M) * N * sizeof(float);
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;

    uint8_t *tilingBuf = static_cast<uint8_t *>(malloc(tilingFileSize));
    const uint32_t preferredCoreNum = ResolvePreferredCoreNum(ascendcPlatform, M, N);
    if (!GenerateTiling(socVersion, tilingBuf, M, N, K, preferredCoreNum)) {
        std::fprintf(stderr, "[ERROR] GenerateTiling failed. Abort run.\n");
        free(tilingBuf);
        return -1;
    }
    auto *tilingMeta = reinterpret_cast<TCubeTiling *>(tilingBuf);
    if (tilingMeta->M == 0U || tilingMeta->N == 0U || tilingMeta->Ka == 0U || tilingMeta->Kb == 0U || tilingMeta->usedCoreNum == 0U ||
        tilingMeta->baseM == 0U || tilingMeta->baseN == 0U || tilingMeta->singleCoreM == 0U || tilingMeta->singleCoreN == 0U) {
        std::fprintf(stderr, "[ERROR] Invalid tiling generated (zero field detected). Abort run.\n");
        free(tilingBuf);
        return -1;
    }
#ifndef CUSTOM_ASCEND310P
    if (tilingMeta->usedCoreNum < 2) {
        std::fprintf(stderr,
                     "[ERROR] Unsupported tiling for this kernel on 910B: usedCoreNum=%u. "
                     "Single-core path is known to produce incorrect results.\n",
                     tilingMeta->usedCoreNum);
        free(tilingBuf);
        return -1;
    }
#endif

#ifdef CUSTOM_ASCEND310P
    const uint32_t blockDim = tilingMeta->usedCoreNum;
#else
    const uint32_t blockDim = (tilingMeta->usedCoreNum + 1U) / 2U;
#endif

    std::printf("[INFO] tiling: M=%u N=%u K=%u key? usedCore=%u baseM=%u baseN=%u singleCoreM=%u singleCoreN=%u blockDim=%u\n",
                tilingMeta->M, tilingMeta->N, tilingMeta->Ka, tilingMeta->usedCoreNum, tilingMeta->baseM, tilingMeta->baseN,
                tilingMeta->singleCoreM, tilingMeta->singleCoreN, blockDim);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *a = (uint8_t *)AscendC::GmAlloc(aFileSize);
    uint8_t *b = (uint8_t *)AscendC::GmAlloc(bFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);
    uint8_t *c = (uint8_t *)AscendC::GmAlloc(cFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    ReadFile("./input/x1_gm.bin", aFileSize, a, aFileSize);
    ReadFile("./input/x2_gm.bin", bFileSize, b, bFileSize);
    ReadFile("./input/bias.bin", biasFileSize, bias, biasFileSize);
    memcpy_s(tiling, tilingFileSize, tilingBuf, tilingFileSize);
    ICPU_RUN_KF(matmul_leakyrelu_custom, blockDim, a, b, bias, c, workspace, tiling);

    WriteFile("./output/output.bin", c, cFileSize);
    AscendC::GmFree((void *)a);
    AscendC::GmFree((void *)b);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)c);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)workspace);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inputAHost;
    uint8_t *inputADevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputAHost), aFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", aFileSize, inputAHost, aFileSize);
    CHECK_ACL(aclrtMemcpy(inputADevice, aFileSize, inputAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputBHost;
    uint8_t *inputBDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBHost), bFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", bFileSize, inputBHost, bFileSize);
    CHECK_ACL(aclrtMemcpy(inputBDevice, bFileSize, inputBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *outputCHost;
    uint8_t *outputCDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&outputCHost), cFileSize));
    CHECK_ACL(aclrtMalloc((void **)&outputCDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *inputBiasHost;
    uint8_t *inputBiasDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBiasHost), biasFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBiasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/bias.bin", biasFileSize, inputBiasHost, biasFileSize);
    CHECK_ACL(aclrtMemcpy(inputBiasDevice, biasFileSize, inputBiasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *tilingHost;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(matmul_leakyrelu_custom)
    (blockDim, stream, inputADevice, inputBDevice, inputBiasDevice, outputCDevice, workspaceDevice, tilingDevice);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtFree(inputADevice));
    CHECK_ACL(aclrtFreeHost(inputAHost));
    CHECK_ACL(aclrtFree(inputBDevice));
    CHECK_ACL(aclrtFreeHost(inputBHost));
    CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", outputCHost, cFileSize);
    CHECK_ACL(aclrtFree(outputCDevice));
    CHECK_ACL(aclrtFreeHost(outputCHost));
    CHECK_ACL(aclrtFree(inputBiasDevice));
    CHECK_ACL(aclrtFreeHost(inputBiasHost));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(workspaceDevice));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf);
    return 0;
}
