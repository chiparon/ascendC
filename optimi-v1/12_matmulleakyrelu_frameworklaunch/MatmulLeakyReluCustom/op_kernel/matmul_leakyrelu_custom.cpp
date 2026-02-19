/**
 * @file matmul_leakyrelu_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t SelectSplitRowNums(const TCubeTiling &tiling)
{
    // Keep splitRowNums power-of-two for cheap division and stable row slicing.
    uint32_t split = 8;
    if (tiling.baseM < 256U) {
        split = 4U;
    }
    if (tiling.baseM < 128U) {
        split = 2U;
    }

    while (split > 1U) {
        const uint32_t splitRow = tiling.baseM / split;
        if ((tiling.baseM % split == 0U) && (splitRow > 0U) && (tiling.singleCoreM % splitRow == 0U)) {
            // Avoid tiny VEC tiles that usually hurt scheduling efficiency.
            if (splitRow * tiling.baseN >= 1024U) {
                return split;
            }
        }
        split >>= 1U;
    }
    return 1U;
}

template <typename aType, typename bType, typename cType, typename biasType> class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

    __aicore__ inline void MatmulCompute();
    __aicore__ inline void LeakyReluCompute(uint32_t count);
    __aicore__ inline void CopyOut(uint32_t count);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    AscendC::GlobalTensor<cType> workspaceGlobal;
    AscendC::LocalTensor<cType> reluInLocal;
    TCubeTiling tiling;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> reluInQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue;
    uint32_t splitRowNums = 0;
    uint32_t splitRowSize = 0;
    uint32_t roundM = 0;
    AscendC::DataCopyParams copyParam = {0, 0, 0, 0};
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias,
                                                                               GM_ADDR c, GM_ADDR workspace,
                                                                               const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    splitRowNums = SelectSplitRowNums(tiling);
    splitRowSize = tiling.baseM / splitRowNums;
    roundM = tiling.singleCoreM / splitRowSize;
    copyParam = {(uint16_t)splitRowSize,
                 (uint16_t)(tiling.baseN * sizeof(cType) / AscendC::DEFAULT_C0_SIZE),
                 0,
                 (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / AscendC::DEFAULT_C0_SIZE)};
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(workspace), tiling.M * tiling.N);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    int32_t offsetBias = 0;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    workspaceGlobal = workspaceGlobal[AscendC::GetBlockIdx() * tiling.singleCoreM * tiling.singleCoreN];
    pipe->InitBuffer(reluInQueue, 1, tiling.baseM * tiling.baseN * sizeof(cType));
    pipe->InitBuffer(reluOutQueue, 1, splitRowSize * tiling.baseN * sizeof(cType));
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process()
{
    if (AscendC::GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }
    matmulObj.SetWorkspace(workspaceGlobal);
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    matmulObj.template Iterate<false>();
    for (int32_t i = 0; i < static_cast<int32_t>(tiling.singleCoreM * tiling.singleCoreN / (tiling.baseM * tiling.baseN)); ++i) {
        MatmulCompute();
        reluInLocal = reluInQueue.DeQue<cType>();
        for (uint32_t j = 0; j < splitRowNums; ++j) {
            LeakyReluCompute(j);
            CopyOut(i * splitRowNums + j);
        }
        reluInQueue.FreeTensor(reluInLocal);
    }
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::MatmulCompute()
{
    reluInLocal = reluInQueue.AllocTensor<cType>();
    matmulObj.template GetTensorC<false>(reluInLocal, false, true);
    reluInQueue.EnQue(reluInLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::LeakyReluCompute(uint32_t count)
{
    auto reluOutLocal = reluOutQueue.AllocTensor<cType>();
    LeakyRelu(reluOutLocal, reluInLocal[count * splitRowSize * tiling.baseN], (cType)0.001, splitRowSize * tiling.baseN);
    reluOutQueue.EnQue(reluOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count)
{
    auto reluOutLocal = reluOutQueue.DeQue<cType>();
    uint32_t startOffset = (count % roundM * splitRowSize * tiling.N + count / roundM * tiling.baseN);
    DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
    reluOutQueue.FreeTensor(reluOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                              int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                              int32_t &offsetBias)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                               GM_ADDR workspace, GM_ADDR tilingGm)
{
    GET_TILING_DATA(tilingData, tilingGm);

    MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &tilingData.cubeTilingData);
    matmulLeakyKernel.Init(a, b, bias, c, workspace, tilingData.cubeTilingData, &pipe);
    matmulLeakyKernel.Process();
}
