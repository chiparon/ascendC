/**
 * @file whole_reduce_sum_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "whole_reduce_sum_custom_tiling.h"

constexpr uint32_t byteAlign = 32;
__aicore__ inline uint32_t ceil_div(uint32_t a, uint32_t b)
{
    if (b == 0)
        return a;
    return (a + b - 1) / b;
}

template <typename datatype> class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, WholeReduceSumCustomTilingData tilingData)
    {
        this->rows = tilingData.rows;
        this->cols = tilingData.cols;
        this->totalLength = tilingData.totalLength;
        this->colAligned = ceil_div(this->cols * sizeof(datatype), byteAlign) * byteAlign;
        this->totalSizeAligned = this->colAligned * this->rows;
        uint32_t reducedSizeAligned = ceil_div(this->rows * sizeof(datatype), byteAlign) * byteAlign;
        xGm.SetGlobalBuffer((__gm__ datatype *)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ datatype *)y, this->rows);
        pipe.InitBuffer(inQueueX, 1, this->totalSizeAligned);
        pipe.InitBuffer(outQueueY, 1, reducedSizeAligned);
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<datatype> xLocal = inQueueX.AllocTensor<datatype>();
        uint32_t colBytes = this->cols * sizeof(datatype);
        AscendC::DataCopyExtParams copyParams = {(uint16_t)this->rows, colBytes, 0, 0, 0};
        uint8_t rpad = (this->colAligned - this->cols * sizeof(datatype)) / sizeof(datatype);
        AscendC::DataCopyPadExtParams<datatype> padParams = {false, 0, rpad, 0};
        AscendC::DataCopyPad<datatype>(xLocal, xGm, copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<datatype> xLocal = inQueueX.DeQue<datatype>();
        AscendC::LocalTensor<datatype> yLocal = outQueueY.AllocTensor<datatype>();

        int32_t mask = 256 / sizeof(datatype);
        int32_t srcStride = this->colAligned / byteAlign;
        AscendC::WholeReduceSum<datatype, true>(yLocal, xLocal, this->cols, this->rows, 1, 1, srcStride);
        outQueueY.EnQue<datatype>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<datatype> yLocal = outQueueY.DeQue<datatype>();
        uint16_t reducedBytes = this->rows * sizeof(datatype);
        AscendC::DataCopyExtParams copyParams = {1, reducedBytes, 0, 0, 0};
        AscendC::DataCopyPad<datatype>(yGm, yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::GlobalTensor<datatype> xGm;
    AscendC::GlobalTensor<datatype> yGm;
    uint32_t totalLength;
    uint32_t rows;
    uint32_t cols;
    uint32_t colAligned;
    uint32_t totalSizeAligned;
};

__aicore__ inline void CopyTiling(WholeReduceSumCustomTilingData *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (int i = 0; i < sizeof(WholeReduceSumCustomTilingData) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

extern "C" __global__ __aicore__ void whole_reduce_sum_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KernelAdd<half> op;
    WholeReduceSumCustomTilingData tilingData;
    CopyTiling(&tilingData, tiling);
    op.Init(x, y, tilingData);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void whole_reduce_sum_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y,
                                uint8_t *workspace, uint8_t *tiling)
{
    whole_reduce_sum_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif