/**
 * @file reduce_custom.cpp
 *
 * Copyright (C) 2022-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#define REDUCE_TILING_1 1
#define REDUCE_TILING_2 2
#define REDUCE_TILING_3 3
#define REDUCE_TILING_4 4
#define REDUCE_TILING_5 5
#define REDUCE_TILING_F16_1 11
#define REDUCE_TILING_F16_2 12
#define REDUCE_TILING_F16_3 13

template<typename DTYPE>
class KernelReduce {
static constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
static constexpr uint32_t DEFAULT_REP_STRIDE = 8;
static constexpr uint32_t REP_LEN = 256;
static constexpr uint32_t BLK_LEN = 32;
static constexpr uint32_t ONE_REPEAT_FLOAT_SIZE = REP_LEN / 4;
static constexpr uint32_t BINARY_BOUNDARY = DEFAULT_REP_STRIDE * 2;
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t outLength)
    {
        this->totalLength = totalLength;
        this->outLength = outLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE *)x, totalLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE *)z, outLength);
        pipe.InitBuffer(inQueueX, 1, totalLength * sizeof(DTYPE));
        pipe.InitBuffer(outQueueZ, 1, outLength * sizeof(DTYPE));
    }

    template<size_t ComputeKey = 0>
    __aicore__ inline void Compute()
    {
        if constexpr (ComputeKey == REDUCE_TILING_1) {
            Compute1();
        } else if constexpr (ComputeKey == REDUCE_TILING_2) {
            Compute2();
        } else if constexpr (ComputeKey == REDUCE_TILING_3) {
            Compute3();
        } else if constexpr (ComputeKey == REDUCE_TILING_4) {
            Compute4();
        } else if constexpr (ComputeKey == REDUCE_TILING_5) {
            Compute5();
        }
    }

    template<size_t ComputeKey = 0>
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute<ComputeKey>();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.AllocTensor<DTYPE>();
        AscendC::DataCopy(xLocal, xGm, totalLength);
        inQueueX.EnQue(xLocal);
    }
    // Only WholeReduceSum is used under 256B.
    __aicore__ inline void Compute1()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DTYPE>(0, totalLength);
        AscendC::WholeReduceSum<DTYPE, false>(zLocal, xLocal, 1, AscendC::MASK_PLACEHOLDER,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    // One WholeReduceSum and one BlockReduceSum are used in (256B,2KB](for float input) and (256B,4KB](for half input).
    __aicore__ inline void Compute2()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();
        constexpr uint32_t c0Count = BLK_LEN / sizeof(DTYPE);
        const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;
        pipe.InitBuffer(calcBuf, blockNum0 * sizeof(DTYPE));
        AscendC::LocalTensor<DTYPE> tempTensor1 = calcBuf.Get<DTYPE>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DTYPE>(0, totalLength);
        AscendC::BlockReduceSum<DTYPE, false>(tempTensor1, xLocal, 1, AscendC::MASK_PLACEHOLDER,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<DTYPE>(0, blockNum0);
        AscendC::WholeReduceSum<DTYPE, false>(zLocal, tempTensor1, 1, AscendC::MASK_PLACEHOLDER,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    // Two WholeReduceSum are used in (2KB,16KB](for float input) and (4KB,32KB](for half input).
    __aicore__ inline void Compute3()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();
        const uint32_t repeatNum = (totalLength * sizeof(DTYPE) + REP_LEN - 1) / REP_LEN;
        pipe.InitBuffer(calcBuf, repeatNum * sizeof(DTYPE));
        AscendC::LocalTensor<DTYPE> tempTensor1 = calcBuf.Get<DTYPE>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DTYPE>(0, totalLength);
        AscendC::WholeReduceSum<DTYPE, false>(tempTensor1, xLocal, 1, AscendC::MASK_PLACEHOLDER,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<DTYPE>(0, repeatNum);
        AscendC::WholeReduceSum<DTYPE, false>(zLocal, tempTensor1, 1, AscendC::MASK_PLACEHOLDER,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void Compute4()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();

        int64_t start = AscendC::GetSystemCycle();
        WholeReduceSumImpl(zLocal, xLocal, 1, totalLength);
        int64_t runCycle = AscendC::GetSystemCycle() - start;
        (void)runCycle;

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void Compute5()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();

        int64_t start = AscendC::GetSystemCycle();
        BinaryReduceSumImpl(zLocal, xLocal, 1, totalLength);
        int64_t runCycle = AscendC::GetSystemCycle() - start;
        (void)runCycle;

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.DeQue<DTYPE>();
        AscendC::DataCopy(zGm, zLocal, this->outLength);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void WholeReduceSumImpl(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
        const uint32_t bsLength, const uint32_t hLength)
    { 
        AscendC::SetMaskCount();
        for (uint32_t i = 0; i < bsLength; i++) {
            uint32_t totalNum = hLength;
            AscendC::LocalTensor<float> srcTmp = src[i * hLength];
            AscendC::LocalTensor<float> dstTmp = dst[i * hLength];
            while (totalNum > 1) {
                AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, totalNum);
                AscendC::WholeReduceSum<float, false>(dstTmp, srcTmp, AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
                    DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
                AscendC::PipeBarrier<PIPE_V>();
                totalNum = AscendC::DivCeil(totalNum, ONE_REPEAT_FLOAT_SIZE);
                srcTmp = dstTmp;
            }
        }
        AscendC::ResetMask();
        AscendC::SetMaskNorm();
    }

    __aicore__ inline void BinaryReduceSumImpl(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
    const uint32_t bsLength, const uint32_t hLength)
    {
        AscendC::BinaryRepeatParams binaryParams;
        AscendC::UnaryRepeatParams unaryParams;
        AscendC::SetMaskCount();
        for (uint32_t i = 0; i < bsLength; i++) {
            uint32_t totalNum = hLength;
            AscendC::LocalTensor<float> srcTmp = src[i * hLength];
            AscendC::LocalTensor<float> dstTmp = dst[i * hLength];
            while (totalNum > ONE_REPEAT_FLOAT_SIZE) {
                uint32_t halfNum = AscendC::DivCeil(totalNum, BINARY_BOUNDARY) * DEFAULT_REP_STRIDE;
                AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, totalNum - halfNum);
                AscendC::Add<float, false>(dstTmp, srcTmp, srcTmp[halfNum], AscendC::MASK_PLACEHOLDER, 1, binaryParams);
                AscendC::PipeBarrier<PIPE_V>();
                totalNum = halfNum;
                srcTmp = dstTmp;
            }
            AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, totalNum);
            AscendC::WholeReduceSum<float, false>(dstTmp, srcTmp, AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
                DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::ResetMask();
        AscendC::SetMaskNorm();
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<DTYPE> xGm;
    AscendC::GlobalTensor<DTYPE> zGm;
    uint32_t totalLength;
    uint32_t outLength;
};

extern "C" __global__ __aicore__ void reduce_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(REDUCE_TILING_1)) {
        KernelReduce<float> op;
        op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
        op.Process<REDUCE_TILING_1>();
    } else if (TILING_KEY_IS(REDUCE_TILING_2)) {
        KernelReduce<float> op;
        op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
        op.Process<REDUCE_TILING_2>();
    } else if (TILING_KEY_IS(REDUCE_TILING_3)) {
        KernelReduce<float> op;
        op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
        op.Process<REDUCE_TILING_3>();
    } else if (TILING_KEY_IS(REDUCE_TILING_4)) {
        KernelReduce<float> op;
        op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
        op.Process<REDUCE_TILING_4>();
    } else if (TILING_KEY_IS(REDUCE_TILING_5)) {
        KernelReduce<float> op;
        op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
        op.Process<REDUCE_TILING_5>();
    } else if (TILING_KEY_IS(REDUCE_TILING_F16_1)) {
        KernelReduce<half> op;
        op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
        op.Process<REDUCE_TILING_1>();
    } else if (TILING_KEY_IS(REDUCE_TILING_F16_2)) {
        KernelReduce<half> op;
        op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
        op.Process<REDUCE_TILING_2>();
    } else if (TILING_KEY_IS(REDUCE_TILING_F16_3)) {
        KernelReduce<half> op;
        op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
        op.Process<REDUCE_TILING_3>();
    }
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void reduce_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *z,
                   uint8_t *workspace, uint8_t *tiling)
{
    reduce_custom<<<blockDim, l2ctrl, stream>>>(x, z, workspace, tiling);
}
#endif
