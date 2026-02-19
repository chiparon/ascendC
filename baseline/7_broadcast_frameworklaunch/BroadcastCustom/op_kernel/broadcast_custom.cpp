/**
 * @file broadcast_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;

class KernelBroadcastCustom {
public:
    __aicore__ inline KernelBroadcastCustom() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tilenum, uint32_t tmpSize,
                                uint32_t dim, uint32_t axis, uint32_t num, uint32_t bLength)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tilenum = tilenum;
        this->tileLength = this->blockLength / tilenum / BUFFER_NUM;
        this->dim = dim;
        this->tmpSize = tmpSize;
        this->axis = axis;
        this->num = num;
        this->bLength = bLength;
        if (this->dim == 1) {
            this->tileLength2 = num;
        } else {
            this->tileLength2 = this->tileLength * num;
        }

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->tileLength2 * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength2 * sizeof(DTYPE_Y));
        if (this->tmpSize != 0) {
            pipe.InitBuffer(tmpQueue, this->tmpSize);
        }
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tilenum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        if (this->tmpSize == 0) {
            if (this->dim == 1) {
                const uint32_t srcShape[] = {1};
                const uint32_t dstShape[] = {this->num};
                AscendC::BroadCast<DTYPE_X, 1, 0>(yLocal, xLocal, dstShape, srcShape);
            } else {
                const uint32_t srcShape[] = {this->bLength, this->tileLength / this->bLength};
                if (this->axis == 0) {
                    const uint32_t dstShape[] = {this->bLength * this->num, this->tileLength / this->bLength};
                    AscendC::BroadCast<DTYPE_X, 2, 0>(yLocal, xLocal, dstShape, srcShape);
                } else {
                    const uint32_t dstShape[] = {this->bLength, this->tileLength / this->bLength * this->num};
                    AscendC::BroadCast<DTYPE_X, 2, 1>(yLocal, xLocal, dstShape, srcShape);
                }
            }
        } else {
            AscendC::LocalTensor<uint8_t> tmpTensor = tmpQueue.Get<uint8_t>();
            if (this->dim == 1) {
                const uint32_t srcShape[] = {1};
                const uint32_t dstShape[] = {this->num};
                AscendC::BroadCast<DTYPE_X, 1, 0>(yLocal, xLocal, dstShape, srcShape, tmpTensor);
            } else {
                const uint32_t srcShape[] = {this->bLength, this->tileLength / this->bLength};
                if (this->axis == 0) {
                    const uint32_t dstShape[] = {this->bLength * this->num, this->tileLength / this->bLength};
                    AscendC::BroadCast<DTYPE_X, 2, 0>(yLocal, xLocal, dstShape, srcShape, tmpTensor);
                } else {
                    const uint32_t dstShape[] = {this->bLength, this->tileLength / this->bLength * this->num};
                    AscendC::BroadCast<DTYPE_X, 2, 1>(yLocal, xLocal, dstShape, srcShape, tmpTensor);
                }
            }
            tmpQueue.FreeTensor(tmpTensor);
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength2], yLocal, this->tileLength2);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tilenum;
    uint32_t tileLength;
    uint32_t tileLength2;
    uint32_t dim;
    uint32_t tmpSize;
    uint32_t axis;
    uint32_t num;
    uint32_t bLength;
};

extern "C" __global__ __aicore__ void broadcast_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelBroadcastCustom op;
    op.Init(x, y, tilingData.totalLength, tilingData.tilenum, tilingData.tmpSize, tilingData.dim, tilingData.axis,
            tilingData.num, tilingData.bLength);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}