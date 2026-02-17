#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import os
import numpy as np


def get_shape():
    m = int(os.getenv("MATMUL_M", "1024"))
    n = int(os.getenv("MATMUL_N", "640"))
    k = int(os.getenv("MATMUL_K", "256"))
    return m, n, k


def gen_golden_data_simple():
    m, n, k = get_shape()

    seed = int(os.getenv("MATMUL_SEED", "2026"))
    rng = np.random.default_rng(seed)

    input_a = rng.integers(1, 10, [m, k], dtype=np.int32).astype(np.float16)
    input_b = rng.integers(1, 10, [k, n], dtype=np.int32).astype(np.float16)
    input_bias = rng.integers(1, 10, [n], dtype=np.int32).astype(np.float32)

    alpha = 0.001
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)
    golden = np.where(golden >= 0, golden, golden * alpha)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    input_bias.tofile("./input/bias.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
