#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed <= 0:
            return default
        return parsed
    except Exception:
        return default


def gen_golden_data():
    m = get_env_int("MATMUL_M", 1024)
    n = get_env_int("MATMUL_N", 640)
    k = get_env_int("MATMUL_K", 256)

    input_a = np.random.randint(1, 10, [m, k]).astype(np.float16)
    input_b = np.random.randint(1, 10, [k, n]).astype(np.float16)
    input_bias = np.random.randint(1, 10, [n]).astype(np.float32)

    alpha = 0.001
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)
    golden = np.where(golden >= 0, golden, golden * alpha)

    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    input_a.tofile("./input/input_a.bin")
    input_b.tofile("./input/input_b.bin")
    input_bias.tofile("./input/input_bias.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
