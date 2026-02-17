# 上下文压缩（当前状态）

## 1) 目标与口径
- 最终目标未变：高维 shape（至少 `S1(2048,2048,2048)`、`S2(4096,1024,4096)`）性能提升 `>=20%`，且接口兼容、精度正确。
- 当前只聚焦 `Ascend910B3`，不再按 SOC 分线。
- `results.md` 被判定为旧数据，不再作为参考。

## 2) 版本分层（已落实）
- `v1`：回退为“上一版可对照版本”（用于稳定基线）。
  - 路径：`Optimized/v1_shape_parallel/MatmulLeakyReluInvocationAsync`
  - 已回退文件：
    - `matmul_leakyrelu_custom.cpp`
    - `matmul_leakyrelu_custom_tiling.cpp`
    - `run.sh`（默认 SOC 回到 `Ascend910B1`，实测时可显式 `-v Ascend910B3`）
- `v2`：用于算法改造与 bug 修复。
  - 路径：`Optimized/v2_algorithm/MatmulLeakyReluInvocationAsync`

## 3) v2 失败现象（你提供）
- `kernel_results/v2_try1.md`：`S1/S2/S3` 均失败，`S4` 通过。
- 典型错误：大量输出在固定偏移后为 0，`error ratio` 很高。
- 运行中出现：`main.cpp:186 aclError:507015`，在 `aclrtSynchronizeStream` 处报错，说明 kernel 执行阶段异常。

## 4) 已做修复（不是阉割）
目标是直接修掉触发错误的执行逻辑：
1. **修正 v2 kernel 路径**
- 去掉高风险的多-tile预取流水（该路径与 `507015 + 大面积错写` 高相关）。
- 保留 in-place LeakyReLU（减少中间队列访存）。
- 文件：`Optimized/v2_algorithm/MatmulLeakyReluInvocationAsync/matmul_leakyrelu_custom.cpp`

2. **增强错误防护**
- `CHECK_ACL` 改为 fail-fast（ACL错误直接中止），防止继续产出无效性能数据。
- 文件：`Optimized/v2_algorithm/MatmulLeakyReluInvocationAsync/data_utils.h`

## 5) 本地验证结果（v2修复后）
- 编译通过：`--build-only`
- `S4 (512,128,512)`：`test pass`，`error ratio: 0.0000`
- `S3 (1024,512,1024)`：`test pass`，`error ratio: 0.0000`

## 6) 当前结论
- v2 的致命错误路径已被替换为可运行且本地精度通过的实现。
- 仍需在 NPU 上重跑确认 `S1/S2` 是否彻底消除 `507015` 与结果错误。
- 建议顺序：`S1A(repeat=1)` → `S1B` → `S2A/S2B` → `S3A/S3B`。
