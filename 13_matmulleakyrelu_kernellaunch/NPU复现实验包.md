# MatmulLeakyRelu NPU 复现实验包

## 1. 目标
- 在 NPU 环境复现已验证的优化收益。
- 保持接口与脚本不变，仅通过 `run.sh` 参数控制。

## 2. 代码路径
- 工作目录：
`/home/viparon/share/VMShare/AIal/envtrial/samples/operator/ascendc/0_introduction/13_matmulleakyrelu_kernellaunch/Optimized/v1_shape_parallel/MatmulLeakyReluInvocationAsync`

## 3. 环境前置检查
1. `ASCEND_HOME_PATH` 或 `ASCEND_INSTALL_PATH` 指向有效 CANN。
2. 目标卡可见（如 `npu-smi info` 可正常返回）。
3. 当前用户有设备运行权限。
4. 磁盘可写（`/tmp`、当前目录 `input/output`、build/install 目录）。

## 4. 一次性构建（NPU）
```bash
cd /home/viparon/share/VMShare/AIal/envtrial/samples/operator/ascendc/0_introduction/13_matmulleakyrelu_kernellaunch/Optimized/v1_shape_parallel/MatmulLeakyReluInvocationAsync

bash run.sh -r npu \
  -d /tmp/matmul_opt_npu_build \
  -p /tmp/matmul_opt_npu_out \
  --build-only
```

## 5. 对比口径
- A 组（baseline-like）：`--force-core 2`
- B 组（optimized）：`--force-core 0`
- 自适应上限：`MATMUL_ADAPTIVE_MAX_CORE=4`（默认即 4，建议显式导出）
- 每个 shape 至少 `repeat=3`，推荐 `repeat=5`。

```bash
export MATMUL_ADAPTIVE_MAX_CORE=4
```

## 6. 复现命令（推荐顺序）

### S3: (1024, 512, 1024)
```bash
# A 组
bash run.sh -r npu -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 1024 --n 512 --k 1024 --repeat 5 --force-core 2 --run-only

# B 组
bash run.sh -r npu -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 1024 --n 512 --k 1024 --repeat 5 --force-core 0 --run-only
```

### S1: (2048, 2048, 2048)
```bash
# A 组
bash run.sh -r npu -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 2048 --n 2048 --k 2048 --repeat 3 --force-core 2 --run-only

# B 组
bash run.sh -r npu -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 2048 --n 2048 --k 2048 --repeat 3 --force-core 0 --run-only
```

### S2: (4096, 1024, 4096)
```bash
# A 组
bash run.sh -r npu -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 4096 --n 1024 --k 4096 --repeat 3 --force-core 2 --run-only

# B 组
bash run.sh -r npu -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 4096 --n 1024 --k 4096 --repeat 3 --force-core 0 --run-only
```

### S4: (512, 128, 512)（功能回归点）
```bash
bash run.sh -r npu -d /tmp/matmul_opt_npu_build -p /tmp/matmul_opt_npu_out \
  --m 512 --n 128 --k 512 --repeat 3 --force-core 0 --run-only
```

## 7. 预期检查项
每次运行都应包含：
1. `error ratio: 0.0000`（或在阈值内）
2. `test pass`
3. 输出 `PERF` 三项：`AVG_MS/P50_MS/P90_MS`
4. `md5` 的 `golden.bin` 与 `output.bin` 一致

## 8. 结果记录模板
| Shape | 组别 | force-core | repeat | AVG(ms) | P50(ms) | P90(ms) | error ratio | pass/fail |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| (1024,512,1024) | A | 2 | 5 |  |  |  |  |  |
| (1024,512,1024) | B | 0 | 5 |  |  |  |  |  |
| (2048,2048,2048) | A | 2 | 3 |  |  |  |  |  |
| (2048,2048,2048) | B | 0 | 3 |  |  |  |  |  |
| (4096,1024,4096) | A | 2 | 3 |  |  |  |  |  |
| (4096,1024,4096) | B | 0 | 3 |  |  |  |  |  |
| (512,128,512) | B | 0 | 3 |  |  |  |  |  |

## 9. 异常处理清单（最小）
1. 若出现 `basic_string::_S_construct null not valid` 或 `SIGABRT`：
   - 确认 `-r npu` 时设备与驱动可用；
   - 重新执行 `--build-only`；
   - 清理后重跑：删除 `/tmp/matmul_opt_npu_build` 与 `/tmp/matmul_opt_npu_out`。
2. 若出现精度失败：
   - 先固定 `--force-core 4` 验证稳定性；
   - 再回到 `--force-core 0` 比较。
3. 若运行特别慢或不稳定：
   - 先将 `repeat` 降到 1 验证可运行，再回升到 3/5。

