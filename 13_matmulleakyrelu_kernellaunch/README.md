## 概述
本样例介绍MatmulLeakyRelu算子的核函数直调方法。

## 目录结构介绍
```
├── 13_matmulleakyrelu_kernellaunch     // 使用核函数直调的方式调用MatmulLeakyRelu自定义算子。
│   ├── CppExtensions                   // kernel侧的核函数调用程序，通过使用ACLRT_LAUNCH_KERNEL调用宏来完成验证
│   ├── MatmulLeakyReluInvocation       // host侧的核函数调用程序，包含CPU侧和NPU侧两种运行验证方法
│   └── MatmulLeakyReluInvocationAsync  // host侧的核函数调用程序，包含CPU侧和NPU侧两种运行验证方法，使用了Matmul API异步Iterate接口
```

## 算子描述
算子使用了MatmulLeakyRelu高阶API，实现了快速的MatmulLeakyRelu矩阵乘法的运算操作。

MatmulLeakyRelu的计算公式为：

```
C = A * B + Bias
C = C > 0 ? C : C * 0.001
```

- A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
- C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
- Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。

## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MatmulLeakyRelu</td></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">a</td><td align="center">1024 * 256</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">b</td><td align="center">256 * 640</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">bias</td><td align="center">640</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">1024 * 640</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_leakyrelu_custom</td></tr>
</table>

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。

### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 编译运行样例工程
- [CppExtensions样例运行](./CppExtensions/README.md)
- [MatmulLeakyReluInvocation样例运行](./MatmulLeakyReluInvocation/README.md)
- [MatmulLeakyReluInvocationAsync样例运行](./MatmulLeakyReluInvocationAsync/README.md)

## 更新说明
| 时间       | 更新事项                      | 注意事项                                         |
| ---------- | ----------------------------- | ------------------------------------------------ |
| 2024/01/04 | 新增Kernel Launch调用算子样例 | 需要基于社区CANN包7.0.0.alpha003及之后版本运行   |
| 2024/02/23 | 新增pybind11调用算子样例      | 需要基于社区CANN包8.0.RC1.alpha001及之后版本运行 |
| 2024/05/21 | 新增README                    |                                                  |
| 2024/05/25 | 取消TCubeTiling大小硬编码 | 所有样例需要基于社区CANN包8.0.RC2.alpha002及之后版本运行 |
| 2024/06/11 | 取消workspace大小硬编码 |                                        |
| 2024/06/19 | 新增MatmulLeakyRelu异步Iterate接口调用样例 | 本样例仅支持Atlas A2训练系列产品 |
| 2024/11/11 | 样例目录调整 |                                        |

## 已知issue

  暂无
