## 概述
利用aclnn框架，提供了ReduceCustom自定义算子的一个测试工程。

## 目录结构介绍

```
├── AclNNInvocationNaive
│   ├── CMakeLists.txt      // 编译规则文件
│   ├── main.cpp            // 单算子调用应用的入口
│   └── run.sh              // 编译运行算子的脚本
```

## 代码实现介绍

完成自定义算子的开发部署后，可以通过单算子调用的方式来验证单算子的功能。main.cpp代码为单算子API执行方式。单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。

自定义算子编译部署后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，形如：

```cpp
// 获取算子使用的workspace空间大小
aclnnStatus aclnnReduceCustomGetWorkspaceSize(const aclTensor *x, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
// 执行算子
aclnnStatus aclnnReduceCustom(void *workspace, int64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

其中aclnnReduceCustomGetWorkspaceSize为第一段接口，主要用于计算本次API调用计算过程中需要多少的workspace内存。获取到本次API计算需要的workspace大小之后，按照workspaceSize大小申请Device侧内存，然后调用第二段接口aclnnReduceCustom执行计算。具体参考[单算子API调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)章节。

## 运行样例算子

### 1. 编译算子工程

运行此样例前，请参考[编译算子工程](../README.md#operatorcompile)完成前期准备。

### 2. aclnn调用样例运行

  用户可参考run.sh脚本进行编译与运行。
  ```bash
  bash run.sh
  ```

## 更新说明


| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/09/05 | 新增本readme |
| 2024/11/11 | 样例目录调整 |
