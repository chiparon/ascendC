## 概述
本样例基于ReduceCustom算子工程，介绍了AscendC基础API WholeReduceSum和BlockReduceSum的用法。WholeReduceMax/BlockReduceMax以及WholeReduceMin/BlockReduceMin的使用可以将本样例用作参考样例。

## 目录结构介绍
```
├── 14_reduce_frameworklaunch  // 使用框架调用的方式调用ReduceCustom算子
│   ├── AclNNInvocationNaive   // 通过aclnn调用的方式调用ReduceCustom算子, 简化了编译脚本
│   ├── ReduceCustom           // ReduceCustom算子工程
│   ├── install.sh             // 脚本，调用msOpGen生成自定义算子工程，并编译
│   └── ReduceCustom.json      // ReduceCustom算子的原型定义json文件
```

## 算子描述

ReduceCustom算子实现了连续内存上数据元素的累加，返回累加结果的功能。对应的数学表达式为：
```
z = sum(x)
```

样例将一段连续的输入做累加，得到这段连续buffer内元素的和。

1、在小于256B时，采用WholeReduceSum一次性可以得到结果。

2、长度在float输入(256B,2KB]，或者half输入(256B,4KB]时。由于同样长度的输入，BlockReduceSum比WholeReduceSum的执行速度更快，所以采用一条BlockReduceSum加一条WholeReduceSum的做法，得到更高的计算效率。

3、长度在float输入(2KB,16KB]，或者half输入(4KB,32KB]时。由于一条WholeReduceSum的累加效率比使用两条BlockReduceSum的累加效率更高。所以采用两条WholeReduceSum（而不是两条BlockReduceSum+一条WholeReduceSum），得到这段buffer的累加和。

4、长度在float输入为10000时，对应WholeReduceSumImpl中的处理方法，在Counter模式下，采用WholeReduceSum指令，循环处理二维数据中的每一行，得到每一行的归约运行结果。

5、长度在float输入为20000时，对应BinaryReduceSumImpl中的处理方法，在Counter模式下，先将运算数据一分为二，使用Add指令将两部分数据相加，循环往复，最后一条WholeReduceSum指令得到归约的运行结果。此种操作方式，相比较WholeReduceSum单指令操作的方式，在数据量较大，循环次数较多的场景下，性能更优。

注意代码中使用了Counter模式。

## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Reduce</td></tr>
</tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape_range</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">(0,4096]float;(0,16384]half</td><td align="center">float;half</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">32</td><td align="center">float;half</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reduce_custom</td></tr>
</table>

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 算子工程介绍
其中，算子工程目录ReduceCustom包含算子的实现文件，如下所示:
```
├── ReduceCustom            // ReduceCustom自定义算子工程
│   ├── op_host             // host侧实现文件
│   └── op_kernel           // kernel侧实现文件
```
CANN软件包中提供了工程创建工具msOpGen，ReduceCustom算子工程可通过ReduceCustom.json自动创建，自定义算子工程具体请参考[Ascend C算子开发](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)>工程化算子开发>创建算子工程 章节。

创建完自定义算子工程后，开发者重点需要完成算子工程目录CustomOp下host和kernel的功能开发。为简化样例运行流程，本样例已在ReduceCustom目录准备好了必要的算子实现，install.sh脚本会自动将实现复制到CustomOp对应目录下，再编译算子。

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 调用msOpGen工具生成自定义算子工程；
- 完成算子host和kernel实现；
- 编译自定义算子工程生成自定义算子包；
- 安装自定义算子包到自定义算子库中；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 生成自定义算子工程，复制host和kernel实现并编译算子<a name="operatorcompile"></a>
  - 切换到msOpGen脚本install.sh所在目录
    ```bash
    # 若开发者以git命令行方式clone了master分支代码，并切换目录
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/14_reduce_frameworklaunch
    ```

  - 调用脚本，生成自定义算子工程，复制host和kernel实现并编译算子
    - 方式一：配置环境变量运行脚本   
      请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量命令。
      - 默认路径，root用户安装CANN软件包
        ```bash
        export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
        ```
      - 默认路径，非root用户安装CANN软件包
        ```bash
        export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
        ```
      - 指定路径install_path，安装CANN软件包
        ```bash
        export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
        ```
        运行install.sh脚本
        ```bash
        bash install.sh -v [SOC_VERSION]
        ```
    - 方式二：指定命令行安装路径来运行脚本
      ```bash
      bash install.sh -v [SOC_VERSION] -i [ASCEND_INSTALL_PATH]
      ```
    参数说明：
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
        - Atlas A2训练系列产品/Atlas 800I A2推理产品
    - ASCEND_INSTALL_PATH：CANN软件包安装路径

    脚本运行成功后，会在当前目录下创建CustomOp目录，编译完成后，会在CustomOp/build_out中，生成自定义算子安装包custom_opp_\<target os>_\<target architecture>.run，例如“custom_opp_ubuntu_x86_64.run”。

    备注：如果要使用dump调试功能，需要移除op_host内的Atlas 训练系列产品、Atlas 200/500 A2 推理产品的配置项。

### 3. 部署自定义算子包
- 部署自定义算子包前，请确保存在自定义算子包默认部署路径环境变量ASCEND_OPP_PATH
    ```bash
    echo $ASCEND_OPP_PATH
    # 输出示例 /usr/local/Ascend/ascend-toolkit/latest/opp

    # 若没有，则需导出CANN环境变量
    source [ASCEND_INSTALL_PATH]/bin/setenv.bash
    # 例如 source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
    ```
    参数说明：
    - ASCEND_INSTALL_PATH：CANN软件包安装路径，一般和上一步中指定的路径保持一致

- 在自定义算子安装包所在路径下，执行如下命令安装自定义算子包
    ```bash
    cd CustomOp/build_out
    ./custom_opp_<target os>_<target architecture>.run
    ```
  命令执行成功后，自定义算子包中的相关文件将部署至opp算子库环境变量ASCEND_OPP_PATH指向的的vendors/customize目录中。

### 4. 调用执行算子工程
- [aclnn调用ReduceCustom算子工程(代码简化)](./AclNNInvocationNaive/README.md)

## 更新说明
| 时间       | 更新事项                     |
| ---------- | ---------------------------- |
| 2024/09/14 | 新增ReduceCustom样例 |
| 2024/11/18 | 算子工程改写为由msOpGen生成 |
| 2025/07/07 | 增加两种归约操作样例 |
