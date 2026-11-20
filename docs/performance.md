# 性能说明

这里分别比较完整评分的 CPU/GPU 耗时，以及两种 GPU IC 实现的耗时。所有对照都计算 PearsonIC 和 RankIC，并使用相同输入。

## 测试设置

| 项目 | 配置 |
| --- | --- |
| CPU / 内存 | Intel Core i9-14900K，24 核 / 32 线程，32 GB |
| GPU | NVIDIA GeForce RTX 5070 Ti，16 GB |
| 系统 | Ubuntu 24.04.3，NVIDIA 驱动 595.99.02 |
| 工具链 | Python 3.12.3，NumPy 2.5.4，GCC 13.3.0，CMake 3.28.3，CUDA 12.8 |

使用 256 个表达式、400 只仿真标的、每条序列 23400 点，预测未来 30 个采样点。固定种群，只评分一次；预热后交替测量五轮，取中位数。环境配置和运行入口见 [README](../README.md#运行)。

标签是用于评价因子的未来收益率。标签能否参与评分由布尔标记 `eligible` 表示，为真的比例就是**标签有效率**；再排除因子或标签中的 NaN、Inf，得到实际参与 IC 计算的**联合有效率**。

真实市场中的缺失数据、预测期限、因子预热、标的和时段都会影响可用样本数，因此测试覆盖不同有效率。这里仅改变仿真标签的评分标记，保持行情和因子输入相同。三档标签有效率为 99.86%、50.00%、30.47%，对应联合有效率 93.71%、46.92%、28.60%。

## 完整评分：CPU 与 GPU

计时包括数据准备、因子计算、两项 IC、指标汇总和结果回传。GPU 使用融合 kernel 同时计算两项 IC。

| 标签有效率 | C++/OpenMP | CUDA | CPU / CUDA 加速比 |
| --- | ---: | ---: | ---: |
| 99.86% | 25.389 s | 1.501 s | 16.91× |
| 50.00% | 22.487 s | 1.420 s | 15.83× |
| 30.47% | 17.159 s | 1.394 s | 12.31× |

本次完整评分的 OpenMP 配置为：CPU 后端 24 线程，CUDA 路径的主机端 8 线程。后者用于数据准备等 CPU 工作，因子与 IC 计算由 GPU 执行，设备线程另行配置。8 是测量脚本的预设值，未验证为最优；上表反映这两套配置的结果。

## GPU IC 计算：融合与独立实现

因子、标签已在显存中，只测 PearsonIC + RankIC，排除分配、传输和因子计算。使用未插桩 CUDA Events，各批同轮耗时相加后取五轮中位数。加速比为“CUB 排序 + 独立 Pearson”的耗时除以融合 kernel 耗时。

| 标签有效率 | CUB 排序 + 独立 Pearson | 融合 kernel | 加速比 |
| --- | ---: | ---: | ---: |
| 99.86% | 397.027 ms | 287.163 ms | 1.38× |
| 50.00% | 382.865 ms | 193.555 ms | 1.98× |
| 30.47% | 389.631 ms | 160.386 ms | 2.43× |

CUB 对照使用独立 FP64 Pearson，融合核使用缩放、均值补偿的 FP32 Pearson；两者都按下面的精度标准检查。

融合核让一个 CTA 处理一个因子、一只标的和一个期限：读取输入并计算 Pearson → 压紧联合有效点 → X 排序求秩 → Y 排序求秩 → INT64 统计得到 RankIC。

- 两项 IC 共用输入读取；无效点不进入后续排序，因此稀疏输入的工作量更少。
- 短序列的完整 key 保存在寄存器；长序列拆分 key，并分阶段复用 Shared 和预分配工作区，控制寄存器压力。
- 常数因子的 RankIC 已确定为 NaN，整个 CTA 可以跳过 Y 排序。Pearson 在此之前已完成。

## Pearson 精度

以相同 FP32 输入上的 C++ FP64 结果为参考，检查 `abs(GPU-CPU) <= 3e-7 + 2e-5 × abs(CPU)`。三档仿真共 307200 对输入全部通过，RankIC 逐值一致，NaN 分类一致，Inf 与范围检查均通过。

| 标签有效率 | 最大绝对误差 | 最大相对误差 |
| --- | ---: | ---: |
| 99.86% | 1.60e-7 | 0.176% |
| 50.00% | 2.01e-7 | 8.28% |
| 30.47% | 1.75e-7 | 5.67% |

相对误差包含所有 CPU 非零结果。8.28% 对应 CPU=-2.87014e-8、GPU=-2.63246e-8，绝对误差为 2.38e-9；该样本的相关系数分母约为 27.19，差异来自协方差抵消后的舍入。它通过联合容差，但依赖这一级别数值差异或符号的业务规则仍需单独检查。

Pearson 使用缩放中心矩与 Chan 归约，将均值及其微小残差分开保存。分母不加 epsilon，近零结果不截成零；有效点不足三个或方差为零时返回 NaN。三档完整评分的 Pearson 均值最大绝对差为 3.15e-9，RankIC 汇总、有效计数和候选顺序与 CPU 一致。

已有 455 项测试、3 项 CTest，以及 memcheck、synccheck、racecheck 检查通过，覆盖常数、偏置、极值、次正规数、NaN/Inf、稀疏掩码及容量边界。

## Zero-Spill 与性能分析

另外取 GP 管线捕获的 256 个因子 × 32 只标的、每条序列 23400 点，比较生产融合核与 Zero-Spill 对照。两者算法和输出相同；对照将部分 Y Payload 和 RankX2 暂存到可复用的 Global 工作区，以消除编译器溢出。

| 项目 | 融合 kernel | Zero-Spill 对照 |
| --- | ---: | ---: |
| 每线程寄存器 | 64 | 64 |
| PTXAS spill 写 / 读 | 60 / 52 B | 0 / 0 B |
| 五轮配对 Event 中位数 | 9.100 ms | 9.720 ms |

全部 8192 对输入的两项 IC 逐值一致，NaN 分类一致。Zero-Spill 五轮均更慢，耗时增加约 **6.8%**，因此生产保留耗时更低的融合核。

NCU 采样这组输入的前 512 对：Zero-Spill 的 Local 指令降为零，但 Global 读写指令增加，Long Scoreboard 从 2.35 升至 3.69 cycles / issued instruction；Shared 占用和可驻留 CTA 数不变。寄存器溢出被显式存取替代，等待开销反而增加，零溢出并不保证更快。

Nsys 采样 256 个表达式、32 只仿真标的一次评分：融合 IC 占 GPU kernel 时间约 32.9%，EMA 占 17.3%，滚动双变量统计占 33.0%。Profiler 用于分析瓶颈，上面的正式耗时均来自未插桩测量。

对照源码：[CUB 排序 + 独立 Pearson](../cuda/reference/cub_pair.cu)、[Zero-Spill](../cuda/reference/zero_spill_pair.cu)。

<details>
<summary>构建对照库</summary>

先按 [README](../README.md#运行) 完成生产构建，再运行：

```bash
nvcc -O2 -g -DNDEBUG -std=c++17 -arch=sm_120 -lineinfo -Xptxas=-v \
  --shared --cudart shared -Xcompiler=-fPIC -Icpp/include \
  cuda/reference/cub_pair.cu -o build-study/cub_pair.so
nvcc -O2 -g -DNDEBUG -std=c++17 -arch=sm_120 -lineinfo -Xptxas=-v \
  --shared --cudart shared -Xcompiler=-fPIC -Icpp/include \
  cuda/reference/zero_spill_pair.cu -o build-study/zero_spill_pair.so
```

CUB 对照用于 T=23400，入口为 `prepare_ic_pair`、`launch_ic_pair`；Zero-Spill 对照支持长序列融合核范围，入口为 `launch_zero_spill_pair`。这些对照不进入生产调度。

</details>
