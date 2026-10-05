---
type: experiment
created: 2026-10-05
status: diagnostic-complete
tags: [experiment, lastfm, profiling]
---
# LastFM 两层诊断将重点转向图块准备与短 kernel 提交

**结论：当前证据提示继续检查 DGL 构造、CUDA 提交与版本感知状态准备；每 batch loss 读取和单独 state-inline 不是已确认的主要收益来源。** 这是定位线索，尚未通过新的因果干预或实现消融证明这些开销全部在关键路径上。

## 问题与冻结合同

沿用[短训练合同](2026-10-05-lastfm-short-training.md)，以原生方案诊断：未开 Nsight 的线程 spans；两 rank 同时采集的 Nsight CUDA/NVTX/OSRT/python-gil/context-switch；独立真实 sampler 输入 replay 与 NCU。诊断与正式性能分开。

## 执行与原始证据

- [独占主线程 spans、on-core、GIL、OSRT 和 GPU 活动](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/analysis/thread_gpu_breakdown.json)
- [双卡时间线](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/analysis/thread_gpu_timeline.png)
- [采样 C++ / DGL 拆分](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/analysis/sampler_component_breakdown.json)
- [NCU warp 指标](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/analysis/ncu_sampler_b16_selected.json)与[节点状态依赖代理](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/analysis/dependency_breakdown.json)

Nsight v1 没有触发 capture，不作成功 GPU 采集；v2 通过 `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` 捕获两个 rank，按 host/NVTX 对齐分析稳态窗口。单 rank epoch 标记不等于只分析单 rank，GPU 活动与线程按真实进程分别归属。

## 分项验证

- Python 语法和导出 hash：发布核验通过；未重新编译底层 kernel。
- sampler：早期 / 中段 / 尾部真实输入各 50 次重放结果一致；batch16 输出 hash 与未插桩及 NCU replay 相同。
- Sanitizer / stress：未执行。
- Kernel-only：NCU 15-pass replay/cache-flush 仅作诊断，不是正式 kernel speedup。
- 端到端：仅引用关闭 profiler 的独立 15 次运行，不混入 trace 墙钟。

## 结果、解释和代价

未开 Nsight 的独占主线程墙钟中，sampling 包装约 9%，support snapshot/compute 约 10%，forward/backward dispatch 合计约 39%–43%，loss scalar read 约 0.17%–0.18%。每 batch 新建参数/状态发送线程，进入首条 Python 指令的中位延迟约 0.15 ms；线程区间重叠，不能加到主线程总时间。

真实 batch16 的 sampler 调用中位数 1.11 ms，C++ 部分 0.42 ms、DGL 包装 0.67 ms、最后 synchronize 0.024 ms。C++ 部分还包含 CUDA 编排、采样和结果拷贝，不能当成纯 kernel 时间。

Nsight 中每 rank 约 237,000 个计算 kernel，约 78% ≤5 μs。sampler NCU 为 71 blocks / 188 SM，eligible warps 约 0.0073，long scoreboard 为主要 stall 来源。recent kernel 发射预留 48 KiB dynamic shared memory，源码没有使用该 shared 区；尚未修改或实测收益。小 grid 和包装成本提示，单看理论 occupancy 可能不能解释端到端。

Nsight 每 rank 的显式 profiler overhead 联合区间约 6 s / 24.67 s，不能简单扣除；绝对占比不代表未插桩训练。GPU 无记录活动不等于 SM idle，NCCL kernel 持续时间含 peer wait，主线程 dispatch wall 不等于 on-core。调度记录状态为 Unknown，无法精确区分 blocked 与 runnable。

## 必要依赖与研究假设

1,508 对相邻 batch 都有节点状态 RAW/WAW 重叠。这个 batch DAG 只描述状态前沿代理：参数接收在 backward 后、optimizer.step 前，不能据此宣称整个训练 batch 必须串行。

按 support node 的最近 target-writing-batch 版本计数，chunk=4/8/16 的状态记录准备复用乐观上界约 42.0%/51.5%/56.8%。该上界不包括 learned activations；mail/producer-version 与参数版本须进一步审计。仅按 node ID 会高估复用。原生 prepare_input 已在 memory updater 前 unique，约 88.5% 重复采样槽位不是剩余 GRU 冗余。

与 [DOLPHIN、REAL、TempGNN 摘要](../20-Literature/Index.md)对应的后续推断是：按 chunk 准备采样拓扑、版本感知 support-state packet、保留有序状态/参数提交，减少图块和 CUDA 发射开销。没有实现新调度、没有全文级创新性排重；LastFM 很小且 feature 为零，不预设 disk–RAM 特征搬运为主瓶颈。

## 否定证据与重新考虑条件

loss-deferred 和 state-inline 的正式收益区间都跨 1。不能按某个等待 span 或 NCU stall 比例推导“气泡可完全消除”。只有在完整 producer-version 合同下，用真实关键路径干预和端到端消融确认，才将诊断线索升级为机制结论。

## 允许写进论文的表述

报告既定 LastFM 配置下独立测得的主线程包装成本、短 kernel 分布和 sampler warp 指标；明确 profiler overhead、区间重叠、状态 DAG 代理和复用上界的限制，不宣称未实现的 packet/chunk 收益或八卡线性扩展。

## 下一步

先完成[参数传输消融](2026-10-05-lastfm-parameter-transport.md)，再按关键路径验证选择 DGL/采样准备或 packet 原型。完整 scaling 与精度合同仍待建立。
