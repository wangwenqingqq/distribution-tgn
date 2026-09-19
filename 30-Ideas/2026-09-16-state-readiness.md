---
type: idea
created: 2026-09-19
status: hypothesis
tags: [idea, pipetgl, lastfm, scheduling]
---
# 检查算子边界是否过度约束状态就绪

## 假设与机制
假设：PipeTGL 的批次／算子边界可能引入超过真实节点状态依赖所需的等待，并受 GPU／CPU 内存位置影响。候选方向是在保持训练语义下，按状态版本就绪释放工作，再结合数据位置安排执行。

规划窗口可以覆盖多个 batch，但不改变训练 batch、更新次数或状态可见性。依赖检查须包括目标和采样邻居、memory/mailbox、参数版本、梯度和优化器；目标节点不重叠不代表训练步骤可以乱序。细粒度判断就绪后仍需适度合并执行，防止小 kernel、小消息和 CPU 调度成本吞掉收益。

## 最接近的已有工作
- [DOLPHIN](../20-Literature/DOLPHIN.md)：跨批次窗口和依赖图计划。
- [REAL](../20-Literature/REAL.md)：复用、搬运与负载均衡的共同优化。
- [TempGNN](../20-Literature/TempGNN.md)：依赖准备与有序提交分离；推理机制不能直接外推训练。
- 全文级排重尚未完成，不能宣称以上组合具有新颖性。

## 最便宜的证伪测试
先核验 LastFM 数据，再检查完整读写依赖暴露的并行宽度，比较必要依赖和实际执行关键路径。若可消除等待的乐观收益已经很小，则不进入复杂调度实现。

## 对称判断
- 支持：大量已满足语义依赖的工作仍受粗粒度边界阻塞，而且这种等待落在全局关键路径。
- 反对：必要状态依赖已经高度串行，或新增调度／消息开销超过可消除等待。
- 证据不足：只看到低 GPU 利用率、NCCL 调用很长或某个 kernel 变快，没有跨 rank 端点与状态版本对应。

## 决策与下一步
后续统一 LastFM；目前仅讨论，未启动相关实验。先看 PipeTGL 自身 1/2/4/8 卡强扩展，保持工作量及语义，再单列训练达标指标。近线性是目标，不预设超线性；若出现超线性，应解释缓存、容量或工作量变化。

已有只读拓扑检查显示四个 NUMA 域；不同 GPU 到 CPU 内存及彼此之间的路径可能不同，实际传输代价尚未测定。主工具考虑 Nsight Systems 跨 rank 时间线，辅以 PyTorch Profiler／HTA；Flight Recorder 用于通信匹配与卡死诊断，NCU 用于已定位关键 kernel。参考 [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/) 和 [HTA](https://github.com/facebookresearch/HolisticTraceAnalysis)。
