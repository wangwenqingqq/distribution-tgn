# 课题卡片

截至 2026-10-06；Wikipedia 为八卡历史研究，LastFM 已完成双卡第一轮短训练、新参数传输配对和局部端点版本审计。chunk/packet、完整精度和 1/2/4/8 卡 scaling 仍未实施。

## 一句话主张
PipeTGL 的优化需要同时拆解主机线程、CUDA 提交与状态交接。LastFM 上组合方案降低固定工作量的完整运行时间，但尚未证明更快达到相同精度；单独融合算子没有带来收益。

## 问题与边界
- 问题定义：算子级流水线在主机提交、CUDA stream、状态版本和跨卡通信层面的必要依赖与额外等待。
- 任务 / 输入输出 / 正确性合同：TGN 链路预测训练；保留批次处理量、状态可见性、参数版本和优化器更新语义。冻结权重状态一致不等价于完整训练轨迹逐位一致。
- 范围：LastFM 完整数据与历史数据逐行核验通过，双卡 CPU 线程 / GPU warp 诊断和 15 次短训练已完成。正式 1/2/4/8 卡扩展性尚未测量，不预设磁盘瓶颈或线性 / 超线性加速。

## 创新性检查
- 最近的相关工作：PipeTGL、FlashTGN；DOLPHIN、REAL、TempGNN 的公开摘要已核验，见[文献索引](../20-Literature/Index.md)。
- 核心问题、机制、贡献是否已被覆盖：尚未完成全文级排重，不能把 chunk、复用或细粒度依赖包本身作为新贡献。
- 最便宜的证伪测试：参数传输的三个新配对 seed 已完成，完整用时减少 2.78%，低于 3% 筛选门槛。下一步以保持原生参数版本的接收调度干预验证关键路径等待。局部 CPU 字段和消息版本审计已建立，尚未覆盖所有 target gather 与 GPU consumer 边界。

## 当前证据与下一步
- LastFM 证据：5 方案 × 3 seed × 3 epoch；组合完整进程从 78.26 s 降到 62.53 s，用时减少 20.1%；五方案冻结状态逐位一致、future-read 时间戳审计为零。见[短训练](../40-Experiments/2026-10-05-lastfm-short-training.md)。
- 两层诊断：采样包装占主线程约 9%，中段调用的 DGL 构造约 0.67 ms / C++ sampler 0.42 ms；Nsight 中约 78% 的计算 kernel ≤5 μs。Nsight 有明显自身开销，不能用其绝对时间替代正式计时。见[breakdown](../40-Experiments/2026-10-05-lastfm-thread-breakdown.md)。
- 新配对与版本证据：参数 arena 的六次新进程均通过资源和逐位正确性检查，完整用时几何平均减少 2.78%。四个主要窗口的 256 条消息内容一致，128 个 batch 的 forward 与 optimizer 输入版本不同；完整 epoch 1,509 次 CPU 写入位置已记录。见[补充实验](../40-Experiments/2026-10-06-lastfm-critical-path.md)。
- 历史证据：Wikipedia 的 15 次正式达标实验、状态逐位资格和 GPU 延迟因果干预仍仅适用于原配置。
- 关键不确定性：LastFM Flash 与组合的 AP / 训练轨迹不同；20.1% 仅为固定工作量用时减少。短训练三 seed 不足以确认同精度达标收益，组合内部机制也未完全拆开。
- 下一项可交付工作：区分消息 ready-to-post 与 receiver-ready-to-consumer，冻结参数版本后做接收时机／持久 worker 干预；确认可消除的关键路径等待，再决定 chunk/packet 原型和精度验证。

## 导航
- [资料来源](Source-Map.md)
- [主张与证据](Claim-Evidence.md)
