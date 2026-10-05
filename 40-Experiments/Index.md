# 实验索引

| ID | 问题 | 固定对比合同 | 基线 | 结果与原始证据 | 决策 |
| --- | --- | --- | --- | --- | --- |
| final_confirm_p8 | 候选能否缩短完整达标 | Wiki、八卡、5 新种子、AP≥0.97 连续两轮 | 同状态协议原生／直接融合 | [正式确认](2026-09-11-final-confirmation.md) | 单轮收益成立，额外达标收益尚不稳定 |
| prefix_gpu_delay | 状态链是否限制吞吐 | 同协议、单种子五轮、CUDA 实测延迟 | 零延迟／backward 后延迟 | [因果诊断](2026-09-11-state-critical-path.md) | 支持当前配置状态路径敏感性 |
| lastfm_short_p2 | 组合能否减少固定工作量全程用时 | 双卡、5 方案、3 seed、每次 3 epoch；统一状态协议 | 原生 / loss-deferred / state-inline / direct Flash / 组合 | [短训练筛选](2026-10-05-lastfm-short-training.md) | 组合用时减少 20.1%；精度等价未成立 |
| lastfm_thread_warp | 算子以下的线程与发射开销 | 双 rank host/Nsight；真实 sampler replay/NCU | 未开 profiler 的计时与独立诊断 | [两层 breakdown](2026-10-05-lastfm-thread-breakdown.md) | 优先定位 DGL 构造、短 kernel 提交和状态准备；不宣称气泡可全部消除 |
| lastfm_parameter_transport | 合并消息能否转化为训练收益 | 同实际 payload、同/跨 NUMA；补充保留 native attention 的消融 | 24 张量消息 / 单张量；native 参数 arena | [通信与排除记录](2026-10-05-lastfm-parameter-transport.md) | 微实验有信号；受资源竞争计时剔除，有效性能种子待补 |

使用 [实验模板](../90-Templates/Experiment.md)。只收录轻量摘要与证据指针，不上传原始训练数据或权重。
