# 主张—证据表

C1–C4 为 Wikipedia 八卡历史结论；C6–C8 为新完成的 LastFM 双卡第一轮实验。新 chunk/packet 调度和 scaling 仍处于假设阶段，两个数据集的计时与精度目标不可混用。

| ID | 主张 | 证据与版本 | 对比合同 | 支持 / 反对 / 未知 | 允许表述 | 下一步 |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 状态链限制当前配置吞吐 | [CUDA 延迟干预](../40-Experiments/2026-09-11-state-critical-path.md) | 同一正确状态协议；固定五轮，单种子 | 支持，有范围限制 | 状态更新前每批约 1 ms 的 GPU 延迟使单轮增加 165.8 ms；backward 后相同延迟未显著增加单轮时间 | LastFM 上重新检查，不能直接外推 |
| C2 | 候选提升单轮吞吐 | [最终确认](../40-Experiments/2026-09-11-final-confirmation.md) | 新种子五组；原生、直接融合、候选共用状态约束 | 支持 | 候选相对直接融合单轮加速 1.123× | 区分单轮与完整达标时间 |
| C3 | 候选稳定加快完整达标 | [最终确认](../40-Experiments/2026-09-11-final-confirmation.md) | 原 evaluator AP≥0.97 连续两轮；最多 40 轮 | 证据不足 | 相对直接融合配对几何平均 1.026×，95% 区间 [0.985,1.075]，跨 1 | 不宣称稳定额外端到端加速 |
| C4 | 旧 1.18× 是有效收益 | [撤销记录](../30-Ideas/2026-09-11-withdraw-invalid-speedup.md) | 旧实现存在未来 mailbox 读取 | 反对，已撤销 | 仅作为失败诊断保存 | 不混入有效结果 |
| C5 | 细粒度状态调度改善 LastFM 扩展性 | [方向假设](../30-Ideas/2026-09-16-state-readiness.md)、[两层诊断](../40-Experiments/2026-10-05-lastfm-thread-breakdown.md) | 双卡已观测，尚未实现 chunk/packet 与 scaling | 未知 | 相邻 batch 均有状态依赖，准备/有序提交的细分仍值得验证 | 审计 producer-version 和参数 staleness；不据节点代理 DAG 宣称整批串行 |
| C6 | 组合降低 LastFM 固定工作量全程用时 | [15 次配对短训练](../40-Experiments/2026-10-05-lastfm-short-training.md) | 3 seed × 3 epoch，完整训练 905,172 事件/轮；无 profiler | 支持，有范围限制 | 完整运行 78.26→62.53 s，用时减少 20.1%；配对几何加速 1.252×，三 seed 筛选区间 [1.213,1.297] | Flash/组合 AP 与轨迹不同，验证同精度和 forward/gradient 合同 |
| C7 | 每 batch loss 读取 / state-inline 是主要收益来源 | [配对消融与线程 spans](../40-Experiments/2026-10-05-lastfm-thread-breakdown.md) | 两消融训练参数/状态与原生逐位同；三 seed | 当前不支持 | loss scalar wall span 约 0.18%；两个完整进程收益区间均跨 1 | 优先看 DGL 构造和短 kernel 发射；保留负结果 |
| C8 | 参数合并直接带来训练加速 | [实际 payload 微实验和补充资格](../40-Experiments/2026-10-05-lastfm-parameter-transport.md) | 同 891,604 B，24 条 vs 1 条消息；native arena 有效计时待补 | 训练收益未知 | 同 NUMA 独立 max-endpoint completion 为 0.7358 / 0.0835 ms；不能直接换成端到端收益 | 空闲窗口补三个有效 seed；受干扰计时不作结论 |

编译成功、正确性、sanitizer、kernel-only、端到端性能分别登记，不相互替代。
