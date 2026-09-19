---
type: literature
created: 2026-09-19
status: abstract-only
tags: [literature, distributed]
---
# REAL

## 一句话结论
复用、数据搬运和负载均衡应共同考虑，平均任务数量不足以衡量分布式调度质量。

## 原始来源
- REAL: Efficient Distributed Training of Dynamic GNNs with Reuse-Aware Load Balancing Scheduling，Zihao Fan、Yunzhuo Liu、Tian Guo、Bo Jiang，SC26。
- [官方摘要](https://sc26.conference-program.com/presentation/?id=pap345&sess=sess196)，2026-09-16 核验；未阅读全文。

## 问题、机制和适用边界
摘要描述快照分组、增量聚合，以及 ILP 和贪心两类调度器，以兼顾数据复用、搬运和负载均衡。具体增量聚合的正确性条件尚未从全文核验。

## 与本课题的关系
- 相同点：跨设备调度和数据移动成本。
- 实质区别：快照型 DGNN 的复用不能直接等同于连续时间 TGN 的状态复用；相同节点 ID 不保证相同 memory、mailbox 或参数版本。
- 是否覆盖核心创新：未核验；拓扑与数据位置只是可能的迁移方向，不能宣称原文未覆盖。

## 支持与反对证据
摘要支持多目标协同调度的动机；没有 LastFM 上可以安全复用多少状态的证据。

## 下一步与相关笔记
阅读全文核验适用模型、训练语义和通信模型；关联[状态就绪假设](../30-Ideas/2026-09-16-state-readiness.md)。
