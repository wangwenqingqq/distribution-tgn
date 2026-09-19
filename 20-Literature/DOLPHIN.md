---
type: literature
created: 2026-09-19
status: abstract-only
tags: [literature, scheduling]
---
# DOLPHIN

## 一句话结论
跨批次规划窗口可以用于协调内存层级搬运和依赖执行；是否适合本课题须由实际 I/O 与空泡证据决定。

## 原始来源
- DOLPHIN: Scalable Disk–RAM–GPU Pipelined Training for Massive Temporal GNNs，Zezhong Ding 等，SC26。
- [官方摘要](https://sc26.conference-program.com/presentation/?id=pap204&sess=sess359)，2026-09-16 核验；未阅读全文。

## 问题、机制和适用边界
摘要描述将 batch 组成 chunk，以特征窗口安排 Disk-to-RAM 数据供应，再用依赖图制定并行执行计划，处理大图内存成本和批次边界空泡。具体语义、实现和各项消融尚未知。

## 与本课题的关系
- 相同点：跨批次数据准备和流水线依赖。
- 实质区别：我们当前证据来自内存型八卡配置；LastFM 是否存在磁盘瓶颈尚未知。规划窗口不能等同于改变训练 batch。
- 是否覆盖核心创新：未完成全文级核验，不能把 chunk／依赖图调度单独作为创新。

## 支持与反对证据
摘要支持其机制方向；不据摘要最大加速数与我们的结果做直接性能比较。

## 下一步与相关笔记
全文可得后核验状态和参数语义、设备范围及 baseline；关联[状态就绪假设](../30-Ideas/2026-09-16-state-readiness.md)。
