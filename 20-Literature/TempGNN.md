---
type: literature
created: 2026-09-19
status: abstract-only
tags: [literature, temporal-dependency]
---
# TempGNN

## 一句话结论
分离依赖准备与有序状态提交，提示我们检查粗粒度完成边界是否掩盖合法并行空间。

## 原始来源
- TempGNN: An Efficient Real-Time TGNN Inference Accelerator via Target-centric Dependency Packets，Hui Yu 等，SC26。
- [官方摘要](https://sc26.conference-program.com/presentation/?id=pap142&sess=sess217)，2026-09-16 核验；未阅读全文。

## 问题、机制和适用边界
摘要描述以目标顶点的最小有效时序依赖组织 TDP，分离依赖准备与有序提交，并利用目标间部分重叠。实现为 Xilinx Alveo U280 FPGA，任务是 TGNN 推理。

## 与本课题的关系
- 相同点：状态有序性和细粒度依赖。
- 实质区别：训练还包含参数版本、反向传播、梯度与优化器更新；将推理依赖包搬到 GPU 不自动成为合法训练调度。
- 是否覆盖核心创新：未完成全文级核验；“依赖准备与提交分离”已经是相关机制，不能当作首次提出。

## 支持与反对证据
摘要支持这种执行抽象在其推理范围内的动机；没有证明它在我们的多 GPU 训练配置中有收益。

## 下一步与相关笔记
核验完整读写依赖和语义边界，关联[状态就绪假设](../30-Ideas/2026-09-16-state-readiness.md)及[错误状态释放记录](../30-Ideas/2026-09-11-withdraw-invalid-speedup.md)。
