---
type: decision
created: 2026-09-19
status: accepted
tags: [decision, correctness, negative-result]
---
# 撤销旧 1.18× 达标加速

## 结论
旧候选存在未来状态读取，1.18× 结果不能作为同语义训练加速。历史记录保留，后续仅使用共同状态协议修正后的确认集。

## 备选方案与取舍
仅保护支持节点快照仍不足：冻结权重下 memory/mailbox 最大绝对差约 1.15/1.94，尽管时间戳一致。最终共同要求目标 CPU gather 等待前序状态到达、支持状态复制为私有快照后再释放后继、共享 reset 后执行阶段 barrier。

## 支持、反对和缺失证据
- 旧候选训练诊断 238053 次支持读取中有 25986 次未来 mailbox 时间戳；原适配基线预热也有未来读取。
- 修正后三种实现的冻结权重完整状态逐位一致；不宣称完整训练浮点轨迹一致。
- 证据：归档 `analysis/confirmation_status.json`、`analysis/support_timestamp_audit_summary.json`、`analysis/prefix_frozen_v2_qualification.json`。

## 影响范围与回退方式
不删除旧日志、源文件或结果；归档区分有效确认与历史探索，最终主张见[主张—证据表](../10-Overview/Claim-Evidence.md)。

## 重新考虑条件
任何更早状态释放方案须重新验证读取版本、完整状态、梯度／参数语义和固定工作量，再独立测量性能。

## 下一步
将合法状态就绪条件作为后续调度的前置合同，见[研究假设](2026-09-16-state-readiness.md)。
