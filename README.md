---
type: project-home
project: "Distribution-TGN"
tags: [research]
---
# Distribution-TGN

**一个课题，一个仓库，一份可追溯的知识记录。**

> 更新至 2026-10-05：LastFM 双卡已完成 15 次无 profiler 短训练，以及 CPU 线程 / GPU warp 两层诊断。组合方案固定工作量的完整进程用时减少 20.1%，但 AP 和训练轨迹不同，尚不支持同精度达标加速。Wikipedia 八卡历史证据保留。

最新成果：[LastFM 短训练](40-Experiments/2026-10-05-lastfm-short-training.md)、[两层 breakdown](40-Experiments/2026-10-05-lastfm-thread-breakdown.md)、[参数通信消融与排除记录](40-Experiments/2026-10-05-lastfm-parameter-transport.md)。

历史成果：[Wikipedia 最终对比](40-Experiments/2026-09-11-final-confirmation.md)、[状态路径干预](40-Experiments/2026-09-11-state-critical-path.md)、[撤销旧结果与正确性修复](30-Ideas/2026-09-11-withdraw-invalid-speedup.md)、[状态调度假设](30-Ideas/2026-09-16-state-readiness.md)。代码和轻量实验证据独立保存在 [tgn-pipeflash-artifact](https://github.com/wangwenqingqq/tgn-pipeflash-artifact)。

## 开始使用
- [课题卡片](10-Overview/Research-Brief.md)：问题、机制、边界与下一步
- [主线与资料来源](10-Overview/Source-Map.md)：代码、论文和证据在哪里
- [主张—证据表](10-Overview/Claim-Evidence.md)：哪些结论能说，哪些还不能说
- [使用与同步](99-System/Workflow.md)：日常记录、同步和冲突处理

## 工作区
| 区域 | 内容 |
| --- | --- |
| [收件箱](00-Inbox/Inbox.md) | 临时想法、讨论摘记、待整理材料 |
| [文献](20-Literature/Index.md) | 原文链接、机制、差异与证据 |
| [想法与决策](30-Ideas/Index.md) | 假设、证伪、选择和否决理由 |
| [实验](40-Experiments/Index.md) | 固定合同、原始证据指针、结果解释 |
| [写作](50-Writing/Index.md) | 提纲、论证、图表和审稿反馈 |
| [讨论与会议](60-Meetings/Index.md) | 结论、待办和责任边界 |
| [研究日志](70-Journal/Index.md) | 每天的进展、失败和下一步 |

## 仓库
[wangwenqingqq/distribution-tgn](https://github.com/wangwenqingqq/distribution-tgn) · 默认保持私有 · 不保存密钥、数据集或模型权重。
