---
type: experiment
created: 2026-09-12
status: reviewed-existing-evidence
tags: [experiment, causal, negative-result]
---
# 状态依赖确实暴露，但修正后尚未证明相对 Flash 的达标时间收益

## 问题与冻结合同

假设：PipeTGL 的跨 rank 状态交接暴露于关键路径；合并参数交接、主线程提交状态、延后采样可减少暴露开销。验收必须同时看因果合法性、质量达标与进程总时间，不能仅看 epoch。

本轮原始实验为 2026-09-11，整理日仅阅读报告和 JSON，并核对远端与本地文件 SHA256；没有重跑 GPU。来源 D1–D4 见[来源表](../10-Overview/Source-Map.md)。

- Wikipedia：9,228 节点、157,474 事件，每训练 epoch 110,232 事件；batch 600，1 层，fanout 10，dropout 0。
- 8 张 RTX PRO 6000；FP32、highest matmul precision；PipeTGL vendor commit `c79fbb6e0e39668dd0de541a2c2b0735140099dc`。修改版本以 D1–D4 和 REPRODUCE 指针为准，未冒充完整干净代码 commit。
- 对照：同轮 native、flash_ops、pipeflash_packed_causal；种子 2034–2038，无剔除；最多 40 epoch。
- 质量阈值：validation AP ≥ 0.97 连续两次。AP 是实现中的 batch/rank 均值，不能改称全局排序 AP。
- 主分母：进程到达质量目标的总时间；epoch 时间是辅助指标。完整命令、环境与预热协议保留在 D2，不在知识库重复维护第二套可执行脚本。
- 本平台 NCCL P2P/IB 禁用，使用 lo；不外推到 NVLink 或另一种网络环境。

## 结果

| 同轮方案 | epoch 秒 | 达标进程秒，均值±标准差 | 平均达标 epoch |
| --- | ---: | ---: | ---: |
| native | 0.742 | 32.35 ± 2.91 | 14.8 |
| flash_ops | 0.732 | 30.98 ± 2.93 | 13.8 |
| candidate | 0.652 | 30.19 ± 2.95 | 14.6 |

候选相对 flash 的配对几何平均加速为 1.026×，95% CI [0.985, 1.075]，未排除无收益；相对 native 为 1.072×，[1.034, 1.128]。epoch 加速约 1.12× 不等于进程也加速 1.12×；达标 epoch 波动影响最终结果。

单种子因果干预：状态路径增加 1 ms，epoch 增加 165.8 ms；同量 backward 延迟的差值为 −1.9 ms，未表现为同等暴露。原报告记录 5 epoch 共 920 次注入。此处支持“该平台的状态路径有暴露”，不支持“所有通信都不可隐藏”，也不是多种子干预结论。

状态等待在 native/flash/candidate 分别约 268.7/336.2/307.4 ms 每 rank-epoch；扣除相应覆盖后的 residual 为 9.2/9.5/4.4 ms。等待计数不是纯网络耗时，不能直接拿总等待预测可节省时间。

## 分项验证与撤销记录

- 原始错误候选发现未来状态读取 25,986 / 238,053 次；其约 1.18× 旧结论作废。另有 warmup 基线问题，不能与修正版混用。
- 修复顺序：上一轮接收结束后再读取 target CPU 状态；释放 successor 前保留 support memory/time/mailbox/time 私有快照；reset 后 barrier。
- 冻结状态资格检查：24 份 rank 权重逐位一致，821,715 次审计零未来读取；训练审计 3,347,718 次零未来读取。
- 这些检查支持已覆盖读取边界；不等于完整浮点训练轨迹逐位相同。编译、sanitizer、全设备压力测试没有在本次整理中执行。

## 解释、代价与下一步

状态提交和参数打包（24 次传输合为 1 次）改善调度，但最终质量到达时间收益被训练轮数与进程固定成本稀释。允许写成：**在此 8 卡合同下，候选改善 epoch；相对 flash_ops 的达标进程增益尚不显著。**

下一项应先补低成本时间分解和更稳定的达标统计设计；只有保持修正版因果 gate、同轮强基线、相同质量指标，才值得重新验收端到端收益。禁止恢复旧的含未来读取数字。
