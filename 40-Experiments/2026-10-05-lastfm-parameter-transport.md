---
type: experiment
created: 2026-10-05
status: correctness-passed-performance-pending
tags: [experiment, lastfm, communication]
---
# 参数消息合并的微实验信号尚未转化为有效训练消融

**结论：相同实际 payload 的参数消息合并显著降低独立通信完成耗时；但保留 native attention 的补充性能消融因资源竞争尚未完成，不能把微实验比例当作端到端收益。**

## 问题与冻结合同

从 LastFM 原生运行记录真实 tensor shapes，参数为 24 个张量，共 891,604 B；状态消息有 10/50/90 分位 payload，为 217,440–306,832 B。比较同 NUMA GPU 对与跨 NUMA GPU 对，每 case 5 次 warm-up、20 次测量，两个方向交替；CUDA completion 同步，控制 barrier 在计时外。

两端测到的是同一交换，取每次两端 host completion 的最大值，不把两个 rank 当成 40 个独立样本。耗时包括 API、对端到达和 CUDA 完成，不是纯线速。

## 执行与原始证据

- [双方样本、NUMA 与 payload 结果](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/analysis/communication_breakdown.json)
- [native parameter-arena 冻结资格](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/analysis/parameter_arena_qualification.json)
- [受竞争计时的明确排除](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/analysis/excluded_parameter_trial.json)
- [恢复脚本](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/src/run_parameter_ablation.py)与[归档运行条件](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/REPRODUCE.md)

## 分项验证

- 编译：Python 语法与源文件 hash 通过，没有新 CUDA kernel。
- 正确性：所有微实验接收值校验通过；默认 NCCL 仅完成 payload 校验，没有整轮训练资格。native parameter-arena 冻结状态与原生逐位同，future-read 时间戳审计为零；seed 4101 的最终参数、状态和 AP 曲线与配对原生逐位同。
- Sanitizer / stress：未执行。
- Kernel-only：没有独立 kernel 加速主张。
- 端到端：native-arena 的第一条计时受竞争而排除；第二条被空闲检查跳过、第三条未启动。三个有效性能 seed 均待补。

## 结果、解释和代价

| payload / 方法 | 同 NUMA completion / ms | 跨 NUMA / ms | 同 NUMA 默认 NCCL / ms |
| --- | ---: | ---: | ---: |
| 参数 891,604 B，24 条消息 | 0.7358 | 0.7488 | 0.7410 |
| 相同参数 bytes，1 条消息 | 0.0835 | 0.1082 | 0.0838 |

同 NUMA 约 8.8 倍的独立完成耗时差异，提示 host/API/message-count 成本值得验证。跨 NUMA 合并参数略慢，状态消息的差异较小；这里只有一个同域/跨域 GPU 对和 20 次交换，不能推成普遍通信规律。

补充实现保留 native attention、原生 state-send 线程和 baseline 顺序，仅绑定 contiguous parameter arena、将 clone/发送从列表改为一个张量，保留 Parameter 身份和 Adam 布局。与[组合方案](2026-10-05-lastfm-short-training.md)相比，它用于去除 Flash / state-first 的机制混杂。

第一条补充运行结束时，选定 GPU 的显存 / 计算进程记录显示其他任务开始占用；不能确定整条计时无干扰，因此整次性能排除，只保留正确性证据。后续被 guard 挡下，没有终止其他任务，也没有将跳过记成通过。

## 否定证据与重新考虑条件

当前没有有效 native-arena 训练加速比，不把受竞争的慢计时当成负性能结论，也不凭微实验声称组合收益主要来自参数传输。重新考虑条件是同环境空闲窗口、三个有效配对 seed、前后资源记录及训练一致性检查；若仍无收益，进一步检查 state-first/inline 交互，避免只优化非关键路径消息。

## 允许写进论文的表述

固定 payload 的参数消息合并，在既定同 NUMA 双卡微实验中降低独立 host/CUDA completion 耗时；其训练端到端收益尚未确认。

## 下一步

恢复脚本在发现跳过或受竞争目录时写新版本，保留原始记录。正式续跑需真实空闲 GPU 和完整原环境基线 / checkpoint；这次 GitHub 发布仅归档与离线核验，不启动新服务器实验。
