---
type: experiment
created: 2026-10-06
status: completed
tags: [experiment, lastfm, pipetgl]
---
# LastFM 参数传输消融与端点版本诊断

本轮完成 E0 证据核验、E1 三 seed 新配对和 E2 端点诊断。保留 native attention 的参数 arena 将完整进程用时几何平均减少 2.78%，低于预设 3% 筛选门槛。消息与版本审计确认了原生参数滞后，并暴露消费者较晚进入接收调用的时间间隔；其可消除比例仍需调度干预验证。

## 问题与冻结合同

- 假设与接受标准：仅将 24 个参数张量放入连续 arena、合并为一条张量消息，能否在保留训练轨迹的条件下缩短完整运行。正确性与资源资格分别判定；性能门槛为完整进程用时减少 3%。
- 工作量与语义：LastFM 1,293,103 条事件、1,980 个节点，每遍完整处理 905,172 条训练事件；batch 600、单层、fanout 10、维度 100、dropout 0。保留完整 optimizer warm-up、native attention、每 rank Adam、参数滞后和状态线程；双方共同采用确定性 mailbox、完整末批、prefix snapshot 与 Gloo 控制屏障。
- 对照与顺序：seed 4101 为 native→arena，4102 为 arena→native，4103 为 native→arena；每臂独立启动、完整预热后训练 3 epoch 并验证。六次运行均为新测量。
- 环境：八卡服务器上的空闲 GPU 4/5，均为 NUMA 2，型号 NVIDIA RTX PRO 6000 Blackwell Server Edition；PyTorch 2.11.0+cu130、CUDA build 13.0、driver 590.48.01。沿用禁用 NCCL P2P/IB 的配置，本轮结果限于双卡该配置。
- 代码：基于统一仓库 `f9998a2a1e6645af050c826fe950cbdd97884cbe`；正式运行前冻结所有 E1 源文件 SHA256，见[合同](../artifacts/tgn_lastfm_critical_20261006/plans/critical_20261006_s1_contract.json)。
- 主指标：guard 启动到子进程退出的完整用时，含启动、预热、训练、验证和产物保存。次指标为三个正式 epoch 的跨 rank 全局 wall 区间之和。rank 和 epoch 不作为独立样本。

## 执行与原始证据

E0 核验原归档 22 个源文件及三份数据哈希，确认原 Nsight SQLite、线程事件、依赖记录和 checkpoint 存在。历史归档的 142 个文件哈希与统计复算通过；新 native 的三个 seed 与历史 native 的模型、内存及 AP 逐位一致，仅用来确认来源延续。

E1 的资格运行采用完整预热加 1 epoch，额外验证两个 rank 的 Adam 状态、参数归属和 CPU/CUDA/NumPy/Python RNG。资格通过后执行六次正式配对。GPU 协作锁与前后空闲检查通过，无资源剔除；前后快照无法排除运行中短暂竞争。

E2 先只读分析历史 trace 的五个窗口，再执行一次 native 短诊断，补充逻辑消息 ID、实际 CPU 写入位置、CUDA stream 就绪事件和参数版本。诊断窗口为 16–47、128–159、784–815、1472–1503，并各加两批边界上下文；140–147 只作第二窗口内的示例。

命令、原始文件哈希、导出范围及图表见[本轮归档](../artifacts/tgn_lastfm_critical_20261006/README.md)；数据、权重和完整逐事件 trace 保留在实验服务器。

## 分项验证

- 源码检查与编排：Python 解析、六个独立运行标签、资源排除与训练事件计数检查通过。
- 正确性：三对正式运行的两个 rank 初始和最终模型、最终内存、AP 曲线逐位一致；资格运行另通过 Adam/RNG 检查，future timestamp 为零。
- E2 仪器资格：与新 native 资格基线相比，最终模型、内存、Adam、RNG 和 AP 逐位一致；future timestamp 为零。
- Kernel 性能：历史 kernel/API 关联属于独立诊断，不能充当 E1 性能对照。
- 端到端：六次新正式进程、18 个正式全局 epoch 和六遍完整预热均完成并通过验收。

## 结果与解释

| seed | native 完整用时 s | arena 完整用时 s | 完整用时减少 | 正式训练用时减少 |
| --- | ---: | ---: | ---: | ---: |
| 4101 | 76.225 | 75.665 | 0.73% | 1.24% |
| 4102 | 76.716 | 75.566 | 1.50% | 1.51% |
| 4103 | 78.922 | 74.165 | 6.03% | 4.64% |
| 配对几何平均 | — | — | **2.78%** | **2.48%** |

完整进程配对几何平均加速为 1.028615×，正式训练为 1.025398×。单 session 三个 seed 的改善幅度差异较大，本轮不足以确认稳定收益，也不能把历史组合方案的 20.1% 归因于 arena。

E2 共配对 6,355 条逻辑消息，未发现缺失或歧义。主要窗口的 128 条状态消息和 128 条参数消息全部通过内容哈希及相邻 batch 检查；每条参数消息承载 24 个张量，共 891,604 bytes。加上边界上下文，共验证 280 条消息，另 8 条的消费者落在采集范围外，保持未观测状态。

128 个主要窗口 batch 中，forward 与 optimizer step 输入版本全部不同。例如 rank 0 的 batch 140 使用本地 batch 138 更新后的参数做 forward，之后接收 rank 1 的 batch 139 参数，才执行本地 Adam。该顺序是原生语义，提前覆盖 forward 参数可能改变轨迹。

CPU 写入审计记录完整 epoch 的 1,509 次提交。memory/mailbox 值只提交 push 节点，两个 timestamp 字段则提交全部 target 节点，必须分别追踪。主要窗口的 support snapshot 共解析 221,404 个节点字段读取：185,788 个关联到实际 CPU writer，35,616 个仍为本 epoch 的 reset 值；未发现这些读取区间与相应字段写入重叠，快照也与当时共享表一致。target gather 的结束边界未观测，保留未知；这份局部审计不等于全程序状态正确性证明。

新诊断中，各主要窗口参数接收 host 调用的中位区间约 0.50–0.60 ms，而 payload 生产就绪到接收 stream 就绪约 8.43–9.50 ms。状态消息对应区间约 0.216–0.222 ms 和 6.84–8.41 ms。消费者调用经常晚于生产就绪，长区间包含消费者调度、stream 排队和 NCCL，不能解释为纯链路传输。

历史 Nsight 中，worker 启动请求到首条指令的窗口中位数约 0.75–1.14 ms；rank 0 后段的 state-ready CPU 等待约 2.4–2.6 ms。它们是受 profiler 扰动的 wall spans，可能重叠，不能相加推算端到端收益。历史 kernel/线程 trace 与新版本诊断为两次不同运行，时钟不拼接，整批关键路径尚未完整证明。

## 否定证据与重新考虑条件

参数消息合并的本轮收益低于 3% 筛选门槛。arena 可保留为消融工具；若重新作为主要优化方向，需要新 session 的冻结确认合同和足够大的完整运行收益。

NCCL kernel 驻留包含 peer 等待；计算 kernel 区间的并集也不等于 SM 利用率。当前证据不支持“气泡可以全部消除”、同精度达标加速或八卡线性扩展。

## 允许写进论文的表述

在给定 LastFM 双卡配置下，保留原生训练语义的参数 arena 在三个新配对 seed 中使完整进程用时几何平均减少 2.78%；全部配对最终模型、内存和 AP 逐位一致。端点诊断验证了主要窗口的状态／参数消息内容及原生 forward 与 optimizer 输入版本的差异。

## 下一步

优先围绕接收时机和主机提交做保持版本语义的干预，分别测量 ready-to-post、receiver-ready-to-consumer 及实际 kernel 边界，再判断持久 worker 或提前准备是否能缩短关键路径。chunk/packet、采样 kernel 和 1/2/4/8 卡扩展性仍需独立实验合同。
