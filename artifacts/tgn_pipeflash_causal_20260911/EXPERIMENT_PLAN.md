# 因果实验预案（2026-09-11，分析新诊断结果之前记录）
复用上一轮固定输入/梯度合格的 strict FMA gather 与 Flash segmented attention；旧目录只读，新输出位于本目录。
## 待证伪假设
H1：融合后 state 链成为主要吞吐约束，并随 GPU 数增多出现更长的上游就绪等待。
H2：把下一本地批次的采样移到当前 target-state 发出之后，可以缩短 state 链，且无需删减训练工作或改动数学算子。
竞争解释：参数/Adam 传递链、主机描述符准备、共享内存读取、传输启动或硬件通信路径，而非 state 链。
## 测量与对照
Wiki、1 layer、batch 600、fanout 10、FP32 highest、dropout 0、相同 seed、完整末批与确定最后事件写回。
2/4/8 GPU：native/flash × baseline/state_first，先每组 2 epoch CUDA-event 诊断，后独立不带探针的重复测量。
记录 sender/receiver 的主机调用时间、生产流 ready CUDA event、recv work.wait 后消费流 CUDA event。
每 GPU 用单次同步 anchor 对齐同主机 perf_counter；保存对齐误差区间。每消息无额外同步。
仅将 producer 尚未就绪区间称为依赖等待；双方均已就绪到接收完成的残余包含 NCCL 排队、启动与传输，绝不称为纯链路带宽时间。
探针有开销，诊断 wall 不当作正式吞吐。主动延迟 state/compute 的对照只用来测敏感度，不当作加速。
## 精度/端到端
所有完整训练 epoch 正样本总数必须为 110232；比较初始权重、最终 memory/mailbox 时间戳、有限性、训练/验证曲线。
后续精度目标预先固定为 validation AP >= 0.97 连续 2 epoch；若 20 epoch 内未达标则记为未达标，不事后下调门槛。
正式收益须使用未带探针的端到端和 time-to-target 结果，并报告种子与未达标次数；采样调度对共享状态读版本的潜在影响必须检查。
## 硬件范围
保持上一轮 NCCL_P2P_DISABLE=1、NCCL_IB_DISABLE=1、NCCL_SOCKET_IFNAME=lo 与 Gloo 控制 barrier。
共享服务器，每次 guard 检查空闲卡，不占用他人作业。若八卡被占用就记录缺失，不替代成理论数。

## 追加因果对照（看到初步参数传输诊断后提出，独立标明探索性）
原始 state_first 并未稳定改善全部配置。24 个参数张量合并到连续 allocation，每步只传一个 snapshot，保留 Parameter 对象、形状、Adam 独立状态与原本更新顺序。
transport_control_v3：双端输入均先就绪；24 张量、逐次 pack/unpack、连续 allocation 三版轮换，每组 20 次预热 +100 次计时、3 轮；逐位内容与相同梯度下 5 次 Adam 更新检查全部通过。
先测 parameter-arena × native/flash，随后测 parameter-arena × state_first。这个新交互对照是在初步看到参数等待下降、状态等待上升之后提出的，不当作预注册确认性结果。
时序敏感性：固定 5 epoch、seed 2026，主线程在 target-state update 前，或 backward 返回后，各插入 1/4 ms sleep；同样正样本数。后者可以与尚未完成的 GPU backward 重叠，称为主机调度延迟，不称为 GPU kernel 变慢。用独立无延迟对照；不把延迟版本当作性能基线。

## 最终有效状态协议与确认集
未来支持节点读取审计、冻结权重全状态比较发现两种独立风险：支持读取可能发生在后续写入后；目标 uncached CPU gather 可能发生在先前写入前。前一轮 1.18x 仅保留为未统一因果状态的诊断数据。
最终两侧共同采用：前序状态到达后再读目标 CPU 行；四项支持状态复制到私有 CPU 张量后才释放后继；每轮共享状态 reset 后加 Gloo barrier。候选继续使用参数连续布局、主线程状态入队，以及后移下一本地批次采样。
冻结权重、两轮完整训练加预热和验证：三版完整 memory/mailbox/时间戳逐位相同；24 个 rank 模型初值和最终权重逐位相同；821715 次支持读取未见未来时间戳。
最终新种子 2034-2038，native 基线 / flash 直接融合基线 / flash 候选三组，顺序预先固定。统一 AP>=0.97 连续两轮；上限统一提高到40轮，以减少此前20轮上限的删失，精度门槛保持不变。无性能异常值剔除。
