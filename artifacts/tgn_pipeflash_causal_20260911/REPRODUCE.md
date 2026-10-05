# 复现：PipeTGL + Flash 状态关键路径实验

实验目录：/home/data/wangxuran/tgn_pipeflash_causal_20260911，运行主机 pro6000-8。

本目录的 vendor、pydeps、data 链接到上一轮 tgn_pipeflash_20260909；严格数学 gather 的已编译扩展复制到本目录 cache/strict_gather。保留父目录才能直接复现。没有修改系统驱动、共享 Python 环境或父目录的模型实现。父目录的 source_manifest、data_manifest、兼容性补丁和构建命令仍是依赖来源记录。

## 有效实现

- src/pipe_prefix_bench.py：加入共同状态读取约束的 PipeTGL 对照，variant=native 或 flash。
- src/pipe_prefix_candidate_bench.py：相同状态约束，连续参数布局、在主线程将状态发送入队、后移下一本地批次采样。
- src/prefix_snapshot.py：目标 CPU 状态读取等待前序状态到达；支持 memory/time/mailbox/time 复制为私有 CPU 张量后才释放后继。
- src/parameter_arena.py：保留 Parameter 对象和各自 Adam 状态，将参数数据安排在一段连续空间中。
- src/rendezvous_probe.py：非阻塞 CUDA 事件与同主机时间对齐；分别记录发送端就绪、接收端就绪和接收完成。
- src/cuda_delay.py：通过 CUDA 事件校准并测量实际插入的流延迟。
- analysis/prefix_qualified_implementation.json：最终模型和状态协议的源码哈希。
- analysis/qualified_implementation.json：继承上一轮已通过输出/梯度检查的严格数学 Flash 算子哈希。

## 单组复现

在 pro6000-8 上执行。每次必须使用尚未存在的新标签，旧结果不会被覆盖。guard 只使用空闲 GPU；有其他进程时会跳过。

~~~bash
PF_ROOT=/home/data/wangxuran/tgn_pipeflash_causal_20260911
PF_PY=/home/data/wangxuran/isaacsim6/env/bin/python

$PF_PY $PF_ROOT/src/run_causal.py \
  --tag replay_prefix_native_01 --worlds 8 --gpu-pool 1,2,3,4,5,6,7,0 \
  --variants native --schedules baseline --seeds 2034 --epochs 40 \
  --target-ap 0.97 --entry pipe_prefix_bench.py

$PF_PY $PF_ROOT/src/run_causal.py \
  --tag replay_prefix_candidate_01 --worlds 8 --gpu-pool 1,2,3,4,5,6,7,0 \
  --variants flash --schedules state_first --seeds 2034 --epochs 40 \
  --target-ap 0.97 --entry pipe_prefix_candidate_bench.py
~~~

运行器设置 NCCL_P2P_DISABLE=1、NCCL_IB_DISABLE=1、NCCL_SOCKET_IFNAME=lo，并使用 Gloo 做阶段边界控制。这里的结果不能外推到启用直连 P2P 或 NVLink 的机器。它同时限制 OMP/MKL/OpenBLAS 线程数为 1，使用 run-specific shared-memory names 和项目内临时目录。

完整确认集用 src/replay_final.py --tag 一个新标签，按照 analysis/final_confirmation_plan.json 的条件与顺序重放。种子为 2034–2038；精度门槛是原评估器的 validation AP >= 0.97 连续两轮，三组均最多 40 轮。评估器先平均批次 AP，再跨 rank 平均，不是将所有预测合并后重新排序计算的全局 AP。

## 正确性检查

冻结权重检查使用 --freeze-weights --audit-state，先运行 native baseline，再运行 native/flash candidate，保持同一 seed、GPU 数和 epoch 数。analysis/prefix_frozen_v2_qualification.json 保存了现有三组结果：完整 memory/mailbox/时间戳逐位一致，24 个 rank 的初始与最终权重一致，未来时间戳读取为零。src/check_prefix_frozen.py --version v2 可重新分析已有检查。

--audit-state 会在支持状态快照时检查其时间戳不超过当前批次最大时间。它用于诊断与资格检查，正式计时不带这个选项。有限次检查不代替协议的依赖顺序保证，也不声称完整训练轨迹逐位相同。

## 计时与分析

每轮全局正样本数必须是 110232；预热仍会进行一轮完整优化，同样计入总时间。所有进程的 perf_counter 来自同一主机，global epoch wall 是最早 rank 开始到最晚 rank 完成的区间。完整 process wall 由外层 guard 从 Popen 到子进程退出计时，包括启动、导入、准备、预热、训练、验证、保存和退出。

~~~bash
$PF_PY $PF_ROOT/src/aggregate_causal.py --tag final_confirm_p8
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  $PF_PY $PF_ROOT/src/analyze_confirmation.py --tag final_confirm_p8
$PF_PY $PF_ROOT/src/analyze_rendezvous.py --tag prefix_repeat_probe_base
$PF_PY $PF_ROOT/src/analyze_gpu_delay.py
~~~

带 --probe 的结果包含 CUDA/NVTX 观测开销，不用于正式吞吐和 time-to-target。接收端等待分成“发送端尚未就绪”和“双方就绪之后的残余”；后者仍含启动、排队与传输，不能称为纯链路时间。前者也不是整个 GPU 的 SM 空闲率或可直接相加的 bubble 百分比。

原始运行均在 runs/<label> 下：run.json、stdout/stderr、rank summary、模型和 memory checkpoint；带探针时还有 rendezvous.json。失败或被后续正确性检查否定的实验保留，不能与最终状态协议混合汇总。
