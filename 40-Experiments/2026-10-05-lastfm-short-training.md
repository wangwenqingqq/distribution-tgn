---
type: experiment
created: 2026-10-05
status: screening-complete
tags: [experiment, lastfm, pipetgl]
---
# LastFM 组合方案减少短训练用时，精度等价尚未验证

**结论：组合方案在 15 次固定工作量的双卡短训练中降低完整进程用时，但 AP / 训练轨迹不同，当前不能称为同精度达标加速。**

## 问题与冻结合同

- 任务：LastFM TGN 链路预测。真实 1,293,103 事件、1,980 nodes；每 epoch 905,172 条训练事件。真实 2 维零 edge features，无 node features；与历史 src/dst/time/ext_roll 全部逐行一致。
- 基线：共同修复 prefix snapshot、确定性重复节点 mailbox 选择、完整最后一批与 Gloo 控制 barrier 后的原生 PipeTGL，不是未经修改的官方实现。
- 方案：原生、loss-deferred、state-inline、direct Flash、Flash + parameter arena + state-first/inline。都保留原生 per-rank Adam、warm-up 更新和参数 staleness 语义；Flash 训练轨迹的等价性没有建立。
- 形状：batch=600、fanout=10、1 层、time/embed/memory=100、2 heads、dropout=0。不是历史人工特征或其他层数配置。
- 硬件 / 软件：两张同 NUMA RTX PRO 6000 Blackwell Server Edition，无 NVLink；Python 3.12.3、PyTorch 2.11.0+cu130、CUDA 13。NCCL P2P/IB 禁用，socket=lo。
- 重复：seed 4101/4102/4103，各 3 epoch；5 方案共 15 次。顺序按正序、反序、轮换，正式计时关闭 profiler、线程钩子、状态审计和冻结权重。
- 计时：epoch 为两 rank 共同单调时钟的最早开始至最晚结束；完整进程包括启动、数据图准备、原生 warm-up、3 轮训练与验证、退出。未设 AP 达标接受标准，因此这是吞吐筛选。

## 执行与原始证据

权威发布版本为 [tgn-pipeflash-artifact@79b835cf3b08](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/tree/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004)。

- [逐 seed / epoch 结果与配对区间](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/analysis/paired_short_training.json)
- [30 份正式 rank summary 和统计 run.json](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/tree/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/runs)
- [实测源码、环境和离线复算方法](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/79b835cf3b08791b602c4cf3a9b72b549d16c4ed/tgn_lastfm_thread_20261004/REPRODUCE.md)

运行前后选定 GPU 的显存 / 进程记录均通过检查；它不能排除运行内部短暂竞争。数据、权重、大型 trace 不进入知识仓库或代码归档。

## 分项验证

- 编译：发布时 23 个 Python 文件语法通过、142 份归档文件 hash 通过；没有重新编译 CUDA 依赖。
- 正确性：5 冻结方案初始参数相同、权重不变、最终 memory/mailbox/两种时间戳逐位相同，future-read 时间戳审计为零。30 个正式 rank 的初始权重与基线一致、最终张量有限。
- 完整轨迹：loss-deferred、state-inline 的最终参数、状态和 AP 曲线均与配对原生逐位相同。Flash / 组合不相同，冻结状态资格不证明 forward/gradient 或精度等价。
- Sanitizer / stress：未执行新检查。
- Kernel-only：只有独立诊断，未建立 kernel-only 正式加速比。
- 端到端：15 次短训练通过，共 45 个完整 epoch。根据归档 summary 离线重建所有统计、包括 bootstrap 区间，差异在 `1e-12` 内；这次发布没有重新训练。

## 结果、解释和代价

| 方案 | 平均 epoch / s | 完整进程 / s | 配对完整加速 | 三 seed 筛选区间 |
| --- | ---: | ---: | ---: | --- |
| 原生 | 14.404 | 78.255 | 1.000× | — |
| loss-deferred | 13.746 | 76.186 | 1.029× | [0.976, 1.089] |
| state-inline | 14.278 | 77.972 | 1.004× | [0.980, 1.035] |
| direct Flash | 15.233 | 81.313 | 0.962× | [0.945, 0.980] |
| 组合 | 10.940 | 62.533 | 1.252× | [1.213, 1.297] |

组合平均 epoch 用时减少约 24.1%，完整进程减少约 20.1%。加速比是配对几何均值，用时减少是方案平均时间与原生平均时间的比例，两者分母不同。

| 方案 | seed 4101 最终 AP | seed 4102 | seed 4103 |
| --- | ---: | ---: | ---: |
| 原生 / 两种线程消融 | 0.745365 | 0.580843 | 0.670907 |
| direct Flash | 0.712372 | 0.710395 | 0.571641 |
| 组合 | 0.715627 | 0.707636 | 0.669183 |

AP 非单调且跨 seed 波动较大；不能用三个最终 AP 的均值或冻结 memory 相同来证明精度保持。组合还同时改动计算、参数传输和顺序，机制归因需要[补充参数消融](2026-10-05-lastfm-parameter-transport.md)。

## 否定证据与重新考虑条件

单独 direct Flash 在三个配对 seed 都更慢。两个小型线程改动的区间跨 1，目前不能认定稳定收益。只有在明确计时、相同工作与精度合同，并通过更多重复或解释诊断差异后才重新考虑。

## 允许写进论文的表述

在既定 LastFM 双卡配置、三个配对 seed 和三轮固定工作量短训练中，组合方案相对共同正确性修复后的原生 baseline，完整进程平均用时减少 20.1%。尚未建立同精度或 time-to-target 加速，也没有测量正式八卡 scaling。

## 下一步

先补 native parameter-arena 的有效配对计时、定位组合交互，再验证 Flash forward/gradient 合同与完整精度；研究机制选择见[两层 breakdown](2026-10-05-lastfm-thread-breakdown.md)。[Wikipedia 达标实验](2026-09-11-final-confirmation.md)的结论不外推到这里。
