# PipeTGL + Flash：代码与实验证据归档

截至 2026-10-05，归档 Wikipedia 原型、状态关键路径实验和新完成的 LastFM 双卡线程 / warp 第一轮实验。研究判断与文献笔记见本仓库[研究首页](../README.md)。

本目录由原 `tgn-pipeflash-artifact` 整体迁入，提交历史与实测文件保留，详见[合并记录](MIGRATION.md)。下面及历史复现文档中的“仓库根目录”指本目录 `artifacts/`；从整个项目根目录运行时，在原命令路径前加 `artifacts/`。后续代码、实验结果与研究笔记在 `distribution-tgn` 统一维护。

## LastFM 最新结果

[完整阅读入口](tgn_lastfm_thread_20261004/README.md) · [详细报告](tgn_lastfm_thread_20261004/results_round1.txt) · [运行与核验](tgn_lastfm_thread_20261004/REPRODUCE.md)

- 5 个方案 × 3 个 seed × 3 epoch，共 15 次无 profiler 短训练；每轮完整处理 905,172 条训练事件。
- 组合方案相对共同正确性修复后的原生 baseline，完整运行从 78.26 s 降到 62.53 s，用时减少 20.1%，配对几何平均加速 1.252×。
- 验证 AP 和训练轨迹不同，结果仅支持固定工作量吞吐，尚不支持同精度 / time-to-target 加速。单独 direct Flash 更慢；两个小型线程改动的收益区间跨 1。
- CPU 采样包装约占主线程 9%，中段调用的 DGL 构造约 0.67 ms / C++ sampler 0.42 ms；Nsight 中约 78% 的计算 kernel 不超过 5 μs。诊断已分别登记 wall spans、on-core、GIL、CUDA 活动与 profiler 开销。
- 参数传输微实验的消息合并显著降低独立完成耗时；保留 native attention 的补充消融正确性通过，但有效性能种子因资源竞争尚未补齐。

本轮[文件清单](tgn_lastfm_thread_20261004/ARCHIVE_MANIFEST.json)与[归档核验](tgn_lastfm_thread_20261004/ARCHIVE_VERIFICATION.json)独立保存，包含 30 份轻量 rank summary、逐 seed 指标、图表和实测源码；不上传数据、权重或大型 profiler trace。

## Wikipedia 八卡历史有效结论

- 八卡正确状态协议下，受控 GPU 延迟支持状态链限制吞吐。
- 候选相对直接融合，单轮加速约 1.123×；完整达标时间配对几何平均加速 1.026×，五种子 bootstrap 95% 区间 [0.985, 1.075]，不足以宣称稳定的额外端到端收益。
- 旧 1.18× 结果因未来状态读取已撤销，保留诊断记录，不纳入有效比较。
- 原生 baseline 指共同修正状态读取后的 PipeTGL；直接融合是在同一 PipeTGL 框架内接入融合算子，不是独立完整 FlashTGN。
- 结果限于 Wikipedia、单层、batch 600、单机八卡和禁用 NCCL P2P/IB 的既定配置。

## 阅读入口

| 内容 | 文件 |
| --- | --- |
| 最终结果和边界 | [RESULTS.md](tgn_pipeflash_causal_20260911/RESULTS.md) |
| 原环境运行方式 | [REPRODUCE.md](tgn_pipeflash_causal_20260911/REPRODUCE.md) |
| 图表 | [PDF](tgn_pipeflash_causal_20260911/analysis/causal_evidence.pdf) |
| 正式对比计划 | [final_confirmation_plan.json](tgn_pipeflash_causal_20260911/analysis/final_confirmation_plan.json) |
| 完整达标结果 | [final_confirm_p8_summary.json](tgn_pipeflash_causal_20260911/analysis/final_confirm_p8_summary.json) |
| 逐种子配对与区间 | [final_confirm_p8_pairs.json](tgn_pipeflash_causal_20260911/analysis/final_confirm_p8_pairs.json) |
| 因果诊断 | [final_mechanism_summary.json](tgn_pipeflash_causal_20260911/analysis/final_mechanism_summary.json) |
| 状态正确性资格 | [prefix_frozen_v2_qualification.json](tgn_pipeflash_causal_20260911/analysis/prefix_frozen_v2_qualification.json) |
| Wikipedia 文件校验和归档范围 | [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json) |
| LastFM 两层诊断与短训练 | [README.md](tgn_lastfm_thread_20261004/README.md) |

## 代码入口与历史探索

最终对照入口是 tgn_pipeflash_causal_20260911/src/pipe_prefix_bench.py；
候选入口是 src/pipe_prefix_candidate_bench.py；共同状态协议是 src/prefix_snapshot.py。
正式确认重放入口是 src/replay_final.py，必须使用新运行标签。

保留两个项目原有 src/，以便追溯探索过程；不是所有脚本都属于最终有效协议。
phase*.py、confirm_p8.py 和其他早期控制脚本只作历史记录，不能不加区分地重跑或合并其结果。
有效集合以最终报告和 final_confirmation_plan.json 为准。

## 归档内容与复现范围

历史 Wikipedia 归档保留原源文件、构建补丁、轻量分析文件、15 次最终正式运行的 120 份 rank summary 和图表；LastFM 新增归档以自己的文件清单和资格范围为准。
正式运行的 run.json 只保留统计必需的状态与完整进程耗时，并登记原文件哈希。
大体积逐事件 trace、数据集、模型／状态权重、运行环境和编译二进制未上传，仍由原实验目录保存。
ARCHIVE_MANIFEST.json 列出导入文件、哈希以及分析目录中主动省略的文件。

这是原环境的研究归档，**不是安装依赖后即可在任意机器直接训练的独立发行包**。
运行代码保持实测版本，包括原环境路径；运行依赖见两个项目的 REPRODUCE.md。
父实验源码出处及构建记录在 tgn_pipeflash_20260909/analysis/。
第三方依赖本体没有整体复制；PipeTGL、GNNFlow、DGL 和原 Flash artifact 的版本／文件哈希保留在来源记录中。
Flash snapshot 没有上游 git 元数据，不宣称它是官方最新提交，也不对第三方代码另行授予许可证。

不启动 GPU 训练也可以复算最终正式汇总：

    python3 tgn_pipeflash_causal_20260911/src/aggregate_causal.py --tag final_confirm_p8

该命令仅依赖 Python 标准库，读取本仓库保存的运行汇总并重写对应分析 JSON。
重新验证完整状态逐位一致仍需要未上传的原始状态 checkpoint。
