# 归档核验与运行条件

这是实际环境的研究归档。代码保持实测版本，环境解释器路径和本地依赖仍在源码中；它不是通用安装发行包。本次推送没有重新训练或 profiling。

## 不使用 GPU 的核验

在仓库根目录运行：

```bash
python3 tgn_lastfm_thread_20261004/verify_archive.py
python3 tgn_lastfm_thread_20261004/verify_archive.py --recompute
```

第一条命令检查文件哈希、Python 语法、15 次运行的状态、45 个完整 epoch 和保存的资格报告。第二条还需要 NumPy，在临时目录根据 30 份 rank summary 和 run.json 重新运行 [analyze_campaign.py](src/analyze_campaign.py)，并与保存的统计比较；不会启动 CUDA、修改保存的结果或提交。

`run.json` 导出统计必需字段；rank summary 中 per-batch dispatch 记录按 phase 汇总，计时和验证值保持不变。绝对设备路径用可移植指针替代；源文件和导出文件的不同哈希、导出方式见 [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json)。代码源文件逐字节保留，实测源码清单见 [measured_source_sha256.json](analysis/measured_source_sha256.json)。

资格 JSON 保存已执行检查的结果，不能替代重新验证。冻结权重逐位状态检查、训练轨迹比较和 producer-version 审计需要未上传的原始 checkpoint / 逐读日志。

## 测量合同

- LastFM：真实 2 维零 edge features，无 node feature；完整训练 905,172 条事件/轮。
- TGNN：batch=600、fanout=10、1 层、time/embed/memory=100、2 attention heads、dropout=0。
- 图：加载完整时间图，保留原生 timestamp 过滤；directed、add_reverse=False、shared memory resource。
- 硬件：两张 RTX PRO 6000 Blackwell Server Edition，同 NUMA、188 SM、无 NVLink。
- 软件：[software.json](software.json)，Python 3.12、PyTorch 2.11.0+cu130、CUDA 13、Nsight Systems 2025.5.2、NCU 2025.4.1。环境版本是在发布时读回，没有启动新训练。
- 所有五个方案共用 prefix snapshot、确定性重复节点 mailbox 选择、完整最后一批及 Gloo 控制 barrier 修复。原生 baseline 含这些共同修复。
- NCCL：`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`、`NCCL_SOCKET_IFNAME=lo`；线程数由 guard 固定为 1。
- seed=4101/4102/4103，每个方案 3 epoch，并保留原生 warm-up 更新。运行顺序为正序、反序、轮换，见 [run_campaign.py](src/run_campaign.py)。
- 正式计时关闭 profiler、thread probe、state audit、freeze weights；每个进程从共同主机 `CLOCK_MONOTONIC` 重建全局 epoch 区间。
- guard 在运行前检查 GPU 显存、利用率、计算进程及协作锁；前后资源检查不能排除内部短暂竞争。补充 native-arena 受竞争的计时已单独排除。

## 在匹配环境中重新实验

新建独立实验目录，复制本目录的 `src/`，建立空的 `runs/`、`analysis/`、`raw/`、`data/` 和 `cache/tmp/`。**不要在保存这些历史 run 标签的归档目录里运行 campaign**：已有通过的标签会被复用，不能当成新测量。

按父项目 [REPRODUCE.md](../tgn_pipeflash_causal_20260911/REPRODUCE.md) 和[来源记录](analysis/source_provenance.json)准备 PipeTGL、GNNFlow、patched DGL、Flash snapshot、pydeps、strict-gather 扩展。严格 gather 的 CUDA 源码和构建入口已在父归档 [fused_gather_strict.cu](../tgn_pipeflash_20260909/src/csrc/fused_gather_strict.cu) / [build_strict_gather.py](../tgn_pipeflash_20260909/src/build_strict_gather.py) 中，不重复复制第三方依赖。

在新目录的本地副本中配置 `run_campaign.py` 的 Python 解释器路径，建立脚本所需的 vendor/pydeps/cache 目录及 GPU 协作锁，再取得官方 `raw/lastfm.csv`。数据源和 SHA256 在 [data_manifest.json](analysis/data_manifest.json)。准备数据后依次执行：

```bash
python src/prepare_lastfm.py
python src/run_campaign.py --stage qualification --indices 4,5
python src/qualify_lastfm.py
python src/run_campaign.py --stage performance --indices 4,5
python src/analyze_campaign.py
python src/check_training_models.py
```

上述训练需真实空闲 GPU 和原环境依赖。脚本只终止它自己启动的进程组。`run_parameter_ablation.py` 是未完成的补充消融，需该新目录已有基线、冻结状态和配对性能证据，不是已完成性能实验的替代。

## 两层 profiler 诊断

Nsight v2 使用 CUDA/NVTX/OSRT/python-gil、process-tree context switch，采集 rank 0 的 `epoch:train:0` capture range。必须设置 `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`，再用 `--capture-range=nvtx --nvtx-capture=epoch:train:0 --capture-range-end=stop`；只有一次全局标记，通过同机 host clock 和 hooks 对齐两 rank。

sampler NCU 在独立真实输入 replay 中，过滤 `SampleLayerRecentKernel`，采集 LaunchStats、Occupancy、SchedulerStats、WarpStateStats、MemoryWorkloadAnalysis、SourceCounters。15-pass replay 和默认 cache flush 产生的耗时只作诊断。

`analyze_threads.py` 需要未上传的 SQLite、两 rank thread_intervals 以及未开 Nsight 的 thread probe；`analyze_dependencies.py` 和 sampler replay 需要未上传的 dependency_sets。对应分析结果已保存，但归档不能凭空重建逐事件 trace。
