# 主线与资料来源

| 类型 | 权威来源 | 版本 / 核验日期 | 当前状态 |
| --- | --- | --- | --- |
| 代码 | [代码与证据归档](../artifacts/README.md) | 原导入版本 `79b835cf3b08`，2026-10-05；现统一维护于本仓库 | 保存实测源码、哈希、逐 rank 统计；运行依赖和数据单独保存 |
| 上游 | [PipeTGL](https://github.com/CGCL-codes/PipeTGL) | c79fbb6e0e39668dd0de541a2c2b0735140099dc | baseline 加了共同状态正确性修复，不能简称未经修改的官方版 |
| 文献 | [文献索引](../20-Literature/Index.md) | DOLPHIN / REAL / TempGNN：2026-09-16 核验官方摘要 | 未完成全文阅读与创新性排重 |
| 实验 | [最终报告](../artifacts/tgn_pipeflash_causal_20260911/RESULTS.md) | final_confirm_p8，种子 2034–2038 | 正式 15 次运行；逐 rank 汇总保留，权重和完整原始产物不进 Git |
| LastFM 实验 | [完整结果与入口](../artifacts/tgn_lastfm_thread_20261004/README.md) | 15 次双卡短训练，seed 4101–4103 | 固定工作量全程用时减少 20.1%；AP 等价未成立 |
| 两层观测 | [CPU / GPU breakdown](../artifacts/tgn_lastfm_thread_20261004/analysis/thread_gpu_breakdown.json) | 双 rank Nsight 与未开 Nsight 的线程 spans；sampler NCU | 区分 host wall / on-core / GIL / GPU 活动；profiler 时间不算正式收益 |
| 归档核验 | [导出清单](../artifacts/tgn_lastfm_thread_20261004/ARCHIVE_MANIFEST.json) | 30 份正式 rank summary，原始与导出 SHA256 | 可离线复算配对性能统计；完整正确性重验需要未上传 checkpoint |
| 实现来源 | 归档中的 parent source_manifest、兼容性补丁和严格数学 gather 源码 | Flash 为原有本地 artifact 快照，无上游 git 元数据 | 不声称官方最新版；第三方依赖不整体复制 |
| 数据 | [LastFM 数据清单](../artifacts/tgn_lastfm_thread_20261004/analysis/data_manifest.json) | 1,293,103 事件，1,980 nodes，训练 905,172/轮；真实 2 维零 edge features，无 node features | 与历史 src/dst/time/ext_roll 全部逐行一致；历史 NPZ 只覆盖训练前缀 |

只保存可分享的来源链接或可移植说明。设备绝对路径放入被忽略的 `_local/Source-Map.md`，不复制凭据、会话链接中的令牌或服务器配置。
