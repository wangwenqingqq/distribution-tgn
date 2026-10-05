# 代码与证据合并记录

2026-10-05，将 `tgn-pipeflash-artifact` 的代码与证据合入 `distribution-tgn/artifacts/`，研究笔记保留原编号目录。后续以 `distribution-tgn` 为统一项目入口。

- 导入来源：[tgn-pipeflash-artifact@79b835cf3b087](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/commit/79b835cf3b08791b602c4cf3a9b72b549d16c4ed)。
- 导入原仓库 663 个跟踪文件，包括 Wikipedia 和 LastFM 实验；两个仓库的提交历史通过普通合并保留，没有改写已发布提交。
- 原目录整体加 `artifacts/` 前缀。源码、结果、图表、数据清单和归档校验清单保持原文件内容；本目录首页增加合并后的导航说明，研究笔记改为仓库内相对链接。
- [Wikipedia 原清单](ARCHIVE_MANIFEST.json)的相对路径以本目录为基准；[LastFM 清单](tgn_lastfm_thread_20261004/ARCHIVE_MANIFEST.json)仍以其所在实验目录为基准。原校验报告记录的是归档发布时的检查，不代表重新训练。
- 原仓库保留合并前历史；本项目没有删除它，也没有改变任一仓库的可见性。旧提交链接可继续用于溯源，后续更新只需维护本项目。

合并后核验通过：除调整导航的归档首页外，原仓库 662 个文件逐字节一致；两个归档清单共 657 条文件哈希通过。LastFM 与 Wikipedia 的正式统计均在临时副本复算通过（数值容差 `1e-12`），没有重新训练或修改归档结果。

## 入口与目录

| 内容 | 合并后位置 |
| --- | --- |
| 研究笔记与当前判断 | [研究首页](../README.md)与[主张—证据表](../10-Overview/Claim-Evidence.md) |
| LastFM 双卡第一轮实验 | [tgn_lastfm_thread_20261004/](tgn_lastfm_thread_20261004/README.md) |
| Wikipedia 最终有效实验 | [tgn_pipeflash_causal_20260911/](tgn_pipeflash_causal_20260911/RESULTS.md) |
| Wikipedia 早期原型与撤销结果 | [tgn_pipeflash_20260909/](tgn_pipeflash_20260909/RESULTS.md) |

## 命令路径

从统一项目根目录核验 LastFM：

```bash
python3 artifacts/tgn_lastfm_thread_20261004/verify_archive.py --recompute
```

也可以先 `cd artifacts`，再执行历史文档里的原命令。LastFM 核验需要 NumPy，使用临时目录复算，不启动训练或 CUDA。Wikipedia 的原离线汇总命令从统一根目录改为：

```bash
python3 artifacts/tgn_pipeflash_causal_20260911/src/aggregate_causal.py --tag final_confirm_p8
```

该命令会重写对应汇总 JSON，核验时应在临时副本执行。实际 GPU 实验仍需历史说明中的环境和未上传数据，使用新的独立运行目录，不在归档目录覆盖历史 run 标签。

归档中的统计 JSON 可跟踪；其他 run 输出、数据集、权重、编译二进制和大型 profiler trace 继续排除。运行依赖与正确性限制见各实验的 `REPRODUCE.md`。
