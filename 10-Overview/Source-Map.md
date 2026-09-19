# 主线与资料来源

| 类型 | 权威来源 | 版本 / 核验日期 | 当前状态 |
| --- | --- | --- | --- |
| 代码 | [代码与证据归档](https://github.com/wangwenqingqq/tgn-pipeflash-artifact) | 2026-09-19，保存原源文件与哈希 | 已完成原型；运行依赖和数据单独保存 |
| 上游 | [PipeTGL](https://github.com/CGCL-codes/PipeTGL) | c79fbb6e0e39668dd0de541a2c2b0735140099dc | baseline 加了共同状态正确性修复，不能简称未经修改的官方版 |
| 文献 | [文献索引](../20-Literature/Index.md) | DOLPHIN / REAL / TempGNN：2026-09-16 核验官方摘要 | 未完成全文阅读与创新性排重 |
| 实验 | [最终报告](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/blob/main/tgn_pipeflash_causal_20260911/RESULTS.md) | final_confirm_p8，种子 2034–2038 | 正式 15 次运行；逐 rank 汇总保留，权重和完整原始产物不进 Git |
| 实现来源 | 归档中的 parent source_manifest、兼容性补丁和严格数学 gather 源码 | Flash 为原有本地 artifact 快照，无上游 git 元数据 | 不声称官方最新版；第三方依赖不整体复制 |
| 数据 | Wikipedia 已完成；LastFM 指定为后续数据集 | Wikipedia：9228 节点、157474 事件，训练 110232 正样本/轮 | LastFM 仅发现相关脚本和数据目录链接，实际文件尚未核验 |

只保存可分享的来源链接或可移植说明。设备绝对路径放入被忽略的 `_local/Source-Map.md`，不复制凭据、会话链接中的令牌或服务器配置。
