# 主线与资料来源

核验窗口：2026-09-12 至 2026-09-13（Asia/Shanghai）。本库是证据摘要，不是权威代码/实验目录副本。

| ID | 可移植来源定位 | 本次阅读 / 证据级别 |
| --- | --- | --- |
| D1 | `20260911 causal study / RESULTS.md` | 远端报告下载后双端 SHA 一致；直接阅读 |
| D2 | `20260911 causal study / REPRODUCE.md` | 复现入口与合同；未执行 |
| D3 | `20260911 causal study / final_confirm_p8_summary.json` | 五种子统计；直接阅读 JSON |
| D4 | `20260911 causal study / prefix_frozen_v2_qualification.json` | 冻结前缀资格；直接阅读 JSON |

文件 SHA256 见[证据清单](Evidence-Manifest.md)。这些不是仓库内相对文件路径：原件在权威研究目录，设备定位保存在忽略的 `_local/Source-Map.md`。原始数据不随笔记可用，哈希相同也不单独证明实验方法正确。

## 代码身份

实验继承的 PipeTGL vendor commit 为 `c79fbb6e0e39668dd0de541a2c2b0735140099dc`；候选修改需按 D2 与其实验 manifest 取得，不把 vendor commit 冒充候选的完整版本。四份下载证据已核对远端与本地 SHA256 一致。

## 历史讨论的用途

授权讨论只用于理解目标、修正、建议和定位材料；对话中的测试/显存/性能陈述若无独立原件，在正文标为“历史报告”“估算”或“待验证”。不保存原始聊天、登录资料、设备配置或服务器凭据。
