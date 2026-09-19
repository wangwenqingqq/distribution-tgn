---
type: experiment
created: 2026-09-19
status: completed
tags: [experiment, pipetgl, wikipedia]
---
# 候选有单轮收益，额外完整达标收益尚不稳定

## 问题与冻结合同
- 假设 / 接受标准：候选相对直接融合，能否缩短达到同一验证门槛的完整进程时间。
- 语义 / 数据：Wikipedia 9228 节点、157474 事件；每轮 110232 训练正样本；batch 600、单层、fanout 10、dropout 0、FP32 highest。
- 质量目标：沿用 PipeTGL 的批次/rank 平均 validation AP≥0.97，连续两轮，最多 40 轮。这不是汇集全部预测重新计算的全局排序 AP。
- 基线：共同修正状态协议后的 PipeTGL 原生算子；相同框架内的直接 Flash 融合；融合＋候选调度及参数传输合并。后两者不是独立完整 FlashTGN 系统。
- 环境：8×RTX PRO 6000，单机 PCIe，PyTorch 2.11 / CUDA 13；NCCL P2P 和 IB 关闭。
- 种子 / 原始顺序：2034–2038，三条件顺序预先固定，无异常值剔除。
- 计时：完整进程包括启动、导入、准备、一轮预热优化、训练、验证、保存和退出；单轮取所有 rank 最早开始至最晚结束。正式运行不带探针或状态审计。

## 执行与原始证据
- [权威报告及复现入口](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/tree/main/tgn_pipeflash_causal_20260911)。
- 计划：`analysis/final_confirmation_plan.json`；运行索引：`analysis/final_confirm_p8_campaign.json`。
- 汇总：`analysis/final_confirm_p8_summary.json`；配对区间：`analysis/final_confirm_p8_pairs.json`。
- 命令与依赖见归档 `REPRODUCE.md`；本次归档没有重跑训练。

## 分项验证
- 编译：当时最终 Python 源文件检查通过；依赖构建记录单列。
- 正确性：冻结权重两轮加预热/验证，三版完整 memory/mailbox/时间戳逐位相同，24 个 rank 模型初始/最终权重一致；821715 次支持读取未见未来时间戳。
- Sanitizer / stress：本轮没有独立结果，不以其他检查替代。
- Kernel-only：沿用父实验严格数学融合算子的输出/梯度资格及源文件、二进制哈希，不等价于完整训练轨迹逐位一致。
- 端到端：15/15 正式运行达标，所有训练轮事件计数完整。

## 结果、解释和代价
| 条件 | 全局单轮秒 | 完整达标时间秒，均值±样本标准差 | 平均训练轮数 |
| --- | ---: | ---: | ---: |
| PipeTGL 校正版 | 0.742 | 32.35±2.91 | 14.8 |
| 直接融合 | 0.732 | 30.98±2.93 | 13.8 |
| 融合＋候选 | 0.652 | 30.19±2.95 | 14.6 |

候选相对直接融合单轮加速 1.123×；完整达标配对几何平均加速 1.026×，按种子配对 bootstrap 95% 区间 [0.985,1.075]。相对原生校正版为 1.072×，[1.034,1.128]。区间只描述五个种子的变化。

## 否定证据与重新考虑条件
- 相对直接融合的完整达标区间跨 1，不能把单轮收益当成稳定端到端收益。
- 候选包含状态发送入队、采样位置和连续参数布局三个改动；本轮不是三者各自贡献的完整消融。
- [旧 1.18× 结果已撤销](../30-Ideas/2026-09-11-withdraw-invalid-speedup.md)，不得合并统计。

## 允许写进论文的表述
在既定 Wikipedia 八卡配置中，候选有单轮吞吐收益；相对直接融合的额外完整达标收益尚未得到五种子区间支持。不能外推 LastFM、多机或 P2P/NVLink 平台。

## 下一步
后续使用 LastFM，先核验状态依赖和扩展性，见[方向假设](../30-Ideas/2026-09-16-state-readiness.md)。
