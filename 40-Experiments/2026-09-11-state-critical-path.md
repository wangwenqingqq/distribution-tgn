---
type: experiment
created: 2026-09-19
status: completed
tags: [experiment, critical-path, causal]
---
# 状态路径对受控 GPU 延迟敏感

## 问题与冻结合同
- 假设：状态交接在关键路径上，其他计算存在可重叠空间。
- 使用[最终确认](2026-09-11-final-confirmation.md)相同的正确状态协议和 Wikipedia 工作量。
- 干预：八卡、seed 2026、固定五轮，在目标状态更新前或 backward 后插入校准的 GPU stream 延迟；不改变样本数。
- 计时：所有 rank 全局单轮区间；每个非零延迟条件共 920 次训练期插入，并记录实际 CUDA 延迟。

## 执行与原始证据
- [代码与证据](https://github.com/wangwenqingqq/tgn-pipeflash-artifact/tree/main/tgn_pipeflash_causal_20260911)。
- `src/pipe_prefix_gpu_delay_bench.py`、`src/cuda_delay.py`；有效汇总为 `analysis/final_mechanism_summary.json`。
- `analysis/gpu_delay_summary.json` 还包含旧诊断，不能不加条件直接汇总。

## 分项验证
- 编译 / 正确性：复用已资格通过的状态协议；独立九次训练诊断审计 3347718 次支持读取，未来读取为零。
- Sanitizer / stress：没有本轮独立结果。
- Kernel-only：扰动用 CUDA 事件实测，不以名义 CPU sleep 时间替代。
- 端到端与持续负载：五轮固定工作量诊断；不构成新的同精度端到端确认。

## 结果、解释和代价
| 延迟位置 | 实测每次延迟 ms | 单轮秒 | 相对零延迟变化 |
| --- | ---: | ---: | ---: |
| 无 | 0 | 0.733176 | 0 |
| 目标状态更新前 | 0.998209 | 0.898969 | +165.8 ms |
| backward 后 | 1.001080 | 0.731237 | −1.9 ms |

独立端点诊断采用种子 2026–2028，每次三轮：原生校正版、直接融合、候选的状态上游等待分别约 268.7/336.2/307.4 ms/rank/epoch；双方就绪后残余分别 9.2/9.5/4.4 ms。后者包含提交、排队和传输，不能称纯网络时间；前者不等于 GPU SM 空闲率。

## 否定证据与重新考虑条件
- 干预仅单种子，不能据此估计任意优化的等比例收益。
- 状态约束原生版本已存在，不能称为融合后才出现。
- 三版共同加入的正确性修复可能增加同步；后续 breakdown 须把其代价单列。

## 允许写进论文的表述
受控延迟实验支持状态路径限制当前配置吞吐；不能据此断言网络带宽不足或所有状态依赖都可消除。

## 下一步
在 LastFM 区分必要依赖与实现额外等待，并关联 CPU 提交、CUDA stream、通信端点和数据版本。
