"""Write first-round results with statistical, correctness and resource scope."""
from pathlib import Path
import json
import csv
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
read=lambda name:json.loads((ROOT/'analysis'/name).read_text())
campaign=read('paired_short_training.json')
frozen=read('frozen_qualification.json')
models=read('trained_model_checks.json')
ownership=read('gpu_ownership_audit.json')
threads=read('thread_gpu_breakdown.json')
dependencies=read('dependency_breakdown.json')
sampler=read('sampler_component_breakdown.json')
communication=read('communication_breakdown.json')
ncu=read('ncu_sampler_b16_selected.json')
assert frozen['passed'] and models['passed']
included={x['label'] for x in campaign['rows']}
assert len(included)==15 and all(x['admissible'] for x in ownership['rows'] if x['label'] in included)
names={'baseline':'原生基线','loss_deferred':'延后 loss 读取','state_inline':'状态发送改为同步调用',
       'direct_flash':'仅 direct Flash','combination':'Flash + 参数 arena + state-first/inline'}
base=campaign['aggregate'][0]
lines=['LastFM 双卡线程与 warp 第一轮实验结果',
       '路径：'+str(ROOT),
       '设备：GPU 4/5，RTX PRO 6000 Blackwell，同 NUMA、无 NVLink。',
       '数据：1,293,103 条真实 LastFM 事件；每 epoch 905,172 条训练事件。与历史数据逐行一致。',
       '配置：batch 600，fanout 10，1 层，memory/embed/time 维度 100；真实 2 维零 edge feature，无 node feature。',
       '通信环境：NCCL_P2P_DISABLE=1、NCCL_IB_DISABLE=1、NCCL_SOCKET_IFNAME=lo。',
       '计时：3 个配对种子 × 3 epoch × 5 个方案，无 profiler/线程钩子/状态审计。',
       '全程包含进程启动、数据图准备、原生 warm-up 更新、3 轮训练及验证、退出；epoch 为两 rank 的共同墙钟跨度。',
       '15 次运行前后选定 GPU 均无其他计算进程，显存处于空闲阈值内；该检查不能排除运行内部短暂干扰。','',
       '方案 | 平均 epoch/s | 完整运行/s | 完整用时减少 | 配对全程加速比（3-seed 筛选区间）']
table=[]
for x in campaign['aggregate']:
    ci=x['vs_baseline']['full_process_seconds']['bootstrap_95ci']
    speed=x['vs_baseline']['full_process_seconds']['geometric_speedup']
    reduction=1-x['full_process_mean_seconds']/base['full_process_mean_seconds']
    lines.append(f"{names[x['name']]} | {x['epoch_mean_seconds']:.3f} | {x['full_process_mean_seconds']:.3f} | {reduction:.1%} | {speed:.3f}x [{ci[0]:.3f}, {ci[1]:.3f}]")
    table.append({'name':x['name'],'epoch_seconds':x['epoch_mean_seconds'],
                  'process_seconds':x['full_process_mean_seconds'],'process_time_reduction':reduction,
                  'paired_process_speedup':speed,'screening_ci_low':ci[0],'screening_ci_high':ci[1]})
lines += ['', '组合方案的训练 epoch 用时减少约 24.1%，全程用时减少约 20.1%；这是固定工作量短训练结果。',
          '延后 loss 和 state-inline 的区间跨 1，当前不能认定稳定收益；仅 direct Flash 三个种子均变慢。',
          '三 seed bootstrap 只作筛选，不能替代完整训练/更多重复的统计确认。','',
          '正确性与精度范围：',
          '五个冻结权重方案的初始权重相同、权重保持不变、最终 memory/mailbox 与两种时间戳逐位相同，future-read 审计为零。',
          '30 个 rank 的正式训练初始权重与配对基线逐位相同，最终参数和状态均有限。',
          'loss-deferred、state-inline 的最终参数/状态和 AP 曲线在三个种子、两个 rank 均与原生逐位相同。',
          'Flash 与组合的训练轨迹不同。冻结状态检查没有证明 forward/gradient 等价或精度保持；不宣称同 AP / time-to-target 加速。',
          '最终验证 AP（seed 4101 / 4102 / 4103）：']
for x in campaign['aggregate']:
    lines.append(names[x['name']]+'：'+' / '.join(f'{v:.6f}' for v in x['last_validation_ap']))
lines += ['', 'CPU 线程层：']
for x in threads['host_without_nsys']:
    ex=x['exclusive_main_wall']
    lines.append(f"rank{x['rank']}：采样包装 {ex['sampling_full']['fraction']:.1%}；support snapshot/compute {ex['support_snapshot_and_compute']['fraction']:.1%}；forward {ex['forward']['fraction']:.1%}；backward dispatch {ex['backward_dispatch']['fraction']:.1%}；loss 标量读取 {ex['loss_scalar_read']['fraction']:.2%}。")
lines += ['这些是独占主线程墙钟区间，含 CUDA 等待，不等于 CPU 运算时间；线程与主线程重叠，不能相加。',
          '每 batch 新建状态/参数发送线程，进入第一条 Python 指令的中位延迟约 0.15ms；单独 state-inline 未带来稳定收益。',
          'Nsight 同时记录两个 rank 的 autograd、通信线程、GIL、OSRT、调度及 GPU 活动。调度线程状态为 Unknown，无法精确区分阻塞与 runnable。',
          'Nsight 每 rank 的显式 profiler overhead 联合区间约 6s / 24.67s，不能从总时间简单减去；绝对比例仅用于诊断。','',
          'GPU 线程 / warp 层：']
for x in threads['gpu']:
    d=x['compute_launch_distribution']
    lines.append(f"rank{x['rank']}：计算 kernel {x['compute_kernel_count']} 次，{d['fraction_at_most_5us']:.1%} 不超过 5us，{d['fraction_grid_at_most_2_blocks']:.1%} 仅 1–2 个 block；trace 中计算活动联合时间约 {x['union_seconds']['compute']:.3f}s。")
lines += ['GPU 事件空白不等于 SM idle；NCCL kernel 长度含 peer wait/polling，不等于物理链路传输时间。',
          f"真实 batch16 sampler 的 NCU：grid {ncu['launch__grid_size']['value']} blocks / {ncu['launch__sm_count']['value']} SM；eligible warps {ncu['smsp__warps_eligible.avg.per_cycle_active']['value']}；long scoreboard 为主要 stall 来源。",
          'recent sampler 发射时预留 48KiB dynamic shared memory，而函数源码没有使用该 shared 区；这是可测假设，尚未改动/验证。',
          '当前小 grid 和包装开销使单纯提高理论 occupancy 的端到端空间有限。NCU replay/cache-flush 时间不得当作正式性能。','',
          '采样调用进一步拆分（50 次真实输入重放，中位 ms）：']
for x in sampler['rows']:
    d=x['component_median_ms']
    lines.append(f"batch{x['batch']}：total {x['total_median_ms']:.3f}，C++ sampler {d['native_sampler_host_ms']:.3f}，DGL 包装 {d['to_dgl_host_ms']:.3f}，最后 synchronize {d['final_sync_ms']:.3f}。")
lines += ['batch16 重放输出 hash 与未插桩和 NCU 输出相同；各输入重复结果一致。C++ 时间还包含 CUDA 编排与结果拷贝。','',
          '通信与 NUMA：']
for x in communication['rows']:
    lines.append(f"{x['topology']} / {x['case']}：{x['bytes']} bytes / {x['tensor_count']} tensors，中位 max-endpoint completion {x['median_completion_ms']:.4f}ms。")
same={x['case']:x for x in communication['rows'] if x['topology']=='same_numa'}
lines += [f"同 NUMA、同 891604B：24 次参数消息 vs 1 次消息，独立通信完成耗时比例约 {same['parameter_list']['median_completion_ms']/same['parameter_flat']['median_completion_ms']:.1f}x。",
          '该耗时包含 API、对端到达和 CUDA 完成，只是微实验；不能直接换成训练加速。NUMA 仅比较了一个同域/跨域 GPU 对。',
          '默认 NCCL 的 payload 校验通过，但没有对默认传输完成整轮训练资格检查。','',
          '依赖与 chunk 方向：',
          f"全部 {dependencies['adjacent']['pairs']} 对相邻 batch 都有 RAW 状态与 WAW target 重叠。",
          'batch-level state DAG 是状态前沿代理，不意味着整个 forward/backward 必须串行。参数接收实际在 backward 后、optimizer.step 前。',
          '采样 node slots 约 88.5% 是重复 node ID；原生 prepare_input 已在 memory_updater 前 unique，再展开，不能将其当作剩余 GRU 冗余。']
for x in dependencies['chunk_support_version_bounds']:
    if x['chunk_batches'] in [4,8,16]:
        lines.append(f"chunk {x['chunk_batches']}：support (node,last-writing-batch) 记录的乐观复用上界 {x['optimistic_version_reuse_fraction']:.1%}；错误地只按 node ID 计数会得到 {x['incorrect_node_only_reuse_fraction']:.1%}。")
lines += ['这个上界仅指状态记录准备；mail/producer-version 和跨 rank 权重版本需进一步审计，不能复用整个 learned activation。',
          '结合 DOLPHIN/REAL/TempGNN 摘要，后续优先研究：按 chunk 准备采样拓扑；版本感知 support-state packet；保留有序状态/参数提交，减少 DGL/CPU/CUDA 发射。',
          'LastFM 的数据/特征很小且 feature 为零，disk–RAM 特征搬运暂不是主要方向。以上为本项目推断，非论文复现结果。','',
          '补充参数消融：',
          'native attention + baseline schedule + 原生 state-send 线程，仅采用 parameter arena，冻结检查通过。',
          'seed4101 的 AP 曲线、最终参数/状态与原生逐位相同；但结束时选定 GPU 出现其他任务，因此整次性能计时被排除。',
          'seed4102 被空闲检查跳过，seed4103 未启动。目前其他任务占用八卡，三个有效性能种子尚待空闲窗口。',
          '恢复命令：OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/data/wangxuran/isaacsim6/env/bin/python '+str(ROOT/'src/run_parameter_ablation.py'),
          '脚本保留已跳过/受干扰目录，重试写入新版本目录；不覆盖原始结果。','',
          '未执行：unused shared-memory 改动、chunk/packet 实现、正式 1/2/4/8 卡 scaling、完整精度及 time-to-target 实验。',
          '原始证据：runs/ 下每次 run.json、stdout/stderr、前后 GPU 快照；trace .nsys-rep/.sqlite；NCU .ncu-rep；模型/状态及审计结果。']
(ROOT/'results_round1.txt').write_text('\n'.join(lines)+'\n')
with (ROOT/'analysis/short_training_table.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

fig,axes=plt.subplots(1,2,figsize=(11,4))
labels=['Native','Loss deferred','State inline','Direct Flash','Combination']
for ax,metric,title in zip(axes,['epoch_mean_seconds','full_process_seconds'],['Epoch wall time','Complete process wall time']):
    means=[np.mean([r[metric] for r in campaign['rows'] if r['name']==x['name']]) for x in campaign['aggregate']]
    ax.bar(range(5),means,color=['#8192a1','#a8bdce','#a8bdce','#d59482','#519a7e'],alpha=.85)
    for i,x in enumerate(campaign['aggregate']):
        values=[r[metric] for r in campaign['rows'] if r['name']==x['name']]
        ax.scatter(np.array([-0.1,0,0.1])+i,values,color='#243443',s=18,zorder=3)
    ax.set_xticks(range(5),labels,rotation=23,ha='right');ax.set_ylabel('Seconds');ax.set_title(title);ax.grid(axis='y',alpha=.18)
fig.suptitle('LastFM / 2 GPUs: 3 paired seeds, 3 epochs (unprofiled fixed-work screening)',fontsize=11)
fig.tight_layout();fig.savefig(ROOT/'analysis/short_training_comparison.png',dpi=170)
fig.savefig(ROOT/'analysis/short_training_comparison.svg')

fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
for ax,seed in zip(axes,[4101,4102,4103]):
    for name,color in [('baseline','#536c84'),('direct_flash','#bd8069'),('combination','#519a7e')]:
        row=next(r for r in campaign['rows'] if r['seed']==seed and r['name']==name)
        ax.plot([1,2,3],row['validation_ap'],marker='o',label=name,color=color)
    ax.set_title(f'Seed {seed}');ax.set_xlabel('Epoch');ax.set_xticks([1,2,3]);ax.grid(alpha=.2)
axes[0].set_ylabel('Validation AP');axes[-1].legend(fontsize=8)
fig.suptitle('Short validation curves: equal work does not establish equal convergence',fontsize=11)
fig.tight_layout();fig.savefig(ROOT/'analysis/short_validation_curves.png',dpi=170)
fig.savefig(ROOT/'analysis/short_validation_curves.svg')

hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))}
(ROOT/'analysis/current_source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
status=read('status.json')
status.update({'E1':'complete: exclusive host spans, both-rank on-core/GIL/OSRT and GPU unions; profiling overhead quantified',
               'E2':'complete: sampler NCU plus C++/DGL wrapper breakdown at four real input strata',
               'E2b':'complete: same/cross NUMA and default-NCCL isolated payload correctness',
               'E3':'complete: 15 unprofiled paired short-training runs, 30 rank model checks and ownership audit',
               'E4':'native-arena frozen correctness passed; first performance timing excluded due resource contention; other seeds require idle GPUs. Shared-memory optimization deferred.',
               'E5':'not run: all eight GPUs presently occupied by other tasks',
               'report':'results_round1.txt','performance_claim_scope':'Fixed-work short training only; Flash trajectory and AP differ from baseline.'})
(ROOT/'analysis/status.json').write_text(json.dumps(status,indent=2)+'\n')
print('Saved',ROOT/'results_round1.txt')
