"""Aggregate unprofiled fixed-work paired LastFM runs, retaining every seed."""
from pathlib import Path
import json
import numpy as np
from run_campaign import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
rows = []
for seed in [4101, 4102, 4103]:
    for name, _, _, _ in CONFIGS:
        label = f'short_lastfm_p2_{name}_s{seed}_e3_v1'
        p = ROOT / 'runs' / label
        run = json.loads((p / 'run.json').read_text())
        assert run['status'] == 'passed'
        summaries = [json.loads((p / 'results' / f'rank{i}' / 'summary.json').read_text()) for i in range(2)]
        assert all(not s['config']['profile'] and not s['config'].get('thread_probe', False)
                   and not s['config']['audit_state'] and not s['config']['freeze_weights']
                   and s['parameters_finite'] for s in summaries)
        epochs = []
        for e in range(3):
            parts = [s['epoch_intervals'][e] for s in summaries]
            events = sum(x['positive_edges'] for x in parts)
            assert events == 905172
            starts = [s['process_start_perf_s'] + x['start_s'] for s, x in zip(summaries, parts)]
            ends = [t + x['wall_seconds'] for t, x in zip(starts, parts)]
            epochs.append({'epoch': e, 'global_wall_seconds': max(ends) - min(starts),
                           'rank_wall_seconds': [x['wall_seconds'] for x in parts], 'events': events})
        phases = {}
        for phase in ['warm_up', 'evaluate']:
            lists = [[x for x in s['phase_intervals'] if x['phase'] == phase and 'wall_seconds' in x]
                     for s in summaries]
            assert len(lists[0]) == len(lists[1]), (label, phase)
            durations = []
            for parts in zip(*lists):
                starts = [s['process_start_perf_s'] + x['start_s'] for s, x in zip(summaries, parts)]
                ends = [t + x['wall_seconds'] for t, x in zip(starts, parts)]
                durations.append(max(ends) - min(starts))
            phases[phase] = durations
        train = float(np.mean([x['global_wall_seconds'] for x in epochs]))
        total = run['child_process_wall_seconds']
        rows.append({'seed': seed, 'name': name, 'label': label, 'epochs': epochs,
                     'epoch_mean_seconds': train, 'events_per_second': 905172 / train,
                     'full_process_seconds': total, 'warmup_seconds': sum(phases['warm_up']),
                     'evaluation_seconds': sum(phases['evaluate']),
                     'startup_and_other_seconds': total - sum(phases['warm_up']) -
                        sum(phases['evaluate']) - sum(x['global_wall_seconds'] for x in epochs),
                     'validation_ap': [x['ap'] for x in summaries[0]['validation']],
                     'peak_allocated_bytes_per_rank': [s['peak_allocated_bytes'] for s in summaries]})
aggregate = []
baseline = {x['seed']: x for x in rows if x['name'] == 'baseline'}
rng = np.random.default_rng(4101)
for name, _, _, _ in CONFIGS:
    selected = [x for x in rows if x['name'] == name]
    pairs = {metric: [baseline[x['seed']][metric] / x[metric] for x in selected]
             for metric in ['epoch_mean_seconds', 'full_process_seconds']}
    comparisons = {}
    for metric, ratios in pairs.items():
        logs = np.log(ratios)
        boot = np.exp(rng.choice(logs, size=(10000, len(logs))).mean(axis=1))
        comparisons[metric] = {'paired_ratios': ratios, 'geometric_speedup': float(np.exp(logs.mean())),
                               'bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(),
                               'scope': 'Three-seed screening interval, not final statistical confirmation.'}
    aggregate.append({'name': name,
                      'epoch_mean_seconds': float(np.mean([x['epoch_mean_seconds'] for x in selected])),
                      'full_process_mean_seconds': float(np.mean([x['full_process_seconds'] for x in selected])),
                      'last_validation_ap': [x['validation_ap'][-1] for x in selected],
                      'vs_baseline': comparisons})
report = {'rows': rows, 'aggregate': aggregate,
          'scope': 'Two GPUs 4/5, fixed full LastFM work, three epochs plus warm-up; not time-to-target or eight-GPU scaling.',
          'epoch_clock': 'Union of rank epoch wall intervals on this shared CLOCK_MONOTONIC host.',
          'communication_environment': {'NCCL_P2P_DISABLE': '1', 'NCCL_IB_DISABLE': '1', 'NCCL_SOCKET_IFNAME': 'lo'}}
(ROOT / 'analysis/paired_short_training.json').write_text(json.dumps(report, indent=2) + '\n')
for x in aggregate:
    print(x['name'], 'epoch', round(x['epoch_mean_seconds'], 4), 'process', round(x['full_process_mean_seconds'], 4),
          'epoch_speedup', round(x['vs_baseline']['epoch_mean_seconds']['geometric_speedup'], 4),
          'process_speedup', round(x['vs_baseline']['full_process_seconds']['geometric_speedup'], 4), flush=True)
