"""Read-only E2 launch correlation; unknown semantic links remain unknown."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3


def union_ns(ranges, lo, hi):
    merged = []
    for start, end in sorted((max(a, lo), min(b, hi)) for a, b in ranges if b > lo and a < hi):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum(b - a for a, b in merged)


def family(label):
    if label in ['parameter_send_api', 'parameter_recv_api', 'parameter_clone']:
        return 'parameter_launch_scope'
    if label in ['state_send_api', 'state_receive', 'state_gpu_ready_cpu_wait', 'target_state']:
        return 'state_launch_scope'
    if label.startswith('worker_run/parameter'):
        return 'parameter_worker_scope'
    if label.startswith('worker_run/state'):
        return 'state_worker_scope'
    return 'unknown'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.original_root / 'runs/nsys_lastfm_p2_native_threads_s4101_e1_v2'
    args.output.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect('file:' + str(run / 'trace.sqlite') + '?mode=ro&immutable=1', uri=True)
    con.row_factory = sqlite3.Row
    strings = dict(con.execute('select id,value from StringIds'))
    processes = dict(con.execute('select pid,globalPid from PROCESSES'))
    host_records = [json.loads((run / 'results' / f'rank{rank}' / 'thread_intervals.json').read_text())['events']
                    for rank in range(2)]
    rank_pids = [events[0]['pid'] for events in host_records]
    rank_global = {processes[pid]: rank for rank, pid in enumerate(rank_pids)}
    scopes = []
    for row in con.execute('select rowid,start,end,globalTid,text,textId from NVTX_EVENTS where end is not null and domainId != 1'):
        label = row['text'] or strings.get(row['textId'], '')
        match = re.fullmatch(r'(.+)\|r=(\d+)\|e=(\d+)\|b=(-?\d+)', label)
        if match and int(match[3]) == 0:
            scopes.append({'event_id': 'nvtx_' + str(row['rowid']), 'start': row['start'], 'end': row['end'],
                           'globalTid': row['globalTid'], 'label': match[1],
                           'rank': int(match[2]), 'processing_batch': int(match[4])})
    by_tid = defaultdict(list)
    for scope in scopes:
        by_tid[scope['globalTid']].append(scope)
    indexed = {}
    for tid, entries in by_tid.items():
        entries.sort(key=lambda x: (x['start'], -x['end']))
        starts = [e['start'] for e in entries]
        maximum = []; cursor = 0
        for entry in entries:
            cursor = max(cursor, entry['end']); maximum.append(cursor)
        indexed[tid] = (entries, starts, maximum)

    def enclosing_scope(api):
        if api['globalTid'] not in indexed:
            return None
        entries, starts, maximum = indexed[api['globalTid']]
        pos = bisect_right(starts, api['start']) - 1
        candidates = []
        while pos >= 0 and maximum[pos] >= api['end']:
            item = entries[pos]
            if item['end'] >= api['end']:
                candidates.append(item)
            pos -= 1
        return min(candidates, key=lambda x: x['end'] - x['start']) if candidates else None

    runtime = defaultdict(list)
    for row in con.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        global_pid = row['globalTid'] & ~((1 << 24) - 1)
        if global_pid in rank_global:
            runtime[(global_pid, row['correlationId'])].append(dict(row))
    report = {'input_trace': 'nsys_lastfm_p2_native_threads_s4101_e1_v2/trace.sqlite',
              'windows': [], 'evidence_level': 'observed_launch_link',
              'semantic_critical_path_verified': False,
              'gaps': {'logical_message_id': 'not recorded', 'payload_batch_for_prefetch': 'not recorded in host spans',
                       'memory_mailbox_producer_versions': 'node/target proxy only',
                       'parameter_versions_at_forward_receive_step': 'not recorded'},
              'scope': 'Instrumented historical trace. A launch scope does not prove a kernel contains only one channel. NCCL includes peer waiting; resource spans are not additive.'}
    launch_rows = []
    plot_data = []
    for first, last in [(16, 47), (128, 159), (140, 147), (784, 815), (1472, 1503)]:
        selected = [s for s in scopes if first <= s['processing_batch'] <= last]
        if not selected:
            report['windows'].append({'global_batches': [first, last], 'available': False})
            continue
        lo = min(x['start'] for x in selected); hi = max(x['end'] for x in selected)
        per_rank = []
        kernels_for_plot = []
        for gp, rank in rank_global.items():
            kernels = list(con.execute('select start,end,correlationId,globalPid,streamId,contextId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and end>? and start<?', (gp, lo, hi)))
            counts = Counter(); duration_by_scope = defaultdict(list)
            nccl = []; compute = []; correlations = Counter()
            for kernel in kernels:
                name = strings[kernel['demangledName']]
                is_nccl = 'nccl' in name.lower()
                (nccl if is_nccl else compute).append((kernel['start'], kernel['end']))
                apis = runtime.get((gp, kernel['correlationId']), [])
                scope = enclosing_scope(apis[0]) if len(apis) == 1 else None
                correlations['unique_api' if len(apis) == 1 else 'missing_or_ambiguous_api'] += 1
                category = family(scope['label']) if scope else 'unknown'
                if is_nccl:
                    counts[category] += 1
                    duration_by_scope[category].append((kernel['start'], kernel['end']))
                    launch_rows.append({'window': [first, last], 'rank': rank,
                                        'correlation_id': kernel['correlationId'], 'stream_id': kernel['streamId'],
                                        'context_id': kernel['contextId'], 'gpu_start_ns': kernel['start'],
                                        'gpu_end_ns': kernel['end'], 'launch_scope': scope,
                                        'launch_family': category, 'api_start_ns': apis[0]['start'] if len(apis) == 1 else None,
                                        'api_end_ns': apis[0]['end'] if len(apis) == 1 else None,
                                        'logical_message_id': None, 'payload_batch': None,
                                        'producer_version_verified': False,
                                        'left_censored': kernel['start'] < lo,
                                        'right_censored': kernel['end'] > hi})
                kernels_for_plot.append({'rank': rank, 'start': kernel['start'], 'end': kernel['end'],
                                         'kind': 'nccl' if is_nccl else 'compute'})
            per_rank.append({'rank': rank, 'kernel_api_correlation': dict(correlations),
                             'nccl_launch_scope_counts': dict(counts),
                             'nccl_union_ms': union_ns(nccl, lo, hi) / 1e6,
                             'compute_union_ms': union_ns(compute, lo, hi) / 1e6,
                             'nccl_launch_scope_union_ms': {key: union_ns(value, lo, hi) / 1e6 for key, value in duration_by_scope.items()}})
        report['windows'].append({'global_batches': [first, last], 'available': True,
                                  'mapped_window_ns': [lo, hi], 'window_ms': (hi - lo) / 1e6,
                                  'ranks': per_rank})
        plot_data.append((first, last, lo, hi, selected, kernels_for_plot))
    (args.output / 'existing_launch_correlation.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output / 'existing_launch_events.jsonl').open('w') as file:
        for row in launch_rows:
            file.write(json.dumps(row) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    for first, last, lo, hi, selected, kernels in plot_data:
        fig, ax = plt.subplots(figsize=(13, 5))
        for rank in range(2):
            for scope in selected:
                if scope['rank'] != rank:
                    continue
                category = family(scope['label'])
                if category not in ['parameter_launch_scope', 'state_launch_scope']:
                    continue
                lane = rank * 4 + (0 if category == 'state_launch_scope' else 1)
                ax.broken_barh([((scope['start'] - lo) / 1e6, (scope['end'] - scope['start']) / 1e6)],
                              (lane - .3, .6), facecolors='#7b8d9e', alpha=.6)
            for kernel in kernels:
                if kernel['rank'] != rank:
                    continue
                start = max(kernel['start'], lo); end = min(kernel['end'], hi)
                lane = rank * 4 + (2 if kernel['kind'] == 'compute' else 3)
                ax.broken_barh([((start - lo) / 1e6, (end - start) / 1e6)], (lane - .3, .6),
                              facecolors='#2878b5' if kernel['kind'] == 'compute' else '#e6a23c')
        ax.set_yticks(range(8), [f'R{r}: {name}' for r in range(2)
                                  for name in ['state host scopes', 'parameter host scopes', 'GPU compute', 'GPU NCCL']])
        ax.invert_yaxis(); ax.set_xlim(0, (hi - lo) / 1e6)
        ax.set_xlabel('Historical diagnostic time (ms)')
        ax.set_title(f'LastFM global batches {first}–{last}: launch links observed; message/version links unresolved')
        ax.grid(axis='x', alpha=.2)
        ax.legend(handles=[Patch(color='#7b8d9e', label='Host call; scopes overlap'),
                           Patch(color='#2878b5', label='GPU compute'),
                           Patch(color='#e6a23c', label='NCCL; includes peer wait')], loc='upper right', fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output / f'existing_window_{first}_{last}.png', dpi=140)
        fig.savefig(args.output / f'existing_window_{first}_{last}.svg')
        plt.close(fig)
    print(json.dumps({'windows': len(plot_data), 'nccl_launch_rows': len(launch_rows),
                      'producer_version_verified': False, 'input_read_only': True}, indent=2))


if __name__ == '__main__':
    main()
