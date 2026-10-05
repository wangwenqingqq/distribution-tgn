"""Exclusive main-thread wall breakdown and both-rank Nsight activity unions."""
from pathlib import Path
from collections import defaultdict
from bisect import bisect_right
import json
import re
import sqlite3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACE_RUN = ROOT / 'runs/nsys_lastfm_p2_native_threads_s4101_e1_v2'
PAYLOAD_RUN = ROOT / 'runs/payload_lastfm_p2_native_s4101_e1_v1'

def merge(intervals, lo=-float('inf'), hi=float('inf')):
    out = []
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi):
        if b <= a: continue
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def duration(intervals, lo=-float('inf'), hi=float('inf')):
    return sum(b-a for a,b in merge(intervals, lo, hi))

def intersection(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return total

def stats_ns(x):
    a = np.asarray(x, dtype=float) / 1e6
    return {'count': len(x), 'mean_ms': float(a.mean()), 'p50_ms': float(np.median(a)),
            'p90_ms': float(np.quantile(a, .9)), 'p99_ms': float(np.quantile(a, .99))}

def load_events(run, rank):
    return [x for x in json.loads((run / f'results/rank{rank}/thread_intervals.json').read_text())['events']
            if x['phase'] == 'train' and 8 <= x['batch'] <= 1501]

def host_breakdown(run):
    result = []
    for rank in range(2):
        events = load_events(run, rank)
        main = [x for x in events if x['native_tid'] == x['pid']]
        lo, hi = min(x['start_ns'] for x in main), max(x['end_ns'] for x in main)
        endpoints = sorted({lo, hi} | {x[k] for x in main for k in ['start_ns', 'end_ns']})
        # Deepest active interval wins; nested receive/target/ready waits are exclusive.
        main.sort(key=lambda x: x['start_ns'])
        active = []
        pos = 0
        exclusive = defaultdict(int)
        for a,b in zip(endpoints, endpoints[1:]):
            while pos < len(main) and main[pos]['start_ns'] <= a:
                active.append(main[pos]); pos += 1
            active = [x for x in active if x['end_ns'] > a]
            label = min(active, key=lambda x: x['end_ns']-x['start_ns'])['label'] if active else 'unmarked'
            exclusive[label] += b-a
        assert sum(exclusive.values()) == hi-lo
        bylabel = defaultdict(list)
        for x in events: bylabel[x['label']].append(x['end_ns']-x['start_ns'])
        result.append({'rank': rank, 'window_ns': [lo, hi], 'window_seconds': (hi-lo)/1e9,
                       'exclusive_main_wall': {k: {'seconds': v/1e9, 'fraction': v/(hi-lo)}
                                               for k,v in sorted(exclusive.items(), key=lambda kv: -kv[1])},
                       'inclusive_spans': {k: stats_ns(v) for k,v in bylabel.items()},
                       'scope': 'ThreadProbe enabled, native baseline. Exclusive main-thread wall time is not on-core time and contains CUDA waits. Inclusive worker intervals overlap and must not be added to the main thread.'})
    return result

c = sqlite3.connect(f'file:{TRACE_RUN / "trace.sqlite"}?mode=ro', uri=True)
c.row_factory = sqlite3.Row
strings = dict(c.execute('select id,value from StringIds'))
hooks = []
for row in c.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null and domainId != 1'):
    label = row['text'] or strings.get(row['textId'], '')
    m = re.fullmatch(r'(.+)\|r=(\d+)\|e=(\d+)\|b=(-?\d+)', label)
    if m and m[3] == '0':
        hooks.append(dict(row, label=m[1], rank=int(m[2]), batch=int(m[4])))
trace_events = [load_events(TRACE_RUN, rank) for rank in range(2)]
alignments = []
for rank in range(2):
    host = {(x['label'], x['batch'], x['native_tid']): x for x in trace_events[rank]
            if not x['label'].startswith('worker_first_instruction')}
    diffs = []
    for h in hooks:
        key = (h['label'], h['batch'], h['globalTid'] & ((1<<24)-1))
        if h['rank'] == rank and key in host:
            diffs.append(host[key]['start_ns'] - h['start'])
    assert diffs
    alignments.append({'rank': rank, 'offset_ns': int(np.median(diffs)),
                       'p99_abs_residual_ns': int(np.quantile(np.abs(np.array(diffs)-np.median(diffs)), .99)),
                       'matches': len(diffs)})
# Shared steady window bounded by recorded main-thread ranges on both ranks.
lo = min(x['start_ns']-alignments[rank]['offset_ns'] for rank in range(2)
         for x in trace_events[rank] if x['native_tid'] == x['pid'])
hi = max(x['end_ns']-alignments[rank]['offset_ns'] for rank in range(2)
         for x in trace_events[rank] if x['native_tid'] == x['pid'])
assert 0 < lo < hi < 25e9
width = hi-lo
processes = {row['pid']: row['globalPid'] for row in c.execute('select globalPid,pid from PROCESSES')}
rank_pids = [events[0]['pid'] for events in trace_events]
global_pids = [processes[pid] for pid in rank_pids]
thread_names = {row['globalTid']: strings[row['nameId']] for row in c.execute('select * from ThreadNames')}
role = {}
for rank, events in enumerate(trace_events):
    gp = global_pids[rank]
    role[gp | rank_pids[rank]] = 'main'
    for x in events:
        if x['label'].startswith('worker_run/'):
            role[gp | x['native_tid']] = x['label'].split('/')[1] + '_send'
    for tid,name in thread_names.items():
        if tid & ~((1<<24)-1) == gp and 'autograd' in name:
            role[tid] = 'autograd'

gpu_report = []
gpu_plot = []
cpu_report = []
for rank,gp in enumerate(global_pids):
    kernel = list(c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and end>? and start<?', (gp,lo,hi)))
    compute, nccl, copy, memset = [], [], [], []
    compute_durations = []
    compute_grids = []
    ranking = defaultdict(lambda: {'count': 0, 'sum_ns': 0, 'durations': [], 'grid': []})
    for k in kernel:
        name = strings[k['demangledName']]
        group = nccl if 'nccl' in name.lower() else compute
        group.append((k['start'], k['end']))
        if group is compute:
            compute_durations.append(k['end']-k['start'])
            compute_grids.append(k['gridX']*k['gridY']*k['gridZ'])
        rec = ranking[name]; rec['count'] += 1
        rec['sum_ns'] += min(hi,k['end'])-max(lo,k['start'])
        rec['durations'].append(k['end']-k['start'])
        rec['grid'].append(k['gridX']*k['gridY']*k['gridZ'])
    for table,dest in [('CUPTI_ACTIVITY_KIND_MEMCPY',copy),('CUPTI_ACTIVITY_KIND_MEMSET',memset)]:
        dest.extend((x['start'],x['end']) for x in c.execute(
            f'select start,end from {table} where globalPid=? and end>? and start<?',(gp,lo,hi)))
    groups = {'compute': compute, 'nccl': nccl, 'memcpy': copy, 'memset': memset}
    active = merge([iv for rows in groups.values() for iv in rows],lo,hi)
    all_compute = merge(compute,lo,hi)
    all_nccl = merge(nccl,lo,hi)
    top = []
    for name,v in sorted(ranking.items(),key=lambda kv:-kv[1]['sum_ns'])[:30]:
        top.append({'name': name, 'count': v['count'], 'summed_ms': v['sum_ns']/1e6,
                    'duration': stats_ns(v['durations']), 'median_grid_blocks': float(np.median(v['grid']))})
    gpu_report.append({'rank':rank,'device_id':kernel[0]['deviceId'], 'window_seconds':width/1e9,
                       'union_seconds':{k:duration(v,lo,hi)/1e9 for k,v in groups.items()},
                       'any_activity_seconds':duration(active)/1e9,
                       'no_recorded_activity_fraction':1-duration(active)/width,
                       'compute_nccl_overlap_seconds':intersection(all_compute,all_nccl)/1e9,
                       'compute_kernel_count': len(compute), 'nccl_kernel_count':len(nccl),
                       'compute_launch_distribution': {'duration':stats_ns(compute_durations),
                           'fraction_at_most_5us':float(np.mean(np.array(compute_durations)<=5000)),
                           'fraction_grid_at_most_2_blocks':float(np.mean(np.array(compute_grids)<=2))},
                       'kernel_duration_ranking':top,
                       'scope':'Nsight-instrumented diagnostic. NCCL kernel duration includes peer wait/polling; no-recorded-activity fraction is not SM idle time. Group unions overlap and are not additive.'})
    gpu_plot.append(groups)
    running = defaultdict(list)
    entered = {}
    # Scheduler records before lo establish each tracked thread's state at the window boundary.
    for row in c.execute('select start,isSchedIn,globalTid from SCHED_EVENTS where globalTid>=? and globalTid<? and start<? order by start',(gp,gp+(1<<24),hi)):
        tid,t = row['globalTid'],row['start']
        if tid not in role: continue
        if row['isSchedIn']: entered[tid] = t
        elif tid in entered: running[tid].append((entered.pop(tid),t))
    for tid,t in entered.items(): running[tid].append((t,hi))
    gil = {}
    for label in ['Holding GIL','Waiting for GIL']:
        sid = next(k for k,v in strings.items() if v == label)
        gil[label] = {row['globalTid']:row['total'] for row in c.execute(
            'select globalTid,sum(min(end,?)-max(start,?)) total from NVTX_EVENTS where textId=? and end>? and start<? and globalTid>=? and globalTid<? group by globalTid',
            (hi,lo,sid,lo,hi,gp,gp+(1<<24)))}
    osrt = defaultdict(int)
    for row in c.execute('select nameId,sum(min(end,?)-max(start,?)) total from OSRT_API where globalTid=? and end>? and start<? group by nameId',(hi,lo,gp|rank_pids[rank],lo,hi)):
        osrt[strings[row['nameId']]] += row['total']
    cpu_roles = []
    for ro in ['main','parameter_send','state_send','autograd']:
        tids = [tid for tid,value in role.items() if value==ro and tid & ~((1<<24)-1)==gp]
        run_ns = sum(duration(running[tid],lo,hi) for tid in tids)
        cpu_roles.append({'role':ro,'threads':len(tids),'summed_oncore_seconds':run_ns/1e9,
                          'summed_gil_holding_seconds':sum(gil['Holding GIL'].get(tid,0) for tid in tids)/1e9,
                          'summed_gil_wait_seconds':sum(gil['Waiting for GIL'].get(tid,0) for tid in tids)/1e9})
    overhead = [(x['start'],x['end']) for x in c.execute('select start,end from PROFILER_OVERHEAD where globalTid>=? and globalTid<? and end>? and start<?',(gp,gp+(1<<24),lo,hi))]
    cpu_report.append({'rank':rank,'roles':cpu_roles,
                       'main_offcore_seconds':width-cpu_roles[0]['summed_oncore_seconds']*1e9,
                       'main_osrt_summed_seconds':{k:v/1e9 for k,v in sorted(osrt.items(),key=lambda kv:-kv[1])[:12]},
                       'recorded_profiler_overhead_union_seconds':duration(overhead,lo,hi)/1e9,
                       'scope':'Scheduler thread states are Unknown, so off-core cannot distinguish blocked from runnable. GIL/OSRT assist interpretation; profiler overhead is pervasive and cannot be removed by simple subtraction.'})
    cpu_report[-1]['main_offcore_seconds'] /= 1e9

report = {'host_without_nsys':host_breakdown(PAYLOAD_RUN), 'host_with_nsys':host_breakdown(TRACE_RUN),
          'clock_alignment':alignments,'nsys_window_ns':[lo,hi], 'gpu':gpu_report,'cpu':cpu_report}
(ROOT / 'analysis/thread_gpu_breakdown.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'window_seconds':width/1e9,'host_without_nsys':report['host_without_nsys'],
                  'gpu':[dict(x, kernel_duration_ranking=x['kernel_duration_ranking'][:3]) for x in gpu_report],
                  'cpu':cpu_report},indent=2))

# Small steady slice: 8 sequential global batches across both ranks.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
slice_hooks = [x for x in hooks if 140 <= x['batch'] <= 147]
plot_lo = min(x['start'] for x in slice_hooks)
plot_hi = max(x['end'] for x in slice_hooks)
fig,ax = plt.subplots(figsize=(13,5))
colors = {'compute':'#2878b5','nccl':'#e6a23c','memcpy':'#3b9b68','memset':'#8c75bb'}
lane_labels = []
lane = 0
for rank,gp in enumerate(global_pids):
    for ro in ['main','state_send','parameter_send']:
        for h in slice_hooks:
            if h['rank']!=rank or role.get(h['globalTid'])!=ro or h['label'].startswith('worker_run/'): continue
            color = '#c75c5c' if any(t in h['label'] for t in ['wait','recv','join','receive']) else '#7b8d9e'
            ax.broken_barh([((h['start']-plot_lo)/1e6,(h['end']-h['start'])/1e6)],(lane-.3,.6),facecolors=color)
        lane_labels.append(f'Rank {rank}: {ro}'); lane+=1
    for name in ['compute','nccl']:
        ranges = merge(gpu_plot[rank][name],plot_lo,plot_hi)
        ax.broken_barh([((a-plot_lo)/1e6,(b-a)/1e6) for a,b in ranges],(lane-.3,.6),facecolors=colors[name])
        lane_labels.append(f'Rank {rank}: GPU {name}'); lane+=1
ax.set_yticks(range(lane),lane_labels);ax.invert_yaxis()
ax.set_xlim(0,(plot_hi-plot_lo)/1e6);ax.set_xlabel('Diagnostic elapsed time (ms)')
ax.set_title('Native PipeTGL / LastFM: host threads and two GPU timelines (batches 140-147)')
ax.grid(axis='x',alpha=.2);fig.tight_layout()
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#7b8d9e',label='Host call'),Patch(color='#c75c5c',label='Host wait/receive'),
                   Patch(color=colors['compute'],label='GPU compute'),Patch(color=colors['nccl'],label='NCCL (includes peer wait)')],
          loc='upper right',fontsize=8,ncol=2)
fig.savefig(ROOT/'analysis/thread_gpu_timeline.png',dpi=160)
fig.savefig(ROOT/'analysis/thread_gpu_timeline.svg')
