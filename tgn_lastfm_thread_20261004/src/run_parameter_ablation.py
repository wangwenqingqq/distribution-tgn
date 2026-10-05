"""Isolate parameter packing from Flash attention and state-first scheduling."""
from pathlib import Path
import json
import subprocess
import sys
import csv
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
load = lambda p: torch.load(p,map_location='cpu',weights_only=True)

def compare(a,b):
    assert a.keys()==b.keys()
    rows = {k:{'equal':bool(torch.equal(a[k],b[k])),
               'max_abs':float((a[k]-b[k]).abs().max()),
               'finite':bool(torch.isfinite(b[k]).all())} for k in a}
    return {'bitwise_equal':all(v['equal'] for v in rows.values()),
            'max_abs':max(v['max_abs'] for v in rows.values()),
            'finite':all(v['finite'] for v in rows.values()),'fields':rows}

def gpu_timing_clean(path):
    run=json.loads((path/'run.json').read_text())
    for phase in ['before','after']:
        gpu={int(x[0]):[y.strip() for y in x] for x in csv.reader((path/(phase+'_gpus.txt')).read_text().splitlines())}
        apps=(path/(phase+'_apps.txt')).read_text()
        if any(int(gpu[i][2])>64 or gpu[i][1] in apps for i in run['indices']): return False
    return True

def launch(label,seed,epochs,frozen=False):
    stem=label.rsplit('_v',1)[0]
    version=1
    while True:
        actual_label=f'{stem}_v{version}'
        path=ROOT/'runs'/actual_label
        if not path.exists():break
        previous=json.loads((path/'run.json').read_text())
        if previous['status']=='passed' and (frozen or gpu_timing_clean(path)):return path
        if previous['status'] not in ['passed','skipped']:
            raise RuntimeError(f'{actual_label} requires failure diagnosis before retry')
        # Preserve skipped and contaminated trials; each retry uses a new folder.
        version+=1
    label=actual_label
    cmd=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node','2',
         str(ROOT/'src/pipe_parameter_bench.py'),'--output',str(path/'results'),
         '--dataset','LASTFM','--epochs',str(epochs),'--seed',str(seed),'--variant','native',
         '--schedule','baseline','--complete-batches','--deterministic-mailbox','--control-barrier','gloo']
    if frozen:cmd+=['--freeze-weights','--audit-state']
    guard=[PY,str(ROOT/'src/guard.py'),'--indices','4,5','--label',label,'--timeout','900']
    for env in ['DGLDEFAULTDIR='+str(ROOT/'cache/dgl'),'TMPDIR='+str(ROOT/'cache/tmp'),
                'NCCL_P2P_DISABLE=1','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=lo']:
        guard+=['--env',env]
    print('START',label,flush=True)
    rc=subprocess.run(guard+['--']+cmd).returncode
    if rc:
        (ROOT/'analysis/parameter_ablation_status.json').write_text(json.dumps({
            'state':'waiting_for_idle_gpus' if rc==75 else 'failed','label':label,'exit_code':rc,
            'resume':'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+PY+' '+str(ROOT/'src/run_parameter_ablation.py')},indent=2)+'\n')
        raise SystemExit(rc)
    if not frozen and not gpu_timing_clean(path):
        (ROOT/'analysis/parameter_ablation_status.json').write_text(json.dumps({
            'state':'waiting_for_idle_gpus','excluded_timing_label':label,
            'reason':'Selected GPUs were no longer idle after child exit; foreign activity may have overlapped.',
            'resume':PY+' '+str(ROOT/'src/run_parameter_ablation.py')},indent=2)+'\n')
        print('EXCLUDED timing:',label,flush=True)
        raise SystemExit(75)
    return path

qual = launch('qual_lastfm_p2_native_arena_s4101_e1_v1',4101,1,True)
reference=ROOT/'runs/qual_lastfm_p2_baseline_s4101_e1_v2/results/rank0'
qual_rows=[]
for rank in range(2):
    path=qual/'results'/f'rank{rank}'
    initial=compare(load(reference/'initial_model.pt'),load(path/'initial_model.pt'))
    unchanged=compare(load(path/'initial_model.pt'),load(path/'final_model.pt'))
    states=compare(load(reference/'final_memory.pt'),load(path/'final_memory.pt'))
    audit=json.loads((path/'prefix_audit.json').read_text())
    future=sum(x['future_mailbox_timestamps']+x['future_memory_timestamps'] for x in audit)
    summary=json.loads((path/'summary.json').read_text())
    assert initial['bitwise_equal'] and unchanged['bitwise_equal'] and states['bitwise_equal'] and future==0
    qual_rows.append({'rank':rank,'initial':initial,'unchanged':unchanged,'states':states,'future_reads':future,
                      'events':summary['epoch_intervals'][0]['positive_edges']})
assert sum(x['events'] for x in qual_rows)==905172
(ROOT/'analysis/parameter_arena_qualification.json').write_text(json.dumps({'passed':True,'ranks':qual_rows},indent=2)+'\n')

base_rows=json.loads((ROOT/'analysis/paired_short_training.json').read_text())['rows']
rows=[]
for seed in [4101,4102,4103]:
    run=launch(f'short_lastfm_p2_native_arena_s{seed}_e3_v1',seed,3)
    base=next(x for x in base_rows if x['name']=='baseline' and x['seed']==seed)
    summaries=[json.loads((run/'results'/f'rank{rank}/summary.json').read_text()) for rank in range(2)]
    epoch=[]
    for e in range(3):
        parts=[s['epoch_intervals'][e] for s in summaries]
        assert sum(x['positive_edges'] for x in parts)==905172
        starts=[s['process_start_perf_s']+x['start_s'] for s,x in zip(summaries,parts)]
        ends=[t+x['wall_seconds'] for t,x in zip(starts,parts)]
        epoch.append(max(ends)-min(starts))
    model_comparisons=[]
    for rank in range(2):
        path=run/'results'/f'rank{rank}'
        basepath=ROOT/'runs'/base['label']/'results'/f'rank{rank}'
        initial=compare(load(basepath/'initial_model.pt'),load(path/'initial_model.pt'))
        final=compare(load(basepath/'final_model.pt'),load(path/'final_model.pt'))
        states=compare(load(basepath/'final_memory.pt'),load(path/'final_memory.pt'))
        assert initial['bitwise_equal'] and final['finite'] and states['finite']
        model_comparisons.append({'rank':rank,'initial':initial,'final_parameters':final,'final_states':states})
    process=json.loads((run/'run.json').read_text())['child_process_wall_seconds']
    row={'seed':seed,'label':run.name,'epoch_seconds':epoch,'epoch_mean_seconds':float(np.mean(epoch)),
         'full_process_seconds':process,'validation_ap':[x['ap'] for x in summaries[0]['validation']],
         'baseline_validation_ap':base['validation_ap'],'model_comparisons':model_comparisons}
    row['ap_curve_bitwise_equal']=row['validation_ap']==base['validation_ap']
    row['epoch_speedup']=base['epoch_mean_seconds']/row['epoch_mean_seconds']
    row['process_speedup']=base['full_process_seconds']/process
    rows.append(row)
    (ROOT/'analysis/parameter_transport_ablation_partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',seed,row['epoch_mean_seconds'],row['full_process_seconds'],row['epoch_speedup'],
          'AP_curve_equal',row['ap_curve_bitwise_equal'],flush=True)

comparisons={}
rng=np.random.default_rng(4101)
for metric in ['epoch_speedup','process_speedup']:
    logs=np.log([x[metric] for x in rows]);boot=np.exp(rng.choice(logs,size=(10000,3)).mean(1))
    comparisons[metric]={'paired_ratios':[x[metric] for x in rows],
                         'geometric_speedup':float(np.exp(logs.mean())),
                         'bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
report={'rows':rows,'epoch_mean_seconds':float(np.mean([x['epoch_mean_seconds'] for x in rows])),
        'full_process_mean_seconds':float(np.mean([x['full_process_seconds'] for x in rows])),
        'vs_baseline':comparisons,
        'scope':'Follow-up three-seed screening against prior same-seed baseline runs. Native attention, native state-send threads, baseline scheduling; only contiguous parameter arena and one-message transport changed. Not time-to-target or an interleaved final confirmation.'}
(ROOT/'analysis/parameter_transport_ablation.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'analysis/parameter_ablation_status.json').write_text(json.dumps({'state':'complete'},indent=2)+'\n')
print('COMPLETE',report['epoch_mean_seconds'],report['full_process_mean_seconds'],comparisons,flush=True)
