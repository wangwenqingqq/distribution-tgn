"""Aggregate actual complete-process timing and explicitly labeled per-rank epoch timing."""
import argparse,json,re,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def stats(xs):
    return {'mean':statistics.mean(xs),'sample_std':statistics.stdev(xs) if len(xs)>1 else 0.0,
            'min':min(xs),'max':max(xs),'n':len(xs)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',default='qualified');args=p.parse_args()
    index=json.loads((ROOT/'analysis'/(args.tag+'_campaign.json')).read_text())
    records=[];grouped=collections.defaultdict(list);state_modes=set()
    for entry in index:
        label=entry['label'];path=ROOT/'runs'/label;guard=json.loads((path/'run.json').read_text())
        assert guard['status']=='passed'
        ranks=[json.loads((path/'results'/f'rank{k}'/'summary.json').read_text()) for k in range(entry['world_size'])]
        epochs=ranks[0]['config']['epochs']
        state_modes.add(ranks[0]['config'].get('deterministic_mailbox',False))
        walls=[max(x['epoch_intervals'][e]['wall_seconds'] for x in ranks) for e in range(epochs)]
        counts=[sum(x['epoch_intervals'][e]['positive_edges'] for x in ranks) for e in range(epochs)]
        warm=sum(w['positive_edges'] for x in ranks for w in x['training_work'] if w['phase']=='warm_up')
        assert counts==[110232]*epochs and warm==110232
        assert all(x['parameters_finite'] for x in ranks)
        global_epochs=[]
        global_driver=None
        if all('process_start_perf_s' in x for x in ranks):
            for e in range(epochs):
                starts=[x['process_start_perf_s']+x['epoch_intervals'][e]['start_s'] for x in ranks]
                ends=[a+x['epoch_intervals'][e]['wall_seconds'] for a,x in zip(starts,ranks)]
                global_epochs.append(max(ends)-min(starts))
            driver=[(x['process_start_perf_s']+v['start_s'],v['wall_seconds']) for x in ranks for v in x['phase_intervals'] if v['phase']=='train']
            global_driver=max(a+b for a,b in driver)-min(a for a,b in driver)
        log=(path/'stderr.txt').read_text()
        aps=[float(x) for x in re.findall(r'Validation ap ([0-9.]+)',log)]
        r=dict(entry,complete_process_wall_seconds=guard['child_process_wall_seconds'],
            max_rank_epoch_seconds=walls,mean_max_rank_epoch_seconds=statistics.mean(walls),
            global_epoch_makespan_seconds=global_epochs,mean_global_epoch_seconds=statistics.mean(global_epochs) if global_epochs else None,
            global_training_driver_seconds=global_driver,
            max_rank_training_driver_seconds=max(next(e['wall_seconds'] for e in x['phase_intervals'] if e['phase']=='train') for x in ranks),
            max_rank_native_main_seconds=max(x['native_main_wall_seconds'] for x in ranks),
            positive_training_edges=counts,positive_warmup_edges=warm,validation_ap_by_epoch=aps,
            max_rank_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in ranks))
        records.append(r);grouped[(r['world_size'],r['variant'])].append(r)
    aggregate=[]
    fields=['complete_process_wall_seconds','mean_max_rank_epoch_seconds','max_rank_training_driver_seconds','max_rank_native_main_seconds']
    if all(r['global_training_driver_seconds'] is not None for r in records):fields+=['mean_global_epoch_seconds','global_training_driver_seconds']
    for (world,variant),rows in sorted(grouped.items()):
        row={'world_size':world,'variant':variant,'seeds':[r['seed'] for r in rows]}
        row.update({name:stats([r[name] for r in rows]) for name in fields})
        row['final_validation_ap']=stats([r['validation_ap_by_epoch'][-1] for r in rows])
        aggregate.append(row)
    pairs=[]
    by={(r['world_size'],r['variant'],r['seed']):r for r in records}
    for world in sorted({r['world_size'] for r in records}):
        common=sorted({r['seed'] for r in records if r['world_size']==world and (world,'native',r['seed']) in by and (world,'flash',r['seed']) in by})
        if not common:continue
        pairs.append({'world_size':world,'seeds':common,**{
            f:stats([by[(world,'native',s)][f]/by[(world,'flash',s)][f] for s in common]) for f in fields}})
    assert len(state_modes)==1,'Mixed mailbox contracts'
    result={'tag':args.tag,'settings':{'batch':600,'positive_train_edges':110232,'epochs':3,
        'warmup_optimization_passes':1,'layer':1,'fanout':10,'dimension':100,'heads':2,'dropout':0,
        'precision':'FP32 highest; precise trig plus explicit FMA gather; original fused attention',
        'deterministic_last_event_mailbox':state_modes.pop(),
        'transport':'NCCL P2P disabled, IB disabled, lo; phase barriers use Gloo'},
        'records':records,'aggregate':aggregate,'paired_speedups_native_over_flash':pairs,
        'limitations':['Complete-process wall includes launcher, imports, all preparation, warm-up optimization, training, validation, artifact serialization and exit.',
        'global_epoch_makespan and global_training_driver use the common host monotonic clock, earliest rank start to latest rank completion; only available when every rank records process_start_perf_s. Maximum-rank durations are also retained separately.',
        'Three seeds and three training epochs are exploratory, not a convergence or time-to-target study.',
        'Native per-rank Adam, parameter staleness and sqrt(world-size) learning-rate scaling are retained.',
        'Kernel output/gradient qualification does not establish identical full training trajectories.',
        'Wikipedia one-layer B600 on one shared PCIe host does not establish generality or paper-level speedups.']}
    (ROOT/'analysis'/(args.tag+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'runs':len(records),'aggregate':aggregate,'paired_speedups':pairs},indent=2))
if __name__=='__main__':main()
