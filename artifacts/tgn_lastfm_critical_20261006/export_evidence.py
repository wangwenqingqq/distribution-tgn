"""Export only explicit lightweight evidence; never dataset/weights/raw traces."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent

REMOTE_EXPORT = r'''
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1])
entries={};provenance=[]
def put(name,data,source=None):
    entries[name]=data
    if source is not None:
        provenance.append({'source_relative_path':str(source.relative_to(root)),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'export_relative_path':name,'export_sha256':hashlib.sha256(data).hexdigest(),
            'source_bytes':source.stat().st_size,'export_bytes':len(data)})
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ['pid','native_tid','globalTid','uuids','locks']}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
def dump(x):return (json.dumps(clean(x),indent=2)+'\n').encode()
analysis=['source_inventory_remote.json','runtime.json','paired_qualification.json','qualification_paired_results.json',
    'qualification_status.json','performance_paired_results.json','performance_status.json',
    'e2_semantic_qualification.json','e2_diagnostic_contract.json','historical_native_lineage.json']
for name in analysis:
    p=root/'analysis'/name;put('analysis/'+name,dump(json.loads(p.read_text())),p)
for p in sorted((root/'analysis').glob('critical_20261006_s1_performance_*.json')):
    put('analysis/'+p.name,dump(json.loads(p.read_text())),p)
for p in (root/'plans').glob('*.json'):put('plans/'+p.name,dump(json.loads(p.read_text())),p)
for folder in ['e2_existing','e2_endpoints']:
    for p in sorted((root/'analysis'/folder).iterdir()):
        name='analysis/'+folder+'/'+p.name
        if p.suffix in ['.png','.svg']:put(name,p.read_bytes(),p)
        elif p.suffix=='.json':put(name,dump(json.loads(p.read_text())),p)
        elif p.name in ['matched_messages.jsonl','parameter_versions.jsonl','existing_launch_events.jsonl']:
            lines=[clean(json.loads(x)) for x in p.read_text().splitlines()]
            put(name,(''.join(json.dumps(x)+'\n' for x in lines)).encode(),p)
        elif p.name=='cpu_provenance.jsonl':
            rows=[json.loads(x) for x in p.read_text().splitlines()]
            summaries=[{k:v for k,v in x.items() if k!='node_field_versions'} for x in rows]
            put('analysis/'+folder+'/cpu_provenance_counts.jsonl',(''.join(json.dumps(x)+'\n' for x in summaries)).encode(),p)
            case=[x for x in rows if x['processing_batch']==140]
            put('analysis/'+folder+'/cpu_provenance_case_b140.json',dump(case),p)
ids=[]
for stage in ['qualification','performance']:
    for seed in ([4101] if stage=='qualification' else [4101,4102,4103]):
        for arm in ['native','arena']:ids.append(f'critical_20261006_s1_{stage}_s{seed}_{arm}_a1')
ids.append('critical_20261006_e2_native_s4101_a1')
for label in ids:
    run=root/'runs'/label
    run_anchor=min(json.loads((run/'results'/f'rank{rank}'/'summary.json').read_text())['process_start_perf_s'] for rank in range(2))
    p=run/'run.json';raw=json.loads(p.read_text())
    light={k:raw[k] for k in ['start_utc','end_utc','indices','status','exit_code','child_process_wall_seconds']}
    light['command']=raw['command']
    put('runs/'+label+'/run.json',dump(light),p)
    for rank in range(2):
        directory=run/'results'/f'rank{rank}'
        for name in ['summary.json','prefix_snapshot.json','endpoint_contract.json']:
            p=directory/name
            if not p.exists():continue
            data=json.loads(p.read_text())
            if name=='summary.json':
                for epoch in data['epoch_intervals']:
                    epoch['global_host_start_offset_s']=data['process_start_perf_s']+epoch['start_s']-run_anchor
                    epoch['global_host_end_offset_s']=epoch['global_host_start_offset_s']+epoch['wall_seconds']
                data['work_totals_by_phase']={phase:sum(x['positive_edges'] for x in data['training_work'] if x['phase']==phase)
                                             for phase in ['warm_up','train']}
                for key in ['training_work','phase_intervals','process_start_perf_s']:data.pop(key,None)
                for entry in data.get('validation',[]):entry.pop('absolute_perf_s',None)
            put('runs/'+label+'/rank'+str(rank)+'/'+name,dump(data),p)
        p=directory/'executed_entrypoint.py'
        put('runs/'+label+'/rank'+str(rank)+'/'+p.name,p.read_bytes(),p)
        if '_qualification_' in label or label=='critical_20261006_e2_native_s4101_a1':
            p=directory/'prefix_audit.json'
            if p.exists():
                audit=json.loads(p.read_text())
                data={'snapshots':len(audit),'future_mailbox_timestamps':sum(x['future_mailbox_timestamps'] for x in audit),
                      'future_memory_timestamps':sum(x['future_memory_timestamps'] for x in audit)}
                put('runs/'+label+'/rank'+str(rank)+'/prefix_audit_summary.json',dump(data),p)
        for name in ['initial_model.pt','final_model.pt','final_memory.pt','final_optimizer_rng.pt','endpoint_events.jsonl']:
            p=directory/name
            if p.exists():provenance.append({'source_relative_path':str(p.relative_to(root)),
                 'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_bytes':p.stat().st_size,
                 'omitted_reason':'Weights/state/RNG checkpoints and full diagnostic events stay on server.'})
e2_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'e2').glob('*.py')}
put('analysis/e2_source_hashes.json',dump({'executed_diagnostic_sources':{k:e2_sources[k] for k in ['endpoint_probe.py','pipe_endpoint_bench.py']},
    'offline_analysis_sources':{k:v for k,v in e2_sources.items() if k not in ['endpoint_probe.py','pipe_endpoint_bench.py']},
    'scope':'Diagnostic sources are separate from frozen E1 src; offline scripts do not alter training.'}))
put('REMOTE_EXPORT_PROVENANCE.json',dump({'remote_root':str(root),'records':provenance,
    'excluded':['datasets','weights','environment and dependencies','large profiler SQLite','full node/write event traces','raw device/process configuration'],
    'export_transform':'Explicit allowlist; process identifiers removed; rank summaries omit heavy per-batch events; run.json strips UUIDs/locks; CPU provenance summarized plus batch140 case.'}))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for name,data in sorted(entries.items()):
        entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o644
        archive.addfile(entry,io.BytesIO(data))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='pro6000-8')
    parser.add_argument('--remote-root',default='/home/data/wangxuran/tgn_lastfm_critical_20261006')
    parser.add_argument('--remote-python',default='/home/data/wangxuran/isaacsim6/env/bin/python')
    args=parser.parse_args()
    command=' '.join(shlex.quote(x) for x in [args.remote_python,'-c',REMOTE_EXPORT,args.remote_root])
    result=subprocess.run(['ssh',args.host,command],stdout=subprocess.PIPE,check=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        for entry in archive:
            target=ROOT/entry.name
            assert entry.isfile() and not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(entry).read())
    inventory=ROOT/'analysis/source_inventory_local.json'
    data=json.loads(inventory.read_text())
    data['new_source']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))}
    inventory.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'exported_bytes':len(result.stdout),'run_count':9,'weights_and_datasets_exported':False}))


if __name__=='__main__':main()
