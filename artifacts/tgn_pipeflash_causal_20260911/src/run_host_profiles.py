"""Collect host dispatch timelines without CUDA interception; diagnostic only."""
import json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    index=[]
    for world in [2,4]:
        for variant in ['native','flash']:
            label=f'stable_host_profile_p{world}_{variant}'
            cmd=[sys.executable,str(ROOT/'src/guard.py'),'--indices',','.join(['1','3','4','5'][:world]),'--label',label,'--timeout','100']
            for item in [f'DGLDEFAULTDIR={ROOT}/cache/dgl',f'TMPDIR={ROOT}/cache','NCCL_P2P_DISABLE=1','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=lo']:cmd+=['--env',item]
            cmd+=['--',sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(world),
                str(ROOT/'src/pipe_bench.py'),'--output',str(ROOT/'runs'/label/'results'),'--variant',variant,
                '--epochs','1','--complete-batches','--control-barrier','gloo','--deterministic-mailbox','--profile']
            print('START',label,flush=True);res=subprocess.run(cmd)
            if res.returncode:raise RuntimeError('Host profile failed/skipped: '+label)
            ranks=[json.loads((ROOT/'runs'/label/'results'/f'rank{k}'/'summary.json').read_text()) for k in range(world)]
            origin=min(x['process_start_perf_s'] for x in ranks);events=[]
            for rank,summary in enumerate(ranks):
                base=summary['process_start_perf_s']-origin
                for e in summary['phase_intervals']:
                    if 'label' in e:
                        events.append({'name':e['label'],'cat':e['phase'],'ph':'X','pid':rank,
                            'tid':e.get('thread_id',0),'ts':(base+e['start_s'])*1e6,'dur':e['cpu_dispatch_seconds']*1e6})
                for e in summary['epoch_intervals']:
                    events.append({'name':'complete training epoch','cat':'epoch','ph':'X','pid':rank,'tid':0,
                        'ts':(base+e['start_s'])*1e6,'dur':e['wall_seconds']*1e6})
            trace={'traceEvents':events,'displayTimeUnit':'ms','metadata':{
                'scope':'Host function-dispatch intervals on a shared monotonic clock. Overlap is retained; these are not GPU execution or dependency-idle times.'}}
            (ROOT/'analysis'/(label+'_chrome_trace.json')).write_text(json.dumps(trace)+'\n')
            index.append(label);(ROOT/'analysis/stable_host_profiles.json').write_text(json.dumps(index,indent=2)+'\n')
            print('COMPLETE',label,flush=True)
if __name__=='__main__':main()
