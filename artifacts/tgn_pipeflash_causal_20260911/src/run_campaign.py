"""Sequential guarded PipeTGL/Flash-kernel experiments; abort on any failed qualification."""
import argparse,subprocess,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-pool',default='1,3,4,5')
    p.add_argument('--world-sizes',default='1,2,4');p.add_argument('--seeds',default='2026')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=600)
    p.add_argument('--tag',required=True);p.add_argument('--variants',default='');p.add_argument('--deterministic-mailbox',action='store_true');args=p.parse_args()
    qualification=json.loads((ROOT/'analysis/qualified_implementation.json').read_text())
    assert qualification['passed']
    for name,digest in qualification['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    pool=args.gpu_pool.split(',');records=[]
    for seed in map(int,args.seeds.split(',')):
        for world in map(int,args.world_sizes.split(',')):
            variants=args.variants.split(',') if args.variants else (['native','flash'] if seed%2==0 else ['flash','native'])
            if not set(variants)<=set(['native','flash']):raise ValueError('Invalid variant')
            for variant in variants:
                label=f'{args.tag}_b{args.batch}_p{world}_{variant}_s{seed}_e{args.epochs}'
                cmd=[sys.executable,str(ROOT/'src/guard.py'),'--indices',','.join(pool[:world]),
                    '--label',label,'--timeout','150']
                for item in [f'DGLDEFAULTDIR={ROOT}/cache/dgl',f'TMPDIR={ROOT}/cache',
                    'NCCL_P2P_DISABLE=1','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=lo']:
                    cmd+=['--env',item]
                cmd+=['--',sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(world),
                    str(ROOT/'src/pipe_bench.py'),'--output',str(ROOT/'runs'/label/'results'),
                    '--variant',variant,'--epochs',str(args.epochs),'--batch',str(args.batch),'--seed',str(seed),
                    '--complete-batches','--control-barrier','gloo']
                if args.deterministic_mailbox:cmd+=['--deterministic-mailbox']
                print('START',label,flush=True);r=subprocess.run(cmd)
                if r.returncode:raise RuntimeError(f'Run failed/skipped: {label}, code {r.returncode}')
                summaries=[json.loads((ROOT/'runs'/label/'results'/f'rank{k}'/'summary.json').read_text()) for k in range(world)]
                for epoch in range(args.epochs):
                    edges=sum(x['epoch_intervals'][epoch]['positive_edges'] for x in summaries)
                    if edges!=110232:raise RuntimeError(f'Incomplete epoch: {label}/{epoch}: {edges}')
                records.append({'label':label,'world_size':world,'variant':variant,'seed':seed})
                (ROOT/'analysis'/(args.tag+'_campaign.json')).write_text(json.dumps(records,indent=2)+'\n')
                print('COMPLETE',label,flush=True)
if __name__=='__main__':main()
