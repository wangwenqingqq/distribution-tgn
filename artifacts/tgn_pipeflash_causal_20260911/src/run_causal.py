import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--entry',default='pipe_bench.py');p.add_argument('--batch',type=int,default=600);p.add_argument('--freeze-weights',action='store_true');p.add_argument('--audit-state',action='store_true');p.add_argument('--gpu-pool',default='1,2,3,4,5,6,7,0');p.add_argument('--worlds',default='2,4,8');p.add_argument('--variants',default='native,flash');p.add_argument('--schedules',default='baseline,state_first');p.add_argument('--epochs',type=int,default=2);p.add_argument('--seeds',default='2026');p.add_argument('--probe',action='store_true');p.add_argument('--delay-ms',type=float,default=0);p.add_argument('--delay-stage',default='state');p.add_argument('--target-ap',type=float,default=0);args=p.parse_args()
q=json.loads((ROOT/'analysis/qualified_implementation.json').read_text())
for name,digest in q['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
records=[]
for seed in map(int,args.seeds.split(',')):
 for world in map(int,args.worlds.split(',')):
  conditions=[(v,s) for v in args.variants.split(',') for s in args.schedules.split(',')]
  if seed%2:conditions.reverse()
  for variant,schedule in conditions:
   label=f'{args.tag}_p{world}_{variant}_{schedule}_s{seed}_e{args.epochs}'
   cmd=[sys.executable,str(ROOT/'src/guard.py'),'--indices',','.join(args.gpu_pool.split(',')[:world]),'--label',label,'--timeout','240']
   for item in [f'DGLDEFAULTDIR={ROOT}/cache/dgl',f'TMPDIR={ROOT}/cache','NCCL_P2P_DISABLE=1','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=lo']:cmd+=['--env',item]
   cmd+=['--',sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(world),str(ROOT/'src'/args.entry),'--output',str(ROOT/'runs'/label/'results'),'--variant',variant,'--schedule',schedule,'--epochs',str(args.epochs),'--seed',str(seed),'--batch',str(args.batch),'--complete-batches','--control-barrier','gloo','--deterministic-mailbox','--delay-stage',args.delay_stage,'--delay-ms',str(args.delay_ms),'--target-ap',str(args.target_ap)]
   if args.audit_state:cmd+=['--audit-state']
   if args.freeze_weights:cmd+=['--freeze-weights']
   if args.probe:cmd+=['--rendezvous','--profile']
   print('START',label,flush=True);result=subprocess.run(cmd)
   if result.returncode:raise RuntimeError((label,result.returncode))
   actual_epochs=len(json.loads((ROOT/'runs'/label/'results/rank0/summary.json').read_text())['epoch_intervals'])
   for e in range(actual_epochs):
    total=sum(json.loads((ROOT/'runs'/label/'results'/f'rank{k}/summary.json').read_text())['epoch_intervals'][e]['positive_edges'] for k in range(world))
    assert total==110232,(label,e,total)
   records.append(dict(label=label,entry=args.entry,batch=args.batch,world_size=world,variant=variant,schedule=schedule,seed=seed,epochs=actual_epochs,max_epochs=args.epochs,target_ap=args.target_ap,probe=args.probe,delay_ms=args.delay_ms,delay_stage=args.delay_stage))
   (ROOT/'analysis'/(args.tag+'_campaign.json')).write_text(json.dumps(records,indent=2))
   print('COMPLETE',label,flush=True)
