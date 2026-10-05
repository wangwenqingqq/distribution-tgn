import subprocess,sys
from pathlib import Path
r=Path("/home/data/wangxuran/tgn_pipeflash_20260909")
for variant in ['native','flash']:
 label='stable_cuda_profile_p1_'+variant
 cmd=[sys.executable,str(r/'src/guard.py'),'--indices','1','--label',label,'--timeout','150']
 for x in [f'DGLDEFAULTDIR={r}/cache/dgl',f'TMPDIR={r}/cache','NCCL_P2P_DISABLE=1','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=lo']:cmd+=['--env',x]
 cmd+=['--','nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--output='+str(r/'runs'/label/'trace'),
 sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','1',str(r/'src/pipe_bench.py'),
 '--output',str(r/'runs'/label/'results'),'--variant',variant,'--epochs','1','--complete-batches','--control-barrier','gloo','--deterministic-mailbox','--profile']
 print('START',label,flush=True);subprocess.run(cmd,check=True)
 with (r/'logs'/(label+'_export.log')).open('w') as f:
  subprocess.run(['nsys','export','--type','sqlite','--output',str(r/'runs'/label/'trace.sqlite'),str(r/'runs'/label/'trace.nsys-rep')],stdout=f,stderr=subprocess.STDOUT,check=True)
 with (r/'logs'/(label+'_analysis.log')).open('w') as f:
  subprocess.run([sys.executable,str(r/'src/trace_analysis.py'),'--sqlite',str(r/'runs'/label/'trace.sqlite'),'--output',str(r/'analysis'/(label+'.json'))],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('COMPLETE',label,flush=True)
