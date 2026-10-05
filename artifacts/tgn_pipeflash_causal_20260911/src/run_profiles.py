"""Collect paired diagnostic profiles after, and separate from, throughput runs."""
import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    records=[]
    for world in [2,4]:
        for variant in ['native','flash']:
            label=f'profile_final_p{world}_{variant}'
            cmd=[sys.executable,str(ROOT/'src/guard.py'),'--indices',','.join(['1','3','4','5'][:world]),
                '--label',label,'--timeout','240']
            for item in [f'DGLDEFAULTDIR={ROOT}/cache/dgl',f'TMPDIR={ROOT}/cache',
                'NCCL_P2P_DISABLE=1','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=lo']:cmd+=['--env',item]
            cmd+=['--','nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                '--output='+str(ROOT/'runs'/label/'trace'),sys.executable,'-m','torch.distributed.run',
                '--standalone','--nproc_per_node',str(world),str(ROOT/'src/pipe_bench.py'),
                '--output',str(ROOT/'runs'/label/'results'),'--variant',variant,'--epochs','1',
                '--complete-batches','--control-barrier','gloo','--profile']
            print('START',label,flush=True)
            r=subprocess.run(cmd)
            if r.returncode:raise RuntimeError(f'Profile failed/skipped: {label}')
            export=['nsys','export','--type','sqlite','--output',str(ROOT/'runs'/label/'trace.sqlite'),
                    str(ROOT/'runs'/label/'trace.nsys-rep')]
            with (ROOT/'logs'/(label+'_export.log')).open('w') as log:subprocess.run(export,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (ROOT/'logs'/(label+'_analysis.log')).open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'src/trace_analysis.py'),'--sqlite',
                    str(ROOT/'runs'/label/'trace.sqlite'),'--output',str(ROOT/'analysis'/(label+'.json'))],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            records.append(label);(ROOT/'analysis/final_profiles.json').write_text(json.dumps(records,indent=2)+'\n')
            print('COMPLETE',label,flush=True)
if __name__=='__main__':main()
