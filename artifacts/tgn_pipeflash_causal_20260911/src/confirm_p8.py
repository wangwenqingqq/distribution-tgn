"""Frozen new-seed comparison; condition order alternates within seed."""
import json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
critical=['pipe_bench.py','pipe_arena_inline_bench.py','parameter_arena.py','inline_state.py','rendezvous_probe.py','run_causal.py']
hashes={p:hashlib.sha256((ROOT/'src'/p).read_bytes()).hexdigest() for p in critical}
plan={'seeds':[2029,2030,2031,2032,2033],'world_size':8,
 'conditions':['native baseline','flash baseline','native arena+inline+state_first','flash arena+inline+state_first'],
 'primary':'Paired complete-process time to validation AP >= 0.97 for two consecutive epochs, flash baseline / flash candidate.',
 'secondary':['native baseline / flash candidate','native candidate / flash candidate'],
 'max_epochs':20,'exclude_performance_outliers':False,'source_sha256':hashes,
 'notes':'These seeds have not been used to select the candidate. Candidate frozen before collecting these runs. Same validation target, all batch accounting, optimizer policy and mathematical model as earlier.'}
(ROOT/'analysis/confirmation_plan.json').write_text(json.dumps(plan,indent=2))
records=[]
for seed in plan['seeds']:
 variants=['native','flash'] if seed%2==0 else ['flash','native']
 for v in variants:
  modes=['baseline','candidate'] if (seed+(v=='flash'))%2==0 else ['candidate','baseline']
  for mode in modes:
   for p,digest in hashes.items():assert hashlib.sha256((ROOT/'src'/p).read_bytes()).hexdigest()==digest,p
   tag=f'confirm_p8_s{seed}_{v}_{mode}'
   entry='pipe_bench.py' if mode=='baseline' else 'pipe_arena_inline_bench.py'
   schedule='baseline' if mode=='baseline' else 'state_first'
   cmd=[sys.executable,str(ROOT/'src/run_causal.py'),'--tag',tag,'--worlds','8','--variants',v,'--schedules',schedule,'--epochs','20','--target-ap','0.97','--entry',entry,'--seeds',str(seed)]
   print('CONDITION',seed,v,mode,flush=True);subprocess.run(cmd,check=True)
   result=json.loads((ROOT/'analysis'/(tag+'_campaign.json')).read_text());assert len(result)==1
   result[0]['confirmation_condition']=mode;records+=result
   (ROOT/'analysis/confirm_p8_campaign.json').write_text(json.dumps(records,indent=2))
print('COMPLETE ALL FROZEN CONFIRMATION',flush=True)
