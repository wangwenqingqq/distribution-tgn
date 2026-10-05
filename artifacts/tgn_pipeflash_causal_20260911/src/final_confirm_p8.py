"""Final comparison with both target-read and support-read dependencies enforced."""
import json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
q=json.loads((ROOT/'analysis/prefix_qualified_implementation.json').read_text());assert q['passed']
for name,digest in q['files'].items():assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==digest,name
seeds=[2034,2035,2036,2037,2038]
conditions=[('native','baseline','pipe_prefix_bench.py'),('flash','baseline','pipe_prefix_bench.py'),('flash','candidate','pipe_prefix_candidate_bench.py')]
orders=[[0,1,2],[2,1,0],[1,2,0],[0,2,1],[1,0,2]]
plan={'seeds':seeds,'world_size':8,'conditions':conditions,'orders':orders,'target_ap':0.97,'consecutive_epochs':2,'max_epochs':40,
 'primary':'Paired complete-process time-to-target: flash baseline / flash candidate. Secondary: native baseline / flash candidate.',
 'why_40':'The earlier exploratory 20-epoch cap censored some 8-GPU runs; all final conditions receive the same larger cap. The AP target is unchanged.',
 'state_contract':'Target CPU gathers after predecessor state arrival; support CPU snapshots before successor release; epoch-reset barrier.',
 'qualification':q,'outlier_exclusions':False}
(ROOT/'analysis/final_confirmation_plan.json').write_text(json.dumps(plan,indent=2))
records=[]
for seed,order in zip(seeds,orders):
 for j in order:
  variant,condition,entry=conditions[j]
  for name,digest in q['files'].items():assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==digest,name
  tag=f'final_p8_s{seed}_{variant}_{condition}'
  schedule='baseline' if condition=='baseline' else 'state_first'
  cmd=[sys.executable,str(ROOT/'src/run_causal.py'),'--tag',tag,'--worlds','8','--variants',variant,'--schedules',schedule,'--epochs','40','--target-ap','0.97','--entry',entry,'--seeds',str(seed)]
  print('FINAL CONDITION',seed,variant,condition,flush=True);subprocess.run(cmd,check=True)
  result=json.loads((ROOT/'analysis'/(tag+'_campaign.json')).read_text());assert len(result)==1
  result[0]['confirmation_condition']=condition;records+=result
  (ROOT/'analysis/final_confirm_p8_campaign.json').write_text(json.dumps(records,indent=2))
print('ALL FINAL CONFIRMATIONS COMPLETE',flush=True)
