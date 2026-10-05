import argparse,json,subprocess,sys,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
if not re.fullmatch('[A-Za-z0-9_-]+',args.tag):p.error('Tag must contain only letters, digits, underscore and hyphen')
plan=json.loads((ROOT/'analysis/final_confirmation_plan.json').read_text());records=[]
for seed,order in zip(plan['seeds'],plan['orders']):
 for j in order:
  variant,condition,entry=plan['conditions'][j]
  for name,digest in plan['qualification']['files'].items():assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==digest,name
  tag=f'{args.tag}_s{seed}_{variant}_{condition}'
  schedule='baseline' if condition=='baseline' else 'state_first'
  subprocess.run([sys.executable,str(ROOT/'src/run_causal.py'),'--tag',tag,'--worlds',str(plan['world_size']),'--variants',variant,'--schedules',schedule,'--epochs',str(plan['max_epochs']),'--target-ap',str(plan['target_ap']),'--entry',entry,'--seeds',str(seed)],check=True)
  row=json.loads((ROOT/'analysis'/(tag+'_campaign.json')).read_text())[0];row['confirmation_condition']=condition;records.append(row)
  (ROOT/'analysis'/(args.tag+'_campaign.json')).write_text(json.dumps(records,indent=2))
