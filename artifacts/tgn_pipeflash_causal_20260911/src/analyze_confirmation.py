import json,math,statistics,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--tag',default='confirm_p8');args=parser.parse_args()
data=json.loads((ROOT/'analysis'/(args.tag+'_summary.json')).read_text())
seeds=sorted({r['seed'] for r in data['records']})
by={(r['seed'],r['variant'],r['confirmation_condition']):r for r in data['records']}
comparisons=[
('native_baseline_over_flash_candidate',('native','baseline'),('flash','candidate')),
('flash_baseline_over_flash_candidate',('flash','baseline'),('flash','candidate')),
('native_candidate_over_flash_candidate',('native','candidate'),('flash','candidate')),
]
rows=[]
for name,a,b in comparisons:
 pairs=[];censored=[]
 for seed in seeds:
  if (seed,*a) not in by or (seed,*b) not in by:continue
  x,y=by[(seed,*a)],by[(seed,*b)]
  rec={'seed':seed,'baseline_hit':x['reached_target'],'candidate_hit':y['reached_target'],
       'baseline_wall_s':x['complete_process_wall_s'],'candidate_wall_s':y['complete_process_wall_s'],
       'baseline_epochs':x['actual_epochs'],'candidate_epochs':y['actual_epochs']}
  if rec['baseline_hit'] and rec['candidate_hit']:
   rec['speedup']=rec['baseline_wall_s']/rec['candidate_wall_s'];pairs.append(rec)
  else:censored.append(rec)
 if not pairs and not censored:continue
 log_ratios=np.log([p['speedup'] for p in pairs])
 rng=np.random.default_rng(9102026)
 boot=np.exp(rng.choice(log_ratios,size=(50000,len(log_ratios)),replace=True).mean(axis=1)) if len(log_ratios) else []
 rows.append({'comparison':name,'paired_hits':pairs,'censored':censored,
              'geometric_mean_speedup_completed_pairs':float(np.exp(log_ratios.mean())) if len(log_ratios) else None,
              'seed_bootstrap_95pct_completed_pairs':np.quantile(boot,[.025,.975]).tolist() if len(boot) else None,
              'scope':'Interval resamples seed pairs only and describes completed pairs. Censored runs retained separately, never treated as successful shorter time-to-target.'})
result={'comparisons':rows,'seeds':seeds,'limitation':'Five seeds on one dataset/host; interval excludes variation across hardware, repeated process launches per seed, and datasets.'}
(ROOT/'analysis'/(args.tag+'_pairs.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
