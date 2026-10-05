"""Fixed-work global wall changes caused by controlled host scheduling delays."""
import json,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tags=['sensitivity_zero','sensitivity_state_ms1','sensitivity_compute_ms1','sensitivity_state_ms4','sensitivity_compute_ms4',
 'sensitivity_arena_zero','sensitivity_arena_state_ms1','sensitivity_arena_compute_ms1']
rows=[]
for tag in tags:
 path=ROOT/'analysis'/(tag+'_campaign.json')
 if not path.exists():continue
 for entry in json.loads(path.read_text()):
  base=ROOT/'runs'/entry['label'];ranks=[json.loads((base/f'results/rank{k}/summary.json').read_text()) for k in range(entry['world_size'])]
  walls=[]
  for e in range(entry['epochs']):
   starts=[s['process_start_perf_s']+s['epoch_intervals'][e]['start_s'] for s in ranks]
   ends=[starts[k]+s['epoch_intervals'][e]['wall_seconds'] for k,s in enumerate(ranks)]
   assert sum(s['epoch_intervals'][e]['positive_edges'] for s in ranks)==110232
   walls.append(max(ends)-min(starts))
  rows.append({**entry,'global_epoch_walls_s':walls,'mean_epoch_s':statistics.mean(walls),
               'full_wall_s':json.loads((base/'run.json').read_text())['child_process_wall_seconds']})
zero={(r['entry'],r['variant'],r['schedule']):r for r in rows if r['delay_ms']==0}
for r in rows:
 z=zero[(r['entry'],r['variant'],r['schedule'])]
 r['delta_epoch_s']=r['mean_epoch_s']-z['mean_epoch_s']
 r['nominal_serial_delay_s']=184*r['delay_ms']/1000
 r['nominal_per_gpu_delay_s']=46*r['delay_ms']/1000
 r['delta_over_serial_nominal']=r['delta_epoch_s']/r['nominal_serial_delay_s'] if r['delay_ms'] else None
 print(r['entry'],r['variant'],r['schedule'],r['delay_stage'],r['delay_ms'],'wall',r['mean_epoch_s'],'delta',r['delta_epoch_s'])
(ROOT/'analysis/sensitivity_summary.json').write_text(json.dumps({'rows':rows,'scope':'Host sleep at two different program points, same five epochs and seed. Sleep duration is nominal and OS scheduling can add delay. Post-backward host sleep can overlap pending CUDA work. These data measure host schedule sensitivity, not isolated GPU f&b speed or a bubble percentage.'},indent=2))
