import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tags=['prefix_gpu_delay_zero','prefix_gpu_delay_state','prefix_gpu_delay_compute','gpu_delay_zero_p8','gpu_delay_state_ms1_p8','gpu_delay_compute_ms1_p8',
 'gpu_delay_arena_zero_p4','gpu_delay_arena_state_ms1_p4','gpu_delay_arena_compute_ms1_p4']
rows=[]
for tag in tags:
 p=ROOT/'analysis'/(tag+'_campaign.json')
 if not p.exists():continue
 for r in json.loads(p.read_text()):
  base=ROOT/'runs'/r['label']/'results'
  summaries=[json.loads((base/f'rank{k}/summary.json').read_text()) for k in range(r['world_size'])]
  delay=[json.loads((base/f'rank{k}/cuda_delay.json').read_text()) for k in range(r['world_size'])]
  values=[v['observed_gpu_ms'] for d in delay for v in d['records'] if v['phase']=='train']
  if r['delay_ms']:assert len(values)==184*r['epochs']
  walls=[]
  for e in range(r['epochs']):
   starts=[s['process_start_perf_s']+s['epoch_intervals'][e]['start_s'] for s in summaries]
   ends=[a+s['epoch_intervals'][e]['wall_seconds'] for a,s in zip(starts,summaries)]
   assert sum(s['epoch_intervals'][e]['positive_edges'] for s in summaries)==110232
   walls.append(max(ends)-min(starts))
  rows.append({**r,'mean_epoch_s':statistics.mean(walls),'epoch_walls_s':walls,
   'delay_observed_ms':{'mean':statistics.mean(values),'min':min(values),'max':max(values),'count':len(values)} if values else None,
   'summed_actual_delay_per_global_epoch_s':sum(values)/(r['epochs']*1000)})
zero={(r['entry'],r['world_size'],r['variant']):r for r in rows if not r['delay_ms']}
for r in rows:
 z=zero[(r['entry'],r['world_size'],r['variant'])]
 r['delta_epoch_s']=r['mean_epoch_s']-z['mean_epoch_s']
 r['sensitivity_to_serial_delay']=r['delta_epoch_s']/r['summed_actual_delay_per_global_epoch_s'] if r['delay_ms'] else None
 print(r['world_size'],r['variant'],r['delay_stage'],r['delay_ms'],'epoch',round(r['mean_epoch_s'],6),'delta',round(r['delta_epoch_s'],6),'actual',r['delay_observed_ms'],'slope',r['sensitivity_to_serial_delay'])
(ROOT/'analysis/gpu_delay_summary.json').write_text(json.dumps({'rows':rows,'scope':'Default-CUDA-stream latency is inserted after state receive has enqueued its dependencies and before the target-state update, or after backward and before parameter receive/update. Actual injected time measured by CUDA events. Same complete batches and optimizer operations. This is an artificial readiness perturbation, not an alternative model or production configuration; single seed exploratory unless separately replicated.'},indent=2))
