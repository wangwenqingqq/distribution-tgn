import argparse,json,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def stats(xs):
 return {'n':len(xs),'mean':statistics.mean(xs),'std':statistics.stdev(xs) if len(xs)>1 else None,'min':min(xs),'max':max(xs)}
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
index=json.loads((ROOT/'analysis'/(args.tag+'_campaign.json')).read_text());records=[];groups=collections.defaultdict(list)
for entry in index:
 base=ROOT/'runs'/entry['label'];guard=json.loads((base/'run.json').read_text());assert guard['status']=='passed'
 ranks=[json.loads((base/f'results/rank{k}/summary.json').read_text()) for k in range(entry['world_size'])]
 epochs=len(ranks[0]['epoch_intervals']);assert all(len(x['epoch_intervals'])==epochs for x in ranks)
 assert sum(w['positive_edges'] for x in ranks for w in x['training_work'] if w['phase']=='warm_up')==110232
 walls=[]
 for e in range(epochs):
  assert sum(x['epoch_intervals'][e]['positive_edges'] for x in ranks)==110232
  starts=[x['process_start_perf_s']+x['epoch_intervals'][e]['start_s'] for x in ranks]
  ends=[starts[k]+x['epoch_intervals'][e]['wall_seconds'] for k,x in enumerate(ranks)]
  walls.append(max(ends)-min(starts))
 assert all(x['parameters_finite'] for x in ranks)
 vals=ranks[0]['validation'];assert len(vals)==epochs
 for k in range(1,len(ranks)):
  assert [v['ap'] for v in ranks[k]['validation']]==[v['ap'] for v in vals]
 hit=bool(vals[-1]['reached_target'])
 record={**entry,'actual_epochs':epochs,'complete_process_wall_s':guard['child_process_wall_seconds'],
 'global_epoch_walls_s':walls,'mean_global_epoch_s':statistics.mean(walls),
 'reached_target':hit,'final_ap':vals[-1]['ap'],'best_ap':max(v['ap'] for v in vals),'validation':vals}
 records.append(record);groups[(entry['world_size'],entry['variant'],entry['schedule'])].append(record)
aggregate=[]
for (world,variant,schedule),rows in sorted(groups.items()):
 passed=[r for r in rows if r['reached_target']]
 item={'world_size':world,'variant':variant,'schedule':schedule,'seeds':[r['seed'] for r in rows],
 'target_passes':len(passed),'runs':len(rows),
 **{k:stats([r[k] for r in rows]) for k in ['complete_process_wall_s','mean_global_epoch_s','actual_epochs','final_ap','best_ap']},
 'complete_wall_to_target_s':stats([r['complete_process_wall_s'] for r in passed]) if passed else None}
 aggregate.append(item)
result={'records':records,'aggregate':aggregate,'target_semantics':'AP >= configured target in two consecutive globally reduced validation epochs; complete process wall includes target-triggered early stop, artifacts and exit. Non-hits are censored at max_epochs and do not count as time-to-target improvements.'}
(ROOT/'analysis'/(args.tag+'_summary.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(aggregate,indent=2))
