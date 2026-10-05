import argparse,json,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
runs=json.loads((ROOT/'analysis'/(args.tag+'_campaign.json')).read_text())
by={(r['world_size'],r['variant'],r['schedule'],r['seed']):r for r in runs};rows=[]
for key,a in by.items():
 if a['schedule']!='baseline':continue
 b=by.get((a['world_size'],a['variant'],'state_first',a['seed']))
 if b is None:continue
 dirs=[ROOT/'runs'/r['label']/'results/rank0' for r in [a,b]]
 load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
 initial=[load(p/'initial_model.pt') for p in dirs];final=[load(p/'final_model.pt') for p in dirs];mem=[load(p/'final_memory.pt') for p in dirs]
 initial_equal=all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
 den=sum(float(t.double().square().sum()) for t in final[0].values())
 num=sum(float((final[0][k]-final[1][k]).double().square().sum()) for k in final[0])
 rows.append({'world_size':a['world_size'],'variant':a['variant'],'seed':a['seed'],
              'epochs':[a['epochs'],b['epochs']],'initial_weights_equal':initial_equal,
              'final_weights_relative_l2':(num/den)**.5,
              'memory_timestamps_equal':torch.equal(mem[0]['node_memory_ts'],mem[1]['node_memory_ts']),
              'mailbox_timestamps_equal':torch.equal(mem[0]['mailbox_ts'],mem[1]['mailbox_ts']),
              'final_memory_relative_l2':float((mem[0]['node_memory']-mem[1]['node_memory']).norm()/mem[0]['node_memory'].norm())})
print(json.dumps(rows,indent=2))
(ROOT/'analysis'/(args.tag+'_schedule_state_check.json')).write_text(json.dumps(rows,indent=2))
assert all(r['initial_weights_equal'] and r['memory_timestamps_equal'] and r['mailbox_timestamps_equal'] for r in rows)
