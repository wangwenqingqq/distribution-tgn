"""Match transport endpoints on one host; distinguish producer readiness from post-rendezvous residual."""
import json,argparse,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
records=json.loads((ROOT/'analysis'/(args.tag+'_campaign.json')).read_text())
results=[]
for run in records:
 base=ROOT/'runs'/run['label']/'results'
 summaries=[json.loads((base/f'rank{k}/summary.json').read_text()) for k in range(run['world_size'])]
 traces=[json.loads((base/f'rank{k}/rendezvous.json').read_text()) for k in range(run['world_size'])]
 uncertainty=[max((tr[k]['host_upper_s']-tr[k]['host_lower_s'])/2 for k in ['calibration','end_calibration'] if k in tr)+1e-6 for tr in traces]
 send={};recvs=[]
 for tr in traces:
  for item in tr['records']:
   src,dst=(item['rank'],item['peer']) if item['direction']=='send' else (item['peer'],item['rank'])
   key=(item['channel'],src,dst,item['sequence'])
   if item['direction']=='send':assert key not in send;send[key]=item
   else:recvs.append((key,item))
 matched=[];missing=[]
 for key,r in recvs:
  if r['phase']!='train':continue
  if key not in send:missing.append(key);continue
  s=send[key];assert s['bytes']==r['bytes'],key
  sr=max(s['host_enter_s'],s['gpu_ready_s']);rr=max(r['host_enter_s'],r['gpu_ready_s']);rd=r['gpu_done_s']
  total=max(0,rd-rr);dependency=min(total,max(0,sr-rr));residual=max(0,rd-max(sr,rr))
  er=uncertainty[r['rank']];es=uncertainty[s['rank']]
  dep_lower=min(max(0,total-2*er),max(0,sr-rr-es-er))
  dep_upper=min(total+2*er,max(0,sr-rr+es+er))
  matched.append({'channel':r['channel'],'rank':r['rank'],'epoch':r['epoch'],'sequence':r['sequence'],
   'receive_ready_s':rr,'send_ready_s':sr,'receive_done_s':rd,'receive_span_s':total,
   'upstream_not_ready_s':dependency,'upstream_calibration_lower_s':dep_lower,'upstream_calibration_upper_s':dep_upper,'after_both_ready_s':residual,'bytes':r['bytes'],
   'causal_order_violation_s':max(0,sr-rd),'sender_api_seconds':s['host_launch_return_s']-s['host_enter_s']})
 rows=[]
 for e in range(run['epochs']):
  starts=[s['process_start_perf_s']+s['epoch_intervals'][e]['start_s'] for s in summaries]
  ends=[starts[k]+s['epoch_intervals'][e]['wall_seconds'] for k,s in enumerate(summaries)]
  for rank in range(run['world_size']):
   for channel in ['state','parameter']:
    items=[m for m in matched if m['epoch']==e and m['rank']==rank and m['channel']==channel]
    rows.append({'epoch':e,'rank':rank,'channel':channel,'count':len(items),'global_epoch_s':max(ends)-min(starts),
     **{key:sum(i[key] for i in items) for key in ['receive_span_s','upstream_not_ready_s','upstream_calibration_lower_s','upstream_calibration_upper_s','after_both_ready_s','bytes']},
     'max_causal_order_violation_s':max([i['causal_order_violation_s'] for i in items],default=0)})
 result={**run,'calibration_intervals':[t['calibration'] for t in traces],'missing':missing,'rows':rows,'matches':matched}
 (ROOT/'analysis'/(run['label']+'_rendezvous.json')).write_text(json.dumps(result,indent=2))
 brief={k:v for k,v in run.items()}
 brief['global_epoch_mean_s']=statistics.mean(r['global_epoch_s'] for r in rows)
 brief['anchor_halfwidth_max_s']=max((t['calibration']['host_upper_s']-t['calibration']['host_lower_s'])/2 for t in traces)
 for channel in ['state','parameter']:
  selected=[r for r in rows if r['channel']==channel]
  brief[channel]={key:statistics.mean(r[key] for r in selected) for key in ['receive_span_s','upstream_not_ready_s','upstream_calibration_lower_s','upstream_calibration_upper_s','after_both_ready_s']}
 brief['missing']=len(missing);brief['max_causal_order_violation_s']=max(r['max_causal_order_violation_s'] for r in rows)
 results.append(brief)
 print(json.dumps(brief),flush=True)
(ROOT/'analysis'/(args.tag+'_rendezvous_summary.json')).write_text(json.dumps(results,indent=2))
