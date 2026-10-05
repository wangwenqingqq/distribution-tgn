import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'runs/transport_control_v3/results';a,b=[json.loads((base/f'rank{k}.json').read_text()) for k in range(2)]
rows=[]
for x,y in zip(a,b):
 assert x['mode']==y['mode'] and x['repeat']==y['repeat'] and x['correct'] and y['correct']
 xs=[max(i['end_s'],j['end_s'])-min(i['begin_s'],j['begin_s']) for i,j in zip(x['times'],y['times'])]
 rows.append({'mode':x['mode'],'repeat':x['repeat'],'messages':len(xs),'bytes':x['bytes'],
 'global_wall_mean_us':statistics.mean(xs)*1e6,'global_wall_median_us':statistics.median(xs)*1e6,
 'both_endpoints_bitwise_correct':True,'adam_bitwise_under_identical_gradients':x['adam_bitwise_under_identical_gradients'] and y['adam_bitwise_under_identical_gradients']})
result={'rows':rows,'scope':'Both endpoints have ready inputs before a Gloo barrier outside the timed region. Timed span includes host launch, GPU transfer and required packing/copying; global earliest start through latest CUDA synchronization. This is a warm transport calibration, not a pure wire time or training speedup. 20 untimed warm-up iterations then 100 measured per repeat; three order-alternating repetitions.'}
(ROOT/'analysis/transport_control_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
