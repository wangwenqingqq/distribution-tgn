import torch,json,hashlib,argparse
from pathlib import Path
r=Path('/home/data/wangxuran/tgn_pipeflash_causal_20260911')
parser=argparse.ArgumentParser();parser.add_argument('--version',default='v1');args=parser.parse_args()
labels=[f'prefix_native_frozen_{args.version}_p8_native_baseline_s2026_e2',f'prefix_candidate_frozen_{args.version}_p8_native_state_first_s2026_e2',f'prefix_candidate_frozen_{args.version}_p8_flash_state_first_s2026_e2']
rows=[];load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
base=r/'runs'/labels[0]/'results/rank0'
a=load(base/'final_memory.pt');init=load(base/'initial_model.pt')
for label in labels:
 p=r/'runs'/label/'results/rank0'
 if not (p/'summary.json').exists():continue
 final=load(p/'final_model.pt');mem=load(p/'final_memory.pt')
 state={k:{'bitwise_equal':torch.equal(a[k],mem[k]),'max_abs':float((a[k]-mem[k]).abs().max())} for k in a}
 audits=[json.loads(x.read_text()) for x in (r/'runs'/label/'results').glob('rank*/prefix_audit.json')]
 row={'label':label,'weights_unchanged':all(torch.equal(init[k],final[k]) for k in init),
 'memory_fields':state,'snapshot_calls':sum(len(xs) for xs in audits),
 'support_reads':sum(x['support_nodes'] for xs in audits for x in xs),
 'future_reads':sum(x['future_mailbox_timestamps']+x['future_memory_timestamps'] for xs in audits for x in xs)}
 rows.append(row);print(row)
 assert row['weights_unchanged'] and row['future_reads']==0

(r/('analysis/prefix_frozen_'+args.version+'_qualification.json')).write_text(json.dumps({'rows':rows,'complete':len(rows)==3,'scope':'Two complete training epochs plus warm-up and validation, frozen model weights. All final state tensors bitwise equal across baseline/native-candidate/flash-candidate; audited support timestamps never exceed current batch maximum. Trained trajectory identity is not asserted.'},indent=2))

assert all(all(v['bitwise_equal'] for v in row['memory_fields'].values()) for row in rows),'Frozen-weight state trajectory mismatch'
