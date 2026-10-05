"""Compare every paired initialization and retain trained trajectory differences."""
from pathlib import Path
import json
import torch
from run_campaign import CONFIGS

ROOT=Path(__file__).resolve().parents[1]
load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
rows=[]
for seed in [4101,4102,4103]:
    for rank in range(2):
        base=ROOT/'runs'/f'short_lastfm_p2_baseline_s{seed}_e3_v1/results'/f'rank{rank}'
        initial=load(base/'initial_model.pt')
        final=load(base/'final_model.pt')
        states=load(base/'final_memory.pt')
        for name,_,_,_ in CONFIGS:
            path=ROOT/'runs'/f'short_lastfm_p2_{name}_s{seed}_e3_v1/results'/f'rank{rank}'
            other_initial=load(path/'initial_model.pt')
            assert initial.keys()==other_initial.keys()
            assert all(torch.equal(initial[k],other_initial[k]) for k in initial),(seed,rank,name)
            row={'seed':seed,'rank':rank,'name':name,'initial_bitwise_equal':True}
            for label,reference,filename in [('parameters',final,'final_model.pt'),('states',states,'final_memory.pt')]:
                other=load(path/filename);assert reference.keys()==other.keys()
                assert all(torch.isfinite(x).all() for x in other.values())
                row[label+'_bitwise_equal']=all(torch.equal(reference[k],other[k]) for k in reference)
                row[label+'_max_abs']=max(float((reference[k]-other[k]).abs().max()) for k in reference)
            rows.append(row)
            print(seed,rank,name,row['parameters_bitwise_equal'],row['states_bitwise_equal'],flush=True)
report={'passed':True,'rows':rows,'scope':'All 30 rank initializations match their paired baseline and final tensors are finite. Final equality is reported, not required for Flash variants. Frozen-state equality alone is not forward/gradient or convergence qualification.'}
(ROOT/'analysis/trained_model_checks.json').write_text(json.dumps(report,indent=2)+'\n')
