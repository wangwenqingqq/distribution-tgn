"""Reproduce the paper arithmetic and a finite homogeneous pipeline model."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def finite(n, batches, a, r):
    state_finish=[]; finish=[]
    for i in range(batches):
        state_ready=state_finish[i-1] if i else 0.0
        gpu_ready=finish[i-n] if i>=n else 0.0
        state_finish.append(max(state_ready,gpu_ready)+a)
        finish.append(state_finish[-1]+r)
    return {"batches":batches,"makespan":max(finish,default=0),
            "last_state_interval":state_finish[-1]-state_finish[-2] if batches>1 else None}
results=[]
for speed in [1.0,1.5]:
    a=13.5; local=12.4+a+8.6+37.2/speed+9.2
    for n in [1,2,4,8]:
        wait=max(n*a-local,0.0)
        results.append({"source":"Table 2 REDDIT; sensitivity analysis, not a measured run",
            "fwd_bwd_speedup_assumption":speed,"gpus":n,"a":a,"local_no_wait":local,
            "bubble":wait,"bubble_fraction":wait/(local+wait),
            "ideal_global_interval":max(a,local/n),
            "finite":[finite(n,k,a,local-a) for k in [8,32,1024]]})
(ROOT/'analysis/theory_probe.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
