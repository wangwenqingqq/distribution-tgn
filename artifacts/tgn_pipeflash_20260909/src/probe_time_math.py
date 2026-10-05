import torch,json
from pathlib import Path
torch.set_float32_matmul_precision('highest')
r=Path("/home/data/wangxuran/tgn_pipeflash_20260909")
d=torch.load(r/'runs/qualified_b600_p1_native_s2026_e3/results/rank0/final_model.pt',map_location='cuda',weights_only=True)
w=d['layers.l0h0.time_enc.w.weight'];b=d['layers.l0h0.time_enc.w.bias'];records=[]
for n in [16,1280,18000]:
 for scale in [10000.0,2678373.0]:
  dt=torch.linspace(0,scale,n,device='cuda').reshape(-1,1)
  native=torch.nn.functional.linear(dt,w,b)
  separate=dt*w.T+b
  fma=(dt.double()*w.T.double()+b.double()).float()
  records.append({'n':n,'scale':scale,'linear_vs_separate_max':float((native-separate).abs().max()),
    'linear_vs_fma_max':float((native-fma).abs().max()),'linear_vs_separate_unequal':int((native!=separate).sum()),
    'linear_vs_fma_unequal':int((native!=fma).sum())})
print(json.dumps(records,indent=2));(r/'analysis/time_math_probe.json').write_text(json.dumps(records,indent=2)+'\n')
