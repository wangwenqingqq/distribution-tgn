"""Two-rank matched-readiness transport control; CPU/GPU launch and copies included."""
import os,time,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch,torch.distributed as dist
from packed_parameters import install
from parameter_arena import bind
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');control=dist.new_group(backend='gloo')
output=Path(os.environ['TRANSPORT_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
parent=Path('/home/data/wangxuran/tgn_pipeflash_20260909')
state=torch.load(parent/'runs/stable_b600_p4_native_s2026_e3/results/rank0/initial_model.pt',map_location='cpu',weights_only=True)
shapes=[tuple(t.shape) for t in state.values()];assert len(shapes)==24
def native_send(tensors,peer,group=None):
 reqs=dist.batch_isend_irecv([dist.P2POp(dist.isend,t,peer,group) for t in tensors])
 for req in reqs:req.wait()
def native_recv(tensors,peer,group=None):
 reqs=dist.batch_isend_irecv([dist.P2POp(dist.irecv,t,peer,group) for t in tensors])
 for req in reqs:req.wait()
# Under identical gradients, the contiguous allocation must preserve Adam exactly.
a=torch.nn.ParameterList([torch.nn.Parameter(t.to(rank).clone()) for t in state.values()])
b=torch.nn.ParameterList([torch.nn.Parameter(t.to(rank).clone()) for t in state.values()])
original_ids=[id(p) for p in b.parameters()];bind(b)
assert original_ids==[id(p) for p in b.parameters()]
oa=torch.optim.Adam(a.parameters(),lr=0.0002);ob=torch.optim.Adam(b.parameters(),lr=0.0002)
for step in range(5):
    for i,(pa,pb) in enumerate(zip(a.parameters(),b.parameters())):
        grad=torch.full_like(pa,(i+1)*(step+1)/1024);pa.grad=grad;pb.grad=grad.clone()
    oa.step();ob.step()
    assert all(torch.equal(pa,pb) for pa,pb in zip(a.parameters(),b.parameters()))
optimizer_bitwise_passed=True
records=[]
for repeat in range(3):
 for mode in (['native','packed','arena'] if repeat%2==0 else ['arena','packed','native']):
  ns={'send':native_send,'recv':native_recv}
  if mode=='packed':install(ns)
  tensors=[torch.empty(shape,device=rank,dtype=torch.float32) for shape in shapes]
  if mode=='arena':
   module=torch.nn.ParameterList([torch.nn.Parameter(t,requires_grad=False) for t in tensors]);bind(module)
   tensors=[p.data for p in module.parameters()]
  payload=[module._parameter_arena] if mode=='arena' else tensors
  for i,t in enumerate(tensors):t.fill_(i+repeat/4 if rank==0 else -1)
  torch.cuda.synchronize();dist.barrier(group=control)
  times=[]
  for iteration in range(120):
   torch.cuda.synchronize();dist.barrier(group=control)
   start=time.perf_counter()
   if rank==0:ns['send'](payload,1)
   else:ns['recv'](payload,0)
   torch.cuda.synchronize();end=time.perf_counter()
   if iteration>=20:times.append({'begin_s':start,'end_s':end,'wall_s':end-start})
  correct=all(torch.equal(t,torch.full_like(t,i+repeat/4)) for i,t in enumerate(tensors))
  assert correct
  records.append({'mode':mode,'repeat':repeat,'correct':correct,'adam_bitwise_under_identical_gradients':optimizer_bitwise_passed,'shape_count':len(shapes),'bytes':sum(t.numel()*4 for t in tensors),'times':times})
(output/f'rank{rank}.json').write_text(json.dumps(records,indent=2))
dist.barrier(group=control);dist.destroy_process_group()
