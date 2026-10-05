"""Compare the adapter to the unchanged GNNFlow attention layer on real CPU/GPU DGL blocks."""
import argparse,copy,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor/dgl/python'),str(ROOT/'vendor/GNNFlow'),
              str(ROOT/'vendor/flash-tgn/python'),str(ROOT/'pydeps')]
os.environ.setdefault('DGLBACKEND','pytorch')
import numpy as np
import torch
import dgl
import dgl.function as fn
# DGL renamed this operator; the implementation and semantics are the same.
if not hasattr(fn,'copy_src'):fn.copy_src=fn.copy_u
from gnnflow.models.modules.layers import TransfomerAttentionLayer
from pipe_flash_adapter import prepare_descriptor,torch_packed_forward,flash_forward

def metric(a,b,atol,rtol):
    if a is None or b is None:return {'passed':a is None and b is None,'presence':(a is not None,b is not None)}
    delta=(a-b).detach().float()
    return {'passed':torch.allclose(a,b,atol=atol,rtol=rtol),
            'max_abs':float(delta.abs().max()) if delta.numel() else 0.0,
            'relative_l2':float(delta.norm()/a.detach().float().norm().clamp_min(1e-12))}
def block_from_arrays(counts,seed=2026):
    gen=torch.Generator().manual_seed(seed);q=len(counts);e=sum(counts)
    dst=torch.repeat_interleave(torch.arange(q),torch.tensor(counts))
    order=torch.randperm(e,generator=gen);dst=dst[order]
    b=dgl.create_block((torch.arange(q,q+e),dst),num_src_nodes=q+e,num_dst_nodes=q)
    b.srcdata['h']=torch.randn(q+e,100,generator=gen)
    b.edata['f']=torch.randn(e,172,generator=gen)
    b.edata['dt']=torch.rand(e,generator=gen)*10
    return b
def clone_block(b,device):
    result=dgl.create_block(b.edges(),num_src_nodes=b.num_src_nodes(),num_dst_nodes=b.num_dst_nodes()).to(device)
    result.srcdata['h']=b.srcdata['h'].to(device).detach().clone().requires_grad_(True)
    result.edata['f']=b.edata['f'].to(device).detach().clone().requires_grad_(True)
    result.edata['dt']=b.edata['dt'].to(device).clone()
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(2026)
    torch.set_float32_matmul_precision('highest')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);records=[]
    cases=[('no_edges',[0,0,0,0]),('mixed',[0,1,10,3,0,2]),('full',[10]*128),
           ('mostly_empty',[0]*96+[1,2,3,10]*8),('large_time_deltas',[10]*128)]
    for name,counts in cases:
        b=block_from_arrays(counts)
        if name=='large_time_deltas':b.edata['dt']=torch.linspace(0,2678373,sum(counts))
        desc=prepare_descriptor(b,10).to(args.device)
        layer=TransfomerAttentionLayer(100,172,100,100,2,0.0,0.0).to(args.device)
        layer2=copy.deepcopy(layer);b1=clone_block(b,args.device);b2=clone_block(b,args.device)
        y1=layer(b1);y2=(flash_forward if args.device=='cuda' else torch_packed_forward)(layer2,b2,desc)
        atol,rtol=(5e-5,4e-4) if args.device=='cuda' else (2e-5,2e-4)
        checks={'output':metric(y1,y2,atol,rtol)}
        if y1.requires_grad:
            upstream=torch.randn_like(y1);(y1*upstream).sum().backward();(y2*upstream).sum().backward()
            checks['input_h']=metric(b1.srcdata['h'].grad,b2.srcdata['h'].grad,atol,rtol)
            checks['edge_features']=metric(b1.edata['f'].grad,b2.edata['f'].grad,atol,rtol)
            for (key,p),(key2,p2) in zip(layer.named_parameters(),layer2.named_parameters()):
                assert key==key2;checks['parameter:'+key]=metric(p.grad,p2.grad,atol,rtol)
        record={'case':name,'queries':len(counts),'edges':sum(counts),
                'atol':atol,'rtol':rtol,'checks':checks,'passed':all(v['passed'] for v in checks.values())}
        records.append(record);print(name,record['passed'],checks['output'],flush=True)
    result={'device':args.device,'dgl':dgl.__version__,'torch':torch.__version__,
        'scope':'one-layer attention outputs, input/edge gradients and all parameter gradients; no optimizer-trajectory equivalence claim',
        'passed':all(r['passed'] for r in records),'cases':records}
    (out/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise RuntimeError('Adapter qualification failed')
if __name__=='__main__':main()
