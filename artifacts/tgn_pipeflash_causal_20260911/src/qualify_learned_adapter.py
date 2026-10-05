"""Check large deltas again using learned weights from complete training runs."""
import argparse,copy,json
from pathlib import Path
import torch
from qualify_adapter import block_from_arrays,clone_block,metric,TransfomerAttentionLayer,prepare_descriptor,flash_forward
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_float32_matmul_precision('highest');torch.manual_seed(901);torch.set_num_threads(1)
    records=[]
    for variant in ['native','flash']:
        checkpoint=ROOT/'runs'/f'qualified_b600_p1_{variant}_s2026_e3/results/rank0/final_model.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        for scale in [10000.0,2678373.0]:
            b=block_from_arrays([10]*128)
            b.edata['dt']=torch.linspace(0,scale,b.num_edges())
            desc=prepare_descriptor(b,10).to('cuda')
            layer=TransfomerAttentionLayer(100,172,100,100,2,0.0,0.0)
            layer.load_state_dict({k[len('layers.l0h0.'):]:v for k,v in state.items() if k.startswith('layers.l0h0.')})
            layer=layer.cuda();other=copy.deepcopy(layer)
            b1=clone_block(b,'cuda');b2=clone_block(b,'cuda')
            y1=layer(b1);y2=flash_forward(other,b2,desc)
            checks={'output':metric(y1,y2,5e-5,4e-4)}
            upstream=torch.randn_like(y1);(y1*upstream).sum().backward();(y2*upstream).sum().backward()
            checks['input_h']=metric(b1.srcdata['h'].grad,b2.srcdata['h'].grad,5e-5,4e-4)
            checks['edge_features']=metric(b1.edata['f'].grad,b2.edata['f'].grad,5e-5,4e-4)
            for (k,a),(k2,z) in zip(layer.named_parameters(),other.named_parameters()):
                assert k==k2;checks['parameter:'+k]=metric(a.grad,z.grad,5e-5,4e-4)
            row={'checkpoint':str(checkpoint),'max_time_delta':scale,'checks':checks,'passed':all(v['passed'] for v in checks.values())}
            records.append(row);print(variant,scale,row['passed'],checks['output'],flush=True)
    result={'passed':all(v['passed'] for v in records),'cases':records}
    (out/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise RuntimeError('Learned-weight qualification failed')
if __name__=='__main__':main()
