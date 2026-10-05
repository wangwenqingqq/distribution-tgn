"""Run the native FlashTGN artifact with complete wall timing and optional diagnostic events."""
from __future__ import annotations
import time
PROCESS_START=time.perf_counter()
import argparse,json,os,sys,functools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor/flash-tgn/python'),str(ROOT/'pydeps')]
import numpy as np
import torch
from flash_tgn.config import FlashTGNConfig,set_reproducibility
from flash_tgn.data import load_temporal_graph
from flash_tgn.model import FlashTGN
from flash_tgn.state import FeatureStore,TemporalState
from flash_tgn.temporal_index import TemporalIndex
from flash_tgn.trainer import FlashTGNTrainer

class Meter:
    def __init__(self,enabled):
        self.enabled=enabled;self.records=[];self.events=[]
    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            if not self.enabled:return original(*args,**kwargs)
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();start=time.perf_counter();torch.cuda.nvtx.range_push(label)
            try:return original(*args,**kwargs)
            finally:
                torch.cuda.nvtx.range_pop();end.record()
                self.records.append({'label':label,'cpu_dispatch_start_s':start-PROCESS_START,
                                     'cpu_dispatch_seconds':time.perf_counter()-start})
                self.events.append((begin,end))
        setattr(obj,name,measured)
    def finish(self):
        torch.cuda.synchronize()
        for record,(a,b) in zip(self.records,self.events):record['cuda_event_interval_ms']=a.elapsed_time(b)
        return self.records

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True);ap.add_argument('--batch',type=int,default=600)
    ap.add_argument('--layers',type=int,default=1);ap.add_argument('--epochs',type=int,default=2)
    ap.add_argument('--profile',action='store_true');ap.add_argument('--steps',type=int)
    ap.add_argument('--mode',choices=['online','gpu','cpu'],default='online')
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--evaluate',action='store_true')
    args=ap.parse_args();out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
    cfg=FlashTGNConfig(data='wiki',data_path=str(ROOT/'data/flash'),gpu=0,epochs=args.epochs,
        bsize=args.batch,n_nbrs=10,n_layers=args.layers,dim_time=100,dim_embed=100,num_heads=2,
        dropout=0.0,seed=args.seed,precompute_mode=args.mode,max_train_batches=args.steps,
        train_only=not args.evaluate,model_dir=str(out/'models'))
    set_reproducibility(cfg.seed)
    # Use the same FP32 policy as the adapter qualification and PipeTGL baseline.
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_device(0)
    graph=load_temporal_graph(cfg.data_path,cfg.data,cfg.dim_embed)
    features=FeatureStore(graph,cfg.dim_embed,cfg.device)
    state=TemporalState(graph.num_nodes,cfg.dim_embed,graph.dim_edge,cfg.device)
    index=TemporalIndex(cfg.data_path,cfg.data,graph.num_nodes,cfg.device)
    model=FlashTGN(graph.dim_node,graph.dim_edge,cfg.dim_time,cfg.dim_embed,cfg.num_heads,cfg.dropout,
                   n_layers=cfg.n_layers).to(cfg.device)
    trainer=FlashTGNTrainer(cfg,graph,features,index,state,model)
    meter=Meter(args.profile)
    for obj,name,label in [(trainer,'_build_schedules','topology_precompute'),
        (state,'update_memory_for_batch','state_gather_gru_writeback'),
        (features,'projected_node_features','node_feature_gather'),
        (features,'compact_edge_features','edge_feature_prepare'),
        (model,'forward_embed','attention'),(model,'predict','predictor'),
        (state,'store_raw_messages','mailbox_commit'),
        (trainer.optimizer,'step','optimizer'),(torch.Tensor,'backward','backward')]:
        meter.wrap(obj,name,label)
    epochs=[];native_run=trainer.run_epoch
    def recorded_epoch(*a,**kw):
        torch.cuda.synchronize();start=time.perf_counter()
        result=native_run(*a,**kw)
        torch.cuda.synchronize();end=time.perf_counter()
        start_edge=a[0] if a else kw['start'];end_edge=a[1] if len(a)>1 else kw['end']
        epochs.append({'train':kw.get('train',a[2] if len(a)>2 else None),'start_edge':start_edge,
            'end_edge':end_edge,'edges':end_edge-start_edge,'wall_seconds':end-start,
            'native_loop_seconds':result.loop_time,'loss':result.loss,'ap':result.ap,'auc':result.auc})
        return result
    trainer.run_epoch=recorded_epoch
    torch.cuda.synchronize();ready=time.perf_counter()
    trainer.train()
    torch.cuda.synchronize();finished=time.perf_counter()
    event_records=meter.finish()
    finite=all(torch.isfinite(p).all().item() for p in model.parameters())
    torch.save(model.state_dict(),out/'final_model.pt')
    # This small graph permits a complete state snapshot for later parity checks.
    torch.save({k:getattr(state,k).detach().cpu() for k in ['mem_data','mem_ts','mail_data','mail_ts']},
               out/'final_memory.pt')
    result={'config':vars(args),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'matmul_precision':torch.get_float32_matmul_precision(),
            'setup_seconds_including_imports':ready-PROCESS_START,
            'training_driver_wall_seconds':finished-ready,
            'total_seconds_before_artifact_serialization':finished-PROCESS_START,
            'timing_note':'Driver wall includes native precomputation, reset, train/eval, checkpoint work and final sync. Native loop is reported separately.',
            'diagnostic_events':args.profile,'epochs':epochs,'parameters_finite':finite,
            'memory_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.profile:(out/'events.json').write_text(json.dumps(event_records)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not finite:raise RuntimeError('Nonfinite model')
if __name__=='__main__':main()
