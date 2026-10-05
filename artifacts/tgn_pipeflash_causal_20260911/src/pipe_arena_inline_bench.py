"""Launch the archived PipeTGL entrypoint with explicit compatibility and timing adapters.

This preserves the native multi-GPU training algorithm, including its warm-up
updates and final-local-batch omission. It reports those behaviors rather than
silently changing the amount of training work.
"""
from __future__ import annotations
import time
PROCESS_START=time.perf_counter()
import argparse,copy,functools,json,os,sys,types,faulthandler,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor/PipeTGL'),str(ROOT/'vendor/GNNFlow'),
              str(ROOT/'vendor/dgl/python'),str(ROOT/'vendor/flash-tgn/python'),str(ROOT/'pydeps')]
os.environ.setdefault('DGLBACKEND','pytorch')
import torch
import torch.distributed as dist
import dgl.function as fn
import dgl.utils.shared_mem as shared_mem
if not hasattr(fn,'copy_src'):fn.copy_src=fn.copy_u

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True);p.add_argument('--batch',type=int,default=600)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--variant',choices=['native','flash'],default='native')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--rendezvous',action='store_true')
    p.add_argument('--schedule',choices=['baseline','state_first'],default='baseline')
    p.add_argument('--delay-stage',choices=['state','compute'],default='state')
    p.add_argument('--delay-ms',type=float,default=0)
    p.add_argument('--target-ap',type=float,default=0)
    p.add_argument('--target-consecutive',type=int,default=2)
    p.add_argument('--drain-communications',action='store_true')
    p.add_argument('--control-barrier',choices=['native','gloo'],default='native')
    p.add_argument('--complete-batches',action='store_true')
    p.add_argument('--deterministic-mailbox',action='store_true')
    args=p.parse_args();args.inline_state_send=True;args.parameter_transport='arena';world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    out=Path(args.output)/f'rank{rank}';out.mkdir(parents=True,exist_ok=False)
    stack_log=(out/'thread_stacks.txt').open('w')
    faulthandler.enable(file=stack_log)
    faulthandler.dump_traceback_later(35,repeat=True,file=stack_log)
    # Every run uses distinct POSIX shared-memory names to avoid colliding with other experiments.
    token='pf_'+Path(args.output).parent.name+'_'+Path(args.output).name
    make_shm=shared_mem.create_shared_mem_array;get_shm=shared_mem.get_shared_mem_array
    shared_mem.create_shared_mem_array=lambda name,shape,dtype:make_shm(token+'_'+name,shape,dtype)
    shared_mem.get_shared_mem_array=lambda name,shape,dtype:get_shm(token+'_'+name,shape,dtype)
    import config
    config._tgnn_default_config=copy.deepcopy(config._tgnn_default_config)
    config._tgnn_default_config.update(batch_size=args.batch,dropout=0.0,att_dropout=0.0,
                                     num_layers=1,fanouts=[10])
    from modules.tgnn import TGNN
    from modules.memory import Memory
    from pipe_flash_adapter import install_flash_kernels,transfer_mfgs_with_descriptors
    model_box=[];phase={'name':'setup'};work=[];events=[]
    original_init=TGNN.__init__
    def init(self,*a,**kw):
        original_init(self,*a,**kw);model_box.append(self)
        if args.variant=='flash':install_flash_kernels(self)
        torch.save(self.state_dict(),out/'initial_model.pt')
    TGNN.__init__=init
    if args.deterministic_mailbox:
        import inspect,textwrap
        for method_name in ['update_memory_and_mail','update_memory','update_memory_and_send']:
            method=getattr(TGNN,method_name)
            method_source=textwrap.dedent(inspect.getsource(method))
            old='perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)'
            new="perm = inv.new_full((uni.size(0),), -1).scatter_reduce_(0, inv, perm, reduce='amax', include_self=True)"
            if method_source.count(old)!=2:raise RuntimeError('Unexpected duplicate-node selection sites')
            method_source=method_source.replace(old,new)
            method_ns=dict(method.__globals__)
            exec(compile(method_source,'<last_event_'+method_name+'>','exec'),method_ns)
            setattr(TGNN,method_name,method_ns[method_name])
            (out/(method_name+'_last_event.py')).write_text(method_source)
    original_update=TGNN.update_memory_and_send
    def update(self,b,length,rank_arg,world_arg,group,*a,**kw):
        start=time.perf_counter()
        if world_arg==1:
            # A single GPU has no adjacent peer: use the same target update, with all writes committed locally.
            mem=a[0] if len(a)>0 else kw.get('mem')
            mail=a[1] if len(a)>1 else kw.get('mail')
            result=original_update(self,b,length,rank_arg,world_arg,group,mem,mail,None,None,
                                   edge_feats=kw.get('edge_feats'))
        else:result=original_update(self,b,length,rank_arg,world_arg,group,*a,**kw)
        work.append({'phase':phase['name'],'positive_edges':length//2,
                     'dispatch_start_s':start-PROCESS_START,'dispatch_seconds':time.perf_counter()-start})
        return result
    TGNN.update_memory_and_send=update
    if world==1:
        def recv_single(self,iteration_now,rank_arg,world_arg,device,group=None,src=-1):
            nodes=self.pull_msg[iteration_now//world_arg]
            cached=self.recv_msg[iteration_now//world_arg]
            if cached is not None:nodes=torch.cat([cached,nodes]).sort().values
            return self.node_memory[nodes].to(device),self.mailbox[nodes].to(device)
        Memory.recv_mem=recv_single
        new_group=dist.new_group
        def single_group(ranks=None,*a,**kw):
            return new_group(sorted(set(ranks)) if ranks is not None else None,*a,**kw)
        dist.new_group=single_group

    if args.control_barrier=='gloo':
        original_pg_init=dist.init_process_group
        original_barrier=dist.barrier
        control_group=[]
        def init_with_control(*a,**kw):
            result=original_pg_init(*a,**kw)
            control_group.append(dist.new_group(backend='gloo'))
            return result
        def control_barrier(group=None,async_op=False,device_ids=None):
            if group is None:
                torch.cuda.synchronize()
                return original_barrier(group=control_group[0],async_op=async_op)
            return original_barrier(group=group,async_op=async_op,device_ids=device_ids)
        dist.init_process_group=init_with_control
        dist.barrier=control_barrier

    if args.complete_batches:
        import inspect,textwrap
        overlap_source=textwrap.dedent(inspect.getsource(Memory.findOverlapMem))
        overlap_source=overlap_source.replace('for all_nodes, _, _ in data_loader:','for all_nodes, _, eid in data_loader:')
        overlap_source=overlap_source.replace('all_nodes = all_nodes[:length]','all_nodes = all_nodes[:2*len(eid)]')
        overlap_ns=dict(Memory.findOverlapMem.__globals__)
        exec(compile(overlap_source,'<complete_partial_overlap>','exec'),overlap_ns)
        Memory.findOverlapMem=overlap_ns['findOverlapMem']
        (out/'complete_partial_overlap.py').write_text(overlap_source)
        from modules.IOProcess import IOProcess
        io_init=IOProcess.__init__
        @functools.wraps(io_init)
        def full_io_init(self,*a,**kw):
            bound=inspect.signature(io_init).bind(self,*a,**kw)
            bound.arguments['cnt_iterations']+=bound.arguments['local_world_size']
            return io_init(*bound.args,**bound.kwargs)
        IOProcess.__init__=full_io_init

    source_path=ROOT/'vendor/PipeTGL/scripts/pipeTrain3.py'
    source=source_path.read_text()
    source=source.replace("data_path = '/data/TGL'","data_path = "+repr(str(ROOT/'data/pipe')))
    # The entrypoint performs feature I/O directly. Its unused q_input consumer has no producer.
    source=source.replace('    fetchThread.start()',
                          '    # PipeFlash launch adapter: unused queue consumer disabled for clean termination.')
    if args.complete_batches:
        import re
        pattern=r'(?m)^( +)try:\n\1    next_target_nodes, next_ts, next_eid = next\(train_iter\)\n\1except StopIteration:\n\1    break'
        def allow_last(m):
            i=m.group(1)
            return i+'has_next = True\n'+m.group(0).replace(i+'    break',i+'    has_next = False')
        source,n=re.subn(pattern,allow_last,source)
        if n!=2:raise RuntimeError(f'Expected two native prefetch loops, found {n}')
        source=source.replace('            sampling(next_target_nodes, next_ts, next_eid)','            if has_next: sampling(next_target_nodes, next_ts, next_eid)')
        source=source.replace('        sampling(next_target_nodes, next_ts, next_eid)','        if has_next: sampling(next_target_nodes, next_ts, next_eid)')
        source=source.replace('if iteration_now + 2*args.local_world_size < len(train_loader):','if has_next:')
        source=source.replace('iteration_now+1+args.world_size','iteration_now+1')
        source,n_last=re.subn(r'(?m)^(( +)cache_node_ratio_sum \+= cache\.cache_node_ratio\n\2(?:# )?total_samples \+= num_target_nodes\n\2i \+= 1)$',lambda m:m.group(0)+'\n'+m.group(2)+'if not has_next: break',source)
        if n_last!=2:raise RuntimeError('Expected two final-batch exit sites')
    from rendezvous_probe import schedule_source
    source=schedule_source(source,args.schedule,args.delay_stage,args.delay_ms)
    source=source.replace('    model.to(device)','    model.to(device)\n    bind_parameter_arena(model)')
    assert source.count('[param.data.clone() for param in model.parameters()]')==2
    assert source.count('[param.data for param in model.parameters()]')==2
    source=source.replace('[param.data.clone() for param in model.parameters()]','[model._parameter_arena.clone()]')
    source=source.replace('[param.data for param in model.parameters()]','[model._parameter_arena]')

    source=source.replace('    for e in range(args.epoch):\n','    for e in range(args.epoch):\n        record_epoch_begin(e)\n')
    source=source.replace('        epoch_time = time.time() - start_time','        record_epoch_end(e)\n        epoch_time = time.time() - start_time')
    val_marker='        val_ap, val_auc = val_res[0].item(), val_res[1].item()'
    assert source.count(val_marker)==1
    source=source.replace(val_marker,val_marker+'\n        reached_target = record_validation(e, val_ap, val_auc)')
    if args.target_ap:
        # Keep the specific I/O worker handle, and terminate it after all ranks
        # observe the same globally reduced target at the ordinary epoch boundary.
        source=source.replace('    process.start()','    process.start()\n    io_process = process')
        early_marker='    if args.rank == 0:\n        logging.info(\'Avg epoch time:'
        assert source.count(early_marker)==1
        source=source.replace(early_marker,'        if reached_target: break\n\n    if io_process.is_alive(): io_process.terminate()\n    io_process.join(timeout=2)\n\n'+early_marker)
    (out/'executed_entrypoint.py').write_text(source)
    ns={'__name__':'pipetgl_archived_entry','__file__':str(source_path)}
    epoch_intervals=[];epoch_open={};probe=None;validation=[];target_streak=0
    def record_epoch_begin(e):
        if probe is not None:probe.epoch=e
        epoch_open.update(start=time.perf_counter(),work=len(work))
        if args.profile:torch.cuda.nvtx.range_push('epoch:train:'+str(e))
    def record_epoch_end(e):
        end=time.perf_counter()
        if args.profile:torch.cuda.nvtx.range_pop()
        epoch_intervals.append({'epoch':e,'start_s':epoch_open['start']-PROCESS_START,
            'wall_seconds':end-epoch_open['start'],
            'positive_edges':sum(x['positive_edges'] for x in work[epoch_open['work']:])})
    def record_validation(e,ap,auc):
        nonlocal target_streak
        target_streak=target_streak+1 if args.target_ap and ap>=args.target_ap else 0
        reached=bool(args.target_ap and target_streak>=args.target_consecutive)
        validation.append({'epoch':e,'ap':ap,'auc':auc,'absolute_perf_s':time.perf_counter(),
                           'seconds_since_process_start':time.perf_counter()-PROCESS_START,
                           'target_streak':target_streak,'reached_target':reached})
        return reached
    ns['record_validation']=record_validation
    ns['record_epoch_begin']=record_epoch_begin;ns['record_epoch_end']=record_epoch_end
    saved_argv=sys.argv
    sys.argv=[str(source_path),'--data','WIKI','--epoch',str(args.epochs),'--seed',str(args.seed),
              '--num-workers','0','--print-freq','1000000']
    try:exec(compile(source,str(source_path),'exec'),ns)
    finally:sys.argv=saved_argv
    if world==1:
        ns['send']=lambda *a,**kw:None
        ns['recv']=lambda *a,**kw:True
    if args.variant=='flash':ns['mfgs_to_cuda']=lambda mfgs,device:transfer_mfgs_with_descriptors(mfgs,device,10)
    def wrap_phase(name):
        original=ns[name]
        @functools.wraps(original)
        def run(*a,**kw):
            previous=phase['name'];phase['name']=name
            start=time.perf_counter()
            if args.profile:torch.cuda.nvtx.range_push('phase:'+name)
            try:return original(*a,**kw)
            finally:
                if name=='warm_up' and args.drain_communications:torch.cuda.synchronize()
                if args.profile:torch.cuda.nvtx.range_pop()
                events.append({'phase':name,'start_s':start-PROCESS_START,'wall_seconds':time.perf_counter()-start})
                phase['name']=previous
        ns[name]=run
    for name in ['warm_up','train','evaluate']:wrap_phase(name)
    if args.profile:
        def profile_method(obj,name,label):
            original=getattr(obj,name)
            @functools.wraps(original)
            def traced(*a,**kw):
                begin=time.perf_counter()
                torch.cuda.nvtx.range_push(label)
                try:return original(*a,**kw)
                finally:
                    torch.cuda.nvtx.range_pop()
                    events.append({'phase':phase['name'],'label':label,
                                   'start_s':begin-PROCESS_START,
                                   'cpu_dispatch_seconds':time.perf_counter()-begin,'thread_id':threading.get_ident()})
            setattr(obj,name,traced)
        for obj,name,label in [(TGNN,'update_memory_and_send','target_state_update_and_send'),
            (TGNN,'prepare_input','support_state_read_and_update'),(TGNN,'forward','attention_and_predict'),
            (Memory,'recv_mem','state_receive_and_host_gather'),(Memory,'send_mem','state_send_launch'),
            (torch.Tensor,'backward','backward'),(torch.optim.Adam,'step','optimizer_step')]:
            profile_method(obj,name,label)
        from gnnflow.temporal_sampler import TemporalSampler
        profile_method(TemporalSampler,'sample','temporal_sampling')
        for name,label in [('recv','parameter_receive_or_order_signal'),('send','parameter_send_or_order_signal'),('mfgs_to_cuda','sampled_topology_and_descriptor_transfer')]:
            original=ns[name]
            def profiled(*a,_original=original,_label=label,**kw):
                begin=time.perf_counter();torch.cuda.nvtx.range_push(_label)
                try:return _original(*a,**kw)
                finally:
                    torch.cuda.nvtx.range_pop()
                    events.append({'phase':phase['name'],'label':_label,'start_s':begin-PROCESS_START,
                                   'cpu_dispatch_seconds':time.perf_counter()-begin,'thread_id':threading.get_ident()})
            ns[name]=profiled
    if world>1:
        from inline_state import install as install_inline_state
        install_inline_state()
    if args.rendezvous and world>1:
        from rendezvous_probe import install
        probe=install(ns,rank,out,phase)
    from parameter_arena import bind as bind_parameter_arena
    ns['bind_parameter_arena']=bind_parameter_arena
    torch.set_float32_matmul_precision('highest')
    start=time.perf_counter()
    ns['main']()
    torch.cuda.synchronize()
    finished=time.perf_counter()
    if probe is not None:probe.finish()
    model=model_box[0]
    torch.save(model.state_dict(),out/'final_model.pt')
    torch.save({k:getattr(model.memory,k).detach().cpu() for k in ['node_memory','node_memory_ts','mailbox','mailbox_ts']},out/'final_memory.pt')
    result={'config':vars(args),'world_size':world,'rank':rank,'process_start_perf_s':PROCESS_START,
            'import_and_adapter_seconds':start-PROCESS_START,'native_main_wall_seconds':finished-start,
            'total_seconds_before_artifact_serialization':finished-PROCESS_START,
            'training_work':work,'phase_intervals':events,'epoch_intervals':epoch_intervals,'validation':validation,
            'parameters_finite':all(torch.isfinite(p).all().item() for p in model.parameters()),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'adaptations':['dataset path','run-specific shared-memory names','DGL copy_src alias',
                           'unused queue-consumer disabled','one-GPU ring bypass when applicable',
                           'fixed batch/dimensions/dropout/seed','actual processed-edge accounting'],
            'limitations':['native warm-up performs an extra optimization pass',
                           'complete batches' if args.complete_batches else 'native final local batch omitted',
                           'deterministic last-event mailbox' if args.deterministic_mailbox else 'native duplicate-node scatter selection',
                           'native learning rate scales with sqrt(world size)',
                           'native per-rank Adam and parameter-staleness semantics',
                           'Flash variant uses strict-math gather and original Flash segmented attention' if args.variant=='flash' else 'GNNFlow original attention']}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['training_work']},indent=2),flush=True)
    if not result['parameters_finite']:raise RuntimeError('Nonfinite model parameters')
    faulthandler.cancel_dump_traceback_later()
    dist.destroy_process_group()
if __name__=='__main__':
    try:main()
    except BaseException:
        import multiprocessing
        for child in multiprocessing.active_children():child.terminate()
        for child in multiprocessing.active_children():child.join(timeout=2)
        raise
