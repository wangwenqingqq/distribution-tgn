"""Nonblocking CUDA-event rendezvous observation; no per-message synchronize."""
import threading,time,json,functools
from collections import defaultdict
import torch
import torch.distributed as dist

class RendezvousProbe:
    def __init__(self,rank,out,phase):
        self.rank=rank;self.out=out;self.phase=phase;self.epoch=-1
        self.lock=threading.Lock();self.anchor=None;self.records=[];self.seq=defaultdict(int)
    def calibrate(self):
        samples=[]
        for _ in range(12):
            ev=torch.cuda.Event(enable_timing=True)
            t0=time.perf_counter();ev.record(torch.cuda.current_stream(self.rank));ev.synchronize();t1=time.perf_counter()
            samples.append((t1-t0,ev,{'host_lower_s':t0,'host_upper_s':t1,'midpoint_s':(t0+t1)/2}))
        _,ev,bounds=min(samples,key=lambda x:x[0])
        return ev,bounds
    def begin(self,channel,direction,peer,tensors):
        with self.lock:
            if self.anchor is None:
                self.anchor,self.calibration=self.calibrate()
            key=(channel,direction,peer);seq=self.seq[key];self.seq[key]+=1
        start=time.perf_counter();ready=torch.cuda.Event(enable_timing=True);ready.record(torch.cuda.current_stream(self.rank))
        item={'channel':channel,'direction':direction,'peer':peer,'sequence':seq,'rank':self.rank,
              'phase':self.phase['name'],'epoch':self.epoch,'host_enter_s':start,'ready_event':ready,
              'bytes':sum(x.numel()*x.element_size() for x in tensors) if tensors is not None else 8,
              'tensor_count':len(tensors) if tensors is not None else 1}
        with self.lock:self.records.append(item)
        return item
    def launched(self,item):item['host_launch_return_s']=time.perf_counter()
    def done(self,item):
        event=torch.cuda.Event(enable_timing=True);event.record(torch.cuda.current_stream(self.rank))
        item['done_event']=event;item['host_wait_return_s']=time.perf_counter()
    def finish(self):
        torch.cuda.synchronize()
        end_anchor,end_calibration=self.calibrate()
        gpu_delta_s=self.anchor.elapsed_time(end_anchor)/1000
        clock_scale=(end_calibration['midpoint_s']-self.calibration['midpoint_s'])/gpu_delta_s
        for item in self.records:
            for key in ['ready','done']:
                event=item.pop(key+'_event',None)
                if event is not None:item['gpu_'+key+'_s']=self.calibration['midpoint_s']+clock_scale*self.anchor.elapsed_time(event)/1000
        (self.out/'rendezvous.json').write_text(json.dumps({'calibration':self.calibration,'end_calibration':end_calibration,'gpu_to_host_clock_scale':clock_scale,'records':self.records},indent=2))

class ObservedWork:
    def __init__(self,work,probe,item,last):self.work=work;self.probe=probe;self.item=item;self.last=last
    def wait(self,*a,**kw):
        result=self.work.wait(*a,**kw)
        if self.last:self.probe.done(self.item)
        return result
    def __getattr__(self,name):return getattr(self.work,name)

def install(ns,rank,out,phase):
    import modules.memory as mm
    probe=RendezvousProbe(rank,out,phase)
    for name,direction in [('send','send'),('recv','recv')]:
        original=ns[name]
        def wrapped(tensors,peer,group=None,_fn=original,_direction=direction):
            item=probe.begin('parameter' if tensors is not None else 'evaluation_order',_direction,peer,tensors)
            result=_fn(tensors,peer,group)
            probe.launched(item)
            if _direction=='recv':probe.done(item)
            return result
        ns[name]=wrapped
    original_send=mm.send;original_recv=mm.recv;original_recv_req=mm.recv_req
    def state_send(tensors,rank_arg,peer,group=None):
        item=probe.begin('state','send',peer,tensors)
        result=original_send(tensors,rank_arg,peer,group);probe.launched(item);return result
    def state_recv(tensors,rank_arg,peer,group=None):
        item=probe.begin('state','recv',peer,tensors)
        result=original_recv(tensors,rank_arg,peer,group);probe.launched(item);probe.done(item);return result
    def state_recv_req(tensors,rank_arg,peer,group=None):
        item=probe.begin('state','recv',peer,tensors)
        reqs=original_recv_req(tensors,rank_arg,peer,group);probe.launched(item)
        return [ObservedWork(w,probe,item,k==len(reqs)-1) for k,w in enumerate(reqs)]
    mm.send=state_send;mm.recv=state_recv;mm.recv_req=state_recv_req
    return probe

def schedule_source(source,schedule,delay_stage,delay_ms):
    # Defer the already selected NEXT local batch's sampling until CURRENT target
    # state is sent. All batches, sampling calls and model operations are preserved.
    if schedule=='state_first':
        import re
        pattern=r'(?m)^([ ]+)if has_next: sampling\(next_target_nodes, next_ts, next_eid\)$'
        source,n=re.subn(pattern,lambda m:m.group(1)+'# Next sampling deferred until after current state release.',source)
        if n!=2:raise RuntimeError(('sampling sites',n))
        marker='updated_memory, overlap_nid, sends_thread1 = model.update_memory_and_send(b, update_length, args.rank, args.world_size, groups[idx], mem, mail, push_msg, send_msg, edge_feats=cache.target_edge_features)'
        if source.count(marker)!=2:raise RuntimeError('state update sites')
        # Same indentation as update: memory is always enabled in this experiment.
        source=re.sub(r'(?m)^([ ]+)'+re.escape(marker)+r'$',lambda m:m.group(0)+'\n'+m.group(1)+'if has_next: sampling(next_target_nodes, next_ts, next_eid)',source)
    if delay_ms:
        import re
        if delay_stage=='state':
            pattern=r'(?m)^([ ]+)(updated_memory, overlap_nid, sends_thread1 = model.update_memory_and_send.*)$'
            source,n=re.subn(pattern,lambda m:m.group(1)+'time.sleep('+repr(delay_ms/1000)+')\n'+m.group(0),source)
        elif delay_stage=='compute':
            pattern=r'(?m)^([ ]+)loss.backward\(\)$'
            source,n=re.subn(pattern,lambda m:m.group(0)+'\n'+m.group(1)+'time.sleep('+repr(delay_ms/1000)+')',source)
        else:raise ValueError(delay_stage)
        if n!=2:raise RuntimeError(('delay sites',n))
    return source
