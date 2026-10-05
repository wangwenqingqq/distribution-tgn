"""Capture support state at the current batch frontier before releasing the next batch."""
import inspect,textwrap,threading,json
class PrefixSnapshot:
    def __init__(self,ns,TGNN,Memory,out,inline=False,audit=False):
        import modules.memory as mm
        self.ns=ns;self.out=out;self.inline=inline;self.mm=mm
        self.native_state_send=Memory.send_mem
        self.native_recv_mem=Memory.recv_mem
        self.native_send=ns['send'];self.native_recv=ns['recv']
        self.pending_eval=None;self.eval_thread=None;self.memories=[]
        self.snapshots=0;self.state_releases=0;self.eval_releases=0;self.audit=audit;self.audit_rows=[]
        if audit:
            transfer=ns['mfgs_to_cuda']
            def with_times(mfgs,*a,**kw):
                times=[float(b.srcdata['ts'][:b.num_dst_nodes()].max()) for g in mfgs for b in g]
                result=transfer(mfgs,*a,**kw)
                for b,t in zip([b for g in result for b in g],times):b._prefix_batch_max_ts=t
                return result
            ns['mfgs_to_cuda']=with_times
        owner=self
        def stage_state(memory,mem,mail,rank,world_size,group=None,dst=-1):
            if mem is None:return None
            if getattr(memory,'_prefix_pending_state',None) is not None:raise RuntimeError('Unreleased preceding state')
            memory._prefix_pending_state=(mem,mail,rank,world_size,group,dst)
            if not any(m is memory for m in owner.memories):owner.memories.append(memory)
            return None
        def ordered_recv_mem(memory,*a,**kw):
            previous=getattr(memory,'_prefix_sender_thread',None)
            if previous is not None:previous.join();memory._prefix_sender_thread=None
            return owner.receive_current_frontier(memory,*a,**kw)
        def stage_send(tensors,peer,group=None):
            if tensors is not None:return owner.native_send(tensors,peer,group)
            if owner.pending_eval is not None:raise RuntimeError('Unreleased evaluation order')
            owner.pending_eval=(peer,group)
        def ordered_recv(tensors,peer,group=None):
            if tensors is None and owner.eval_thread is not None:
                owner.eval_thread.join();owner.eval_thread=None
            return owner.native_recv(tensors,peer,group)
        Memory.send_mem=stage_state;Memory.recv_mem=ordered_recv_mem
        ns['send']=stage_send;ns['recv']=ordered_recv
        original=TGNN.prepare_input
        source=textwrap.dedent(inspect.getsource(original))
        for field,local in [('node_memory','mem'),('node_memory_ts','mem_ts'),('mailbox','mail'),('mailbox_ts','mail_ts')]:
            old=f'{local} = self.memory.{field}[pull_nodes].to(device)'
            new=f'{local} = self.memory.{field}[pull_nodes]'
            if source.count(old)!=1:raise RuntimeError(('support gather site',field))
            source=source.replace(old,new)
        marker='    new_memory = self.memory_updater(mem, mail, mem_ts, mail_ts)'
        if source.count(marker)!=1:raise RuntimeError('support update site')
        source=source.replace(marker,"""    _release_prefix_snapshot(self,b,pull_nodes,mem_ts,mail_ts)
    if input is None:
        mem, mem_ts, mail, mail_ts = [x.to(device) for x in (mem, mem_ts, mail, mail_ts)]

"""+marker)
        scope=dict(original.__globals__,_release_prefix_snapshot=self.release)
        exec(compile(source,'<prefix_snapshot_prepare_input>','exec'),scope)
        TGNN.prepare_input=scope['prepare_input']
        (out/'prefix_snapshot_prepare_input.py').write_text(source)
    def receive_current_frontier(self,memory,iteration_now,rank,world_size,device,group=None,src=-1):
        import torch,numpy as np
        cached_idx=memory.recv_msg[iteration_now//world_size]
        cached_mem=cached_mail=None
        if cached_idx is not None and world_size>1:
            if src==-1:src=(rank-1+world_size)%world_size
            if len(cached_idx)>0:
                cached_mem=torch.empty((len(cached_idx),memory.dim_memory),device=device)
                cached_mail=torch.empty((len(cached_idx),memory.dim_raw_message),device=device)
                reqs=self.mm.recv_req([cached_mem,cached_mail],rank,src,group)
                for req in reqs:req.wait()
                # Work.wait orders CUDA consumption but does not block CPU reads.
                # Observe only this state-ready event before gathering shared CPU rows.
                ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(device));ready.synchronize()
            else:
                self.mm.recv(None,rank,src,group)
        uncached_idx=memory.pull_msg[iteration_now//world_size]
        uncached_mem=memory.node_memory[uncached_idx].to(device)
        uncached_mail=memory.mailbox[uncached_idx].to(device)
        if cached_mem is not None:
            idx=torch.cat((cached_idx,uncached_idx))
            mem=torch.cat((cached_mem,uncached_mem),dim=0)
            mail=torch.cat((cached_mail,uncached_mail),dim=0)
        else:
            idx=uncached_idx;mem=uncached_mem;mail=uncached_mail
        order=np.argsort(idx)
        return mem[order],mail[order]

    def release(self,model,b,nodes,mem_ts,mail_ts):
        self.snapshots+=1
        if self.audit:
            hi=b._prefix_batch_max_ts
            future=mail_ts>hi
            self.audit_rows.append({'batch_max_ts':hi,'support_nodes':len(nodes),
                'future_mailbox_timestamps':int(future.sum()),
                'future_memory_timestamps':int((mem_ts>hi).sum()),
                'max_future_seconds':float((mail_ts[future]-hi).max()) if bool(future.any()) else 0})

        memory=model.memory
        payload=getattr(memory,'_prefix_pending_state',None)
        if payload is not None:
            memory._prefix_pending_state=None
            if self.inline:
                mem,mail,rank,world,group,dst=payload
                if dst==-1:dst=(rank+1)%world
                self.mm.send([mem,mail] if mem.shape[0] else None,rank,dst,group)
            else:
                memory._prefix_sender_thread=self.native_state_send(memory,*payload)
            self.state_releases+=1
        if self.pending_eval is not None:
            peer,group=self.pending_eval;self.pending_eval=None
            if self.inline:self.native_send(None,peer,group)
            else:
                self.eval_thread=threading.Thread(target=self.native_send,args=(None,peer,group))
                self.eval_thread.start()
            self.eval_releases+=1
    def finish(self):
        if self.audit:
            (self.out/'prefix_audit.json').write_text(json.dumps(self.audit_rows,indent=2))
            assert all(x['future_mailbox_timestamps']==0 and x['future_memory_timestamps']==0 for x in self.audit_rows),'Future support state despite prefix snapshot'
        for memory in self.memories:
            assert getattr(memory,'_prefix_pending_state',None) is None
            thread=getattr(memory,'_prefix_sender_thread',None)
            if thread is not None:thread.join()
        assert self.pending_eval is None
        if self.eval_thread is not None:self.eval_thread.join()
        (self.out/'prefix_snapshot.json').write_text(json.dumps({'support_snapshots':self.snapshots,
              'state_releases':self.state_releases,'evaluation_releases':self.eval_releases,
              'inline_enqueue':self.inline,'contract':'Private CPU copies of support memory, memory time, mailbox and mailbox time are captured before the next global batch can update target state.'},indent=2))
