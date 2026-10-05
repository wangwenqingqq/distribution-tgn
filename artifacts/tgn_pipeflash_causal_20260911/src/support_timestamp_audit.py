"""Observe timestamps actually supplied to support-memory GRU, without another GPU synchronization."""
import functools,inspect,textwrap,json
def install(ns,TGNN,out,phase):
    samples=[]
    transfer=ns['mfgs_to_cuda']
    def with_batch_time(mfgs,*a,**kw):
        times=[(float(b.srcdata['ts'][:b.num_dst_nodes()].min()),float(b.srcdata['ts'][:b.num_dst_nodes()].max())) for group in mfgs for b in group]
        result=transfer(mfgs,*a,**kw)
        for b,(lo,hi) in zip([b for g in result for b in g],times):
            b._audit_batch_min_ts=lo;b._audit_batch_max_ts=hi
        return result
    ns['mfgs_to_cuda']=with_batch_time
    def observe(b,nodes,mem_ts,mail_ts):
        hi=b._audit_batch_max_ts
        fm=mem_ts>hi;fl=mail_ts>hi
        samples.append({'phase':phase['name'],'batch_min_ts':b._audit_batch_min_ts,'batch_max_ts':hi,
                        'support_nodes':len(nodes),'future_memory_timestamps':int(fm.sum()),
                        'future_mailbox_timestamps':int(fl.sum()),
                        'max_future_mailbox_seconds':float((mail_ts[fl]-hi).max()) if bool(fl.any()) else 0,
                        'examples':[{'node':int(n),'mailbox_ts':float(t)} for n,t in zip(nodes[fl][:5],mail_ts[fl][:5])]})
    original=TGNN.prepare_input
    source=textwrap.dedent(inspect.getsource(original))
    old='mem_ts = self.memory.node_memory_ts[pull_nodes].to(device)'
    assert source.count(old)==1
    source=source.replace(old,'audit_mem_ts = self.memory.node_memory_ts[pull_nodes]\n        mem_ts = audit_mem_ts.to(device)')
    old='mail_ts = self.memory.mailbox_ts[pull_nodes].to(device)'
    assert source.count(old)==1
    source=source.replace(old,'audit_mail_ts = self.memory.mailbox_ts[pull_nodes]\n        mail_ts = audit_mail_ts.to(device)\n        _observe_support(b,pull_nodes,audit_mem_ts,audit_mail_ts)')
    scope=dict(original.__globals__,_observe_support=observe)
    exec(compile(source,'<support_timestamp_audit>','exec'),scope);TGNN.prepare_input=scope['prepare_input']
    (out/'audited_prepare_input.py').write_text(source)
    def finish():(out/'support_timestamp_audit.json').write_text(json.dumps(samples,indent=2))
    return finish
