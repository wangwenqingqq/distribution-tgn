"""Summarize Nsight GPU activity and CPU launch attribution without adding overlapping intervals."""
import argparse,collections,json,sqlite3
from pathlib import Path
def union_ns(intervals):
    total=0;end=-1
    for a,b in sorted(intervals):
        if b>end:total+=b-max(a,end);end=b
    return total
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sqlite',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();c=sqlite3.connect(args.sqlite)
    strings=dict(c.execute('select id,value from StringIds'))
    nvtx=[(a,b,t or strings.get(tid,''),gtid) for a,b,t,tid,gtid in c.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null')]
    phases=[x for x in nvtx if x[2].startswith('epoch:train:')]
    if not phases:
        warm=[x for x in nvtx if x[2]=='phase:warm_up'];ev=[x for x in nvtx if x[2]=='phase:evaluate']
        if len(warm)==len(ev)==1:phases=[(warm[0][1],ev[0][0],'inferred_train_after_warmup_before_eval',warm[0][3])]
    ranges=collections.defaultdict(list)
    stage_labels={'target_state_update_and_send','support_state_read_and_update','attention_and_predict',
        'state_receive_and_host_gather','state_send_launch','backward','optimizer_step','temporal_sampling',
        'parameter_receive_or_order_signal','parameter_send_or_order_signal','sampled_topology_and_descriptor_transfer'}
    for a,b,label,tid in nvtx:
        if label in stage_labels:ranges[tid].append((a,b,label))
    for values in ranges.values():values.sort()
    runtime=collections.defaultdict(list)
    for a,b,tid,corr in c.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        runtime[tid].append((a,b,corr))
    attribution={}
    for tid,rows in runtime.items():
        intervals=ranges[tid];pointer=0;active=[]
        for a,b,corr in sorted(rows):
            while pointer<len(intervals) and intervals[pointer][0]<=a:
                active.append(intervals[pointer]);pointer+=1
            active=[v for v in active if v[1]>=b]
            label=min(active,key=lambda v:v[1]-v[0])[2] if active else 'unattributed'
            attribution[(tid&~((1<<24)-1),corr)]=label
    gpu=[]
    for a,b,dev,pid,corr,name in c.execute('select start,end,deviceId,globalPid,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
        name=strings.get(name,'');kind='nccl_kernel' if 'nccl' in name.lower() else 'compute_kernel'
        gpu.append((a,b,dev,pid,kind,attribution.get((pid,corr),'unattributed'),name))
    for table,kind in [('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
        for a,b,dev,pid,corr in c.execute('select start,end,deviceId,globalPid,correlationId from '+table):
            gpu.append((a,b,dev,pid,kind,attribution.get((pid,corr),'unattributed'),kind))
    result=[]
    for start,end,label,tid in phases:
        pid=tid&~((1<<24)-1);entries=[(max(a,start),min(b,end),d,k,s,n) for a,b,d,p,k,s,n in gpu if p==pid and a<end and b>start]
        by_kind=collections.defaultdict(list);by_stage=collections.defaultdict(list);kernels=collections.Counter()
        for a,b,dev,kind,stage,name in entries:
            by_kind[kind].append((a,b));by_stage[(stage,kind)].append((a,b))
            if kind=='compute_kernel':kernels[name]+=b-a
        active=union_ns([(a,b) for a,b,*_ in entries])
        result.append({'phase':label,'global_pid':pid,'wall_seconds':(end-start)/1e9,
            'gpu_any_activity_union_seconds':active/1e9,
            'no_traced_gpu_activity_fraction':1-active/(end-start),
            'activity_kind_union_seconds':{k:union_ns(v)/1e9 for k,v in by_kind.items()},
            'stage_activity_union_seconds':{stage+':'+kind:union_ns(v)/1e9 for (stage,kind),v in by_stage.items()},
            'top_compute_kernels_summed_seconds':[(n,t/1e9) for n,t in kernels.most_common(12)]})
    envelopes=[]
    for label in sorted({v[2] for v in phases}):
        selected=[v for v in phases if v[2]==label]
        a=min(v[0] for v in selected);b=max(v[1] for v in selected)
        envelopes.append({'phase':label,'global_start_ns':a,'global_end_ns':b,'global_epoch_envelope_seconds':(b-a)/1e9,
                          'rank_count':len(selected),'rank_start_spread_seconds':(max(v[0] for v in selected)-a)/1e9})
    output={'trace':args.sqlite,'phases':result,'global_epoch_envelopes':envelopes,'notes':[
        'Diagnostic profiled execution; not a throughput measurement.',
        'GPU activity is the interval union of traced kernels, copies and memset; no-activity time is not dependency waiting or SM-idle fraction.',
        'Launch attribution uses enclosing NVTX CPU ranges and CUDA correlation IDs. Stage unions can overlap and must not be added.',
        'Legacy trace phase is inferred from warm-up end to evaluation start; new traces use explicit epoch ranges.']}
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
