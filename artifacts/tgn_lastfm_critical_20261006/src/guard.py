import argparse,csv,datetime,fcntl,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from contextlib import ExitStack
p=argparse.ArgumentParser()
p.add_argument('--indices',required=True);p.add_argument('--label',required=True)
p.add_argument('--timeout',type=int,default=60);p.add_argument('--env',action='append',default=[])
p.add_argument('command',nargs=argparse.REMAINDER)
args=p.parse_args();root=Path(__file__).resolve().parents[1]
cmd=args.command[1:] if args.command and args.command[0]=='--' else args.command
if not cmd: p.error('Command required')
indices=[int(i) for i in args.indices.split(',')]
out=root/'runs'/args.label;out.mkdir(exist_ok=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
record={'start_utc':now(),'indices':indices,'command':cmd}
def save():(out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
def snapshot(label):
    outputs=[]
    queries=[['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
             ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']]
    for k,query in enumerate(queries):
        r=subprocess.run(query,capture_output=True,text=True,check=True)
        (out/(label+('_gpus.txt' if k==0 else '_apps.txt'))).write_text(r.stdout)
        outputs.append(r.stdout)
    return {int(r[0]):[x.strip() for x in r] for r in csv.reader(outputs[0].splitlines())},outputs[1]
base=root.parent
locks=sorted(p for d in [base,base/'.locks'] for p in d.iterdir() if p.is_file() and p.name.endswith('.lock') and any(re.search(r'gpu'+str(i)+r'(?:[_.]|$)',p.name) for i in indices))
record['locks']=[str(p) for p in locks];save()
with ExitStack() as stack:
    try:
        for path in locks:
            fp=stack.enter_context(path.open('r+'));fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows,apps=snapshot('before')
        for i in indices:
            if i not in rows:raise RuntimeError('GPU missing '+str(i))
            _,uuid,memory,util=rows[i]
            if int(memory)>64 or int(util)!=0 or uuid in apps:raise RuntimeError('GPU not idle '+str(i))
        uuids=[rows[i][1] for i in indices]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids),PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for item in args.env:
            k,v=item.split('=',1);env[k]=v
        record['uuids']=uuids;record['extra_environment']=args.env;save()
        with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
            process_start=time.perf_counter()
            proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            try:rc=proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                rc=-999
        record['child_process_wall_seconds']=time.perf_counter()-process_start
        record['exit_code']=rc;record['status']='passed' if rc==0 else 'failed'
        snapshot('after')
    except (BlockingIOError,RuntimeError) as e:
        record['status']='skipped';record['reason']=str(e)
record['end_utc']=now();save();print(json.dumps(record),flush=True)
sys.exit(0 if record['status']=='passed' else 75 if record['status']=='skipped' else 1)
