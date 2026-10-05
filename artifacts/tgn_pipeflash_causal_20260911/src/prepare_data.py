"""Prepare identical real edge arrays/features in the two native data layouts."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
def main():
    source=Path('/home/data/wangxuran/tgn_pathweaver_8p_20260909/training/data')
    d=np.load(source/'events.npz');ef=np.load(source/'edge_features.npy',mmap_mode='r')
    f=ROOT/'data/flash/data/wiki';p=ROOT/'data/pipe/WIKI'
    f.mkdir(parents=True,exist_ok=True);p.mkdir(parents=True,exist_ok=True)
    n=int(max(d['u'].max(),d['v'].max()))+1
    src=d['u'].astype(np.int32);dst=d['v'].astype(np.int32);ts=d['ts'].astype(np.float32)
    for name,arr in [('src',src),('dst',dst),('ts',ts)]:np.save(f/(name+'.npy'),arr)
    feats=np.asarray(ef[1:]).copy()
    np.save(p/'edge_features.npy',feats);torch.save(torch.from_numpy(feats),f/'edge_features.pt')
    torch.save(torch.zeros(n,100),f/'node_features.pt')
    with (p/'edges.csv').open('w') as out:
        out.write('eid,src,dst,time,ext_roll\\n'.replace('\\n','\n'))
        for i,(u,v,t,s) in enumerate(zip(d['u'],d['v'],d['ts'],d['split'])):
            out.write(f'{i},{u},{v},{float(t)},{s}\n')
    owners=np.stack([src,dst],1).reshape(-1);nbr=np.stack([dst,src],1).reshape(-1)
    eids=np.repeat(np.arange(len(src),dtype=np.int32),2);times=np.repeat(ts,2)
    order=np.argsort(owners,kind='stable')
    ind=np.concatenate(([0],np.cumsum(np.bincount(owners,minlength=n)))).astype(np.int32)
    for name,arr in [('ind',ind),('nbr',nbr[order]),('eid',eids[order]),('ets',times[order])]:
        np.save(f/('edges.undirected_tcsr.'+name+'.npy'),arr)
    manifest={'source_manifest':json.loads((source/'manifest.json').read_text()),
        'nodes':n,'events':len(src),'edge_feature_dim':int(feats.shape[1]),
        'node_features':'Pipe absent; Flash zero D=100 with identity projection',
        'edge_ids':'zero-based, removed original padding feature row',
        'split_warning':'Native Flash rounds split endpoints to batch size; align before AP comparisons',
        'csr':'undirected, stable ascending timestamp and original event order within each node',
        'files':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()
                 for parent in [f,p] for x in parent.iterdir() if x.is_file()}}
    (ROOT/'analysis/data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(n,len(src),feats.shape)
if __name__=='__main__':main()
