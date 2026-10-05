"""FlashTGN gather/attention kernels adapted to PipeTGL's existing layer/state contract.

No memory update or optimizer order is changed. Descriptor construction and transfer
are explicit and must be included in end-to-end timing.
"""
from __future__ import annotations
from dataclasses import dataclass
import types
import torch
import torch.nn.functional as F

@dataclass
class Descriptor:
    nbr: torch.Tensor
    eid: torch.Tensor
    neg_delta: torch.Tensor
    query_nodes: torch.Tensor
    zero_query_times: torch.Tensor
    valid_flat: torch.Tensor
    seg_ptr: torch.Tensor
    max_k: int
    num_edges: int
    def to(self,device):
        return Descriptor(**{k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v)
                             for k,v in vars(self).items()})

def prepare_descriptor(block,fanout=None):
    """Prepare from a CPU sampled DGL block, before memory is read."""
    if block.device.type!='cpu':raise ValueError('Build descriptors from CPU topology before transfer')
    q=block.num_dst_nodes();e=block.num_edges()
    import numpy as np
    src,dst=block.edges()
    src_np=src.numpy().astype(np.int64,copy=False)
    dst_np=dst.numpy().astype(np.int64,copy=False)
    counts=np.bincount(dst_np,minlength=q)
    k=int(counts.max()) if q else 0
    if fanout is not None and k>fanout:raise ValueError('Fanout exceeded')
    if e and not np.array_equal(src_np,np.arange(q,q+e)):
        raise ValueError('PipeTGL/GNNFlow layer requires one source slot per sampled edge')
    width=max(k,1)
    nbr=np.full((q,width),-1,dtype=np.int32)
    eid=np.full((q,width),-1,dtype=np.int32)
    neg_delta=np.zeros((q,width),dtype=np.float32)
    ptr=np.concatenate(([0],np.cumsum(counts)))
    order=np.argsort(dst_np,kind='stable')
    rows=dst_np[order]
    slots=np.arange(e)-np.repeat(ptr[:-1],counts)
    nbr[rows,slots]=src_np[order];eid[rows,slots]=order
    if e:neg_delta[rows,slots]=-block.edata['dt'].numpy()[order]
    return Descriptor(torch.from_numpy(nbr),torch.from_numpy(eid),torch.from_numpy(neg_delta),
                      torch.arange(q,dtype=torch.int32),torch.zeros(q),
                      torch.from_numpy((nbr>=0).reshape(-1)),
                      torch.from_numpy(ptr.astype(np.int32)),k,e)

def torch_packed_forward(layer,block,desc):
    """Independent differentiable adapter reference; useful before CUDA qualification."""
    q=block.num_dst_nodes();h=block.srcdata['h']
    if desc.num_edges==0:return torch.zeros(q,layer.dim_out,device=h.device)
    dst_h=h[:q]
    qin=torch.cat([dst_h,layer.time_enc(torch.zeros(q,device=h.device))],1)
    eids=desc.eid.reshape(-1)[desc.valid_flat].long()
    nodes=desc.nbr.reshape(-1)[desc.valid_flat].long()
    dt=-desc.neg_delta.reshape(-1)[desc.valid_flat]
    zin=torch.cat([h[nodes],block.edata['f'][eids],layer.time_enc(dt)],1)
    qp=layer.w_q(qin).reshape(q,layer.num_head,-1)
    kp=layer.w_k(zin).reshape(-1,layer.num_head,layer.dim_out//layer.num_head)
    vp=layer.w_v(zin).reshape_as(kp)
    outputs=[]
    for i in range(q):
        begin,end=int(desc.seg_ptr[i]),int(desc.seg_ptr[i+1])
        if begin==end:outputs.append(torch.zeros(layer.dim_out,device=h.device))
        else:
            score=layer.att_act((qp[i]*kp[begin:end]).sum(-1))
            att=layer.att_dropout(score.softmax(0))
            outputs.append((att[...,None]*vp[begin:end]).sum(0).reshape(-1))
    out=torch.stack(outputs)
    return layer.layer_norm(F.relu(layer.dropout(layer.w_out(torch.cat([out,dst_h],1)))))

def flash_forward(layer,block,desc=None):
    from flash_tgn.autograd_ops import FusedGatherEncode,FusedSegmentedAttention
    desc=desc if desc is not None else getattr(block,'_pipeflash_descriptor',None)
    if desc is None:raise RuntimeError('Missing CPU-prepared PipeFlash descriptor')
    q=block.num_dst_nodes();h=block.srcdata['h']
    if desc.num_edges==0:return torch.zeros(q,layer.dim_out,device=h.device)
    if layer.training and layer.att_dropout.p!=0:
        raise ValueError('Flash segmented attention does not implement attention dropout; qualify with p=0')
    if not layer.use_time_enc or not layer.use_edge_feat:
        raise ValueError('Initial adapter requires temporal encoding and edge features')
    if h.device.type!='cuda':raise ValueError('Flash kernel adapter requires CUDA')
    # h already contains the memory and optional projected node features of PipeTGL.
    # Zero query times plus negated precomputed deltas preserve PipeTGL's delta values.
    zeros=torch.zeros_like(h,requires_grad=False)
    z,qin=FusedGatherEncode.apply(zeros,h,block.edata['f'].contiguous(),
        desc.nbr,desc.eid,desc.neg_delta,desc.query_nodes,desc.zero_query_times,
        layer.time_enc.w.weight,layer.time_enc.w.bias)
    qp=layer.w_q(qin)
    zpacked=z[desc.valid_flat]
    # Keep the original parameters and optimizer state; this concatenation is timed.
    kv=torch.cat([layer.w_k(zpacked),layer.w_v(zpacked)],1)
    out=FusedSegmentedAttention.apply(qp,kv,desc.seg_ptr,layer.num_head,desc.max_k)
    result=layer.w_out(torch.cat([out,h[:q]],1))
    return layer.layer_norm(F.relu(layer.dropout(result)))

def install_flash_kernels(model):
    count=0
    for layer in model.layers.values():
        layer.forward=types.MethodType(flash_forward,layer);count+=1
    return count

def transfer_mfgs_with_descriptors(mfgs,device,fanout=None):
    for layers in mfgs:
        for i,block in enumerate(layers):
            desc=prepare_descriptor(block,fanout)
            transferred=block.to(device)
            transferred._pipeflash_descriptor=desc.to(device)
            layers[i]=transferred
    return mfgs
