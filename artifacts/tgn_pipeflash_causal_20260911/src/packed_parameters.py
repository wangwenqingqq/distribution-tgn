"""Coalesce a parameter snapshot into one NCCL message; preserve tensor bytes and optimizer objects."""
import torch
import torch.distributed as dist

def install(ns):
    old_send,old_recv=ns['send'],ns['recv']
    in_flight=[]
    def send(tensors,target,group=None):
        if tensors is None:return old_send(tensors,target,group)
        if not tensors:raise ValueError('Empty parameter list')
        assert all(t.dtype==tensors[0].dtype and t.device==tensors[0].device for t in tensors)
        with torch.cuda.device(tensors[0].device):
            flat=torch.cat([t.reshape(-1) for t in tensors])
            work=dist.batch_isend_irecv([dist.P2POp(dist.isend,flat,target,group)])[0]
            # Explicit lifetime: retain pending buffers through asynchronous completion.
            in_flight[:]=[(b,w) for b,w in in_flight if not w.is_completed()]
            in_flight.append((flat,work))
    def recv(tensors,target,group=None):
        if tensors is None:return old_recv(tensors,target,group)
        with torch.cuda.device(tensors[0].device):
            flat=torch.empty(sum(t.numel() for t in tensors),dtype=tensors[0].dtype,device=tensors[0].device)
            work=dist.batch_isend_irecv([dist.P2POp(dist.irecv,flat,target,group)])[0];work.wait()
            offset=0
            for tensor in tensors:
                tensor.copy_(flat[offset:offset+tensor.numel()].view_as(tensor));offset+=tensor.numel()
    ns['send'],ns['recv']=send,recv
    ns['packed_parameter_in_flight']=in_flight
