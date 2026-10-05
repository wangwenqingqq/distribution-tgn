"""One contiguous parameter allocation, preserving Parameter identities and Adam state layout."""
import torch

def bind(model):
    params=list(model.parameters())
    assert params and all(p.is_contiguous() and p.dtype==params[0].dtype and p.device==params[0].device for p in params)
    with torch.no_grad():
        flat=torch.cat([p.detach().reshape(-1) for p in params])
        offset=0
        for p in params:
            p.data=flat[offset:offset+p.numel()].view_as(p)
            offset+=p.numel()
    model.register_buffer('_parameter_arena',flat,persistent=False)
    model._parameter_arena_layout=[{'name':name,'shape':list(p.shape),'numel':p.numel()} for name,p in model.named_parameters()]
