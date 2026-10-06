import sys
from pathlib import Path
import torch
from torch.autograd import Function
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cache/strict_gather'))

class FusedGatherEncode(Function):
    """Fused node/memory/edge gather + time encoding + concat."""

    @staticmethod
    def forward(ctx, nfeat, mem, efeat, hot_nbr, hot_eid, hot_ets,
                query_nodes, query_times, te_weight, te_bias):
        import _pipeflash_gather_strict as _C

        z_out, q_in = _C.fused_gather_encode_v2(
            nfeat, mem, efeat, hot_nbr, hot_eid, hot_ets,
            query_nodes, query_times,
            te_weight.detach().squeeze(), te_bias.detach())
        ctx.save_for_backward(
            hot_nbr, hot_eid, hot_ets, query_nodes, query_times,
            te_weight, te_bias)
        ctx.num_nodes = nfeat.size(0)
        ctx.num_edges = efeat.size(0)
        return z_out, q_in

    @staticmethod
    def backward(ctx, grad_z, grad_q):
        hot_nbr, hot_eid, hot_ets, query_nodes, query_times, \
            te_weight, te_bias = ctx.saved_tensors
        import _pipeflash_gather_strict as _C

        edge_grad_rows = ctx.num_edges if ctx.needs_input_grad[2] else 0
        grad_nfeat, grad_mem, grad_efeat, grad_te_w, grad_te_b = (
            _C.fused_gather_encode_backward_v2(
                grad_z.contiguous(), grad_q.contiguous(),
                hot_nbr, hot_eid, hot_ets, query_nodes, query_times,
                te_weight.detach().squeeze(), te_bias.detach(),
                ctx.num_nodes, edge_grad_rows)
        )
        return (
            grad_nfeat if ctx.needs_input_grad[0] else None,
            grad_mem,
            grad_efeat if ctx.needs_input_grad[2] else None,
            None, None, None, None, None,
            grad_te_w.unsqueeze(1) if ctx.needs_input_grad[8] else None,
            grad_te_b if ctx.needs_input_grad[9] else None,
        )

