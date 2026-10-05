"""Alternating P2P transfers using shapes observed in actual LastFM training."""
from pathlib import Path
import argparse
import json
import os
import statistics
import time
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--payload-file', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
rank = int(os.environ['RANK'])
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
control = dist.new_group(backend='gloo')
rows = [x for x in json.loads(Path(args.payload_file).read_text()) if x['phase'] == 'train']
parameters = next(x for x in rows if x['label'] == 'parameter_send_api')
states = sorted([x for x in rows if x['label'] == 'state_send_api'], key=lambda x: x['bytes'])
cases = [(f'state_p{int(q*100)}', states[int(q*(len(states)-1))]['shapes']) for q in [.1, .5, .9]]
cases += [('parameter_list', parameters['shapes']),
          ('parameter_flat', [[sum(int(torch.tensor(s).prod()) for s in parameters['shapes'])]])]
report = []
for name, shapes in cases:
    send = [torch.full(s, i+1., device='cuda', dtype=torch.float32) for i, s in enumerate(shapes)]
    recv = [torch.empty_like(t) for t in send]
    torch.cuda.synchronize()
    dist.barrier(group=control)
    measured = []
    for i in range(25):
        sender = i % 2
        start = time.perf_counter_ns()
        reqs = [dist.isend(t, dst=1-rank) for t in send] if rank == sender else [dist.irecv(t, src=1-rank) for t in recv]
        for req in reqs:
            req.wait()
        torch.cuda.synchronize()
        if i >= 5:
            measured.append((time.perf_counter_ns()-start)/1e6)
    assert all(bool((t == i+1.).all()) for i, t in enumerate(recv))
    report.append({'name': name, 'shapes': shapes, 'tensor_count': len(send),
                   'bytes': sum(t.numel()*t.element_size() for t in send),
                   'median_ms': statistics.median(measured), 'samples_ms': measured})
    dist.barrier(group=control)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
(out / f'rank{rank}.json').write_text(json.dumps({'rank': rank, 'cases': report,
    'scope': 'Isolated alternating send/recv, CUDA completion synchronized; includes host/API overhead, not pure wire latency or training time.'}, indent=2)+'\n')
print(json.dumps({'rank': rank, 'cases': [{k:v for k,v in r.items() if k not in ['shapes','samples_ms']} for r in report]}), flush=True)
dist.destroy_process_group()
