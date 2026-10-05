"""Replay real LastFM sampler inputs independently of distributed communication."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/PipeTGL'), str(ROOT / 'vendor/GNNFlow'),
               str(ROOT / 'vendor/dgl/python'), str(ROOT / 'pydeps')]
import numpy as np
import torch
from config import _lastfm_default_config
from gnnflow.utils import load_dataset, build_dynamic_graph
from gnnflow.temporal_sampler import TemporalSampler

parser = argparse.ArgumentParser()
parser.add_argument('--dependency-file', required=True)
parser.add_argument('--global-batch', type=int, default=16)
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--breakdown', action='store_true')
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
torch.cuda.set_device(0)
train, _, _, full = load_dataset('LASTFM', str(ROOT / 'data/pipe'))
record = next(x for x in json.loads(Path(args.dependency_file).read_text())
              if x['epoch'] == 0 and x['global_batch'] == args.global_batch)
nodes = np.asarray(record['input_nodes'], dtype=np.int64)
lo = args.global_batch * 600
rows = train.iloc[lo:lo + 600]
assert len(nodes) == 3 * len(rows)
assert np.array_equal(nodes[:len(rows)], rows['src'].to_numpy())
assert np.array_equal(nodes[len(rows):2*len(rows)], rows['dst'].to_numpy())
times = np.tile(rows['time'].to_numpy(dtype=np.float32), 3)
config = copy.deepcopy(_lastfm_default_config)
config['mem_resource_type'] = 'shared'
graph = build_dynamic_graph(**config, device=0)
graph.add_edges(full['src'].to_numpy(dtype=np.int64), full['dst'].to_numpy(dtype=np.int64),
                full['time'].to_numpy(dtype=np.float32), full['eid'].to_numpy(dtype=np.int64),
                add_reverse=False)
sampler = TemporalSampler(graph, [10], sample_strategy='recent', num_snapshots=1,
                          snapshot_time_window=0, prop_time=False, seed=4101)
sampler.sample(nodes, times)
torch.cuda.synchronize()
wall = []
hashes = []
arrays = None
parts = []
for repeat in range(args.repeats):
    torch.cuda.nvtx.range_push('sampler_replay')
    start = time.perf_counter_ns()
    if args.breakdown:
        raw = sampler._sampler.sample(nodes, times)
        native_done = time.perf_counter_ns()
        b = sampler._to_dgl_block(raw)[0][0]
        dgl_done = time.perf_counter_ns()
    else:
        b = sampler.sample(nodes, times)[0][0]
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    wall.append((end - start) / 1e6)
    if args.breakdown:
        parts.append({'native_sampler_host_ms': (native_done-start)/1e6,
                      'to_dgl_host_ms': (dgl_done-native_done)/1e6,
                      'final_sync_ms': (end-dgl_done)/1e6})
    torch.cuda.nvtx.range_pop()
    a, c = b.edges()
    arrays = {'src_ids': b.srcdata['ID'].numpy(), 'src_ts': b.srcdata['ts'].numpy(),
              'edge_src': a.numpy(), 'edge_dst': c.numpy(),
              'edge_ids': b.edata['ID'].numpy(), 'edge_dt': b.edata['dt'].numpy()}
    digest = hashlib.sha256()
    for k in sorted(arrays):
        digest.update(k.encode())
        digest.update(arrays[k].tobytes())
    hashes.append(digest.hexdigest())
assert len(set(hashes)) == 1, 'Recent sampler output changed across identical replays'
np.savez(out / 'sample.npz', **arrays)
report = {'global_batch': args.global_batch, 'input_nodes': len(nodes),
          'sampled_edges': len(arrays['edge_ids']), 'repeats': args.repeats,
          'sampler_call_ms': wall, 'output_sha256': hashes[0],
          'component_host_ms': parts, 'breakdown_enabled': args.breakdown,
          'graph_policy': config, 'scope': 'Sampler wrapper wall time; NCU runs are diagnostic only.'}
(out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
