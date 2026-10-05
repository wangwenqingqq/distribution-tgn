"""Describe sampled state dependencies; bounds are not an executable schedule."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/nsys_lastfm_p2_native_threads_s4101_e1_v2/results'
batches = sorted([b for r in range(2) for b in json.loads(
    (RUN / f'rank{r}/dependency_sets.json').read_text())], key=lambda b: b['global_batch'])
assert [b['global_batch'] for b in batches] == list(range(1509))
last_write = {}
depths = []
dependencies = []
support_records = []
adjacent = []
slot_dup = []
for i, b in enumerate(batches):
    targets, reads = set(b['targets']), set(b['read_nodes'])
    # Coarse RAW/WAW edges at the state frontier, not a forward/backward barrier.
    pred = {last_write[n] for n in reads | targets if n in last_write}
    dependencies.append(sorted(pred))
    depths.append(1 + max((depths[p] for p in pred), default=0))
    support_records.append({(n, last_write.get(n, -1)) for n in reads - targets})
    slot_dup.append(1 - len(reads) / b['node_slots'])
    if i:
        prev = set(batches[i-1]['targets'])
        adjacent.append({'batch': i, 'target_overlap': len(targets & prev),
                         'read_from_previous_targets': len(reads & prev),
                         'fraction_reads_from_previous_targets': len(reads & prev) / len(reads),
                         'support_from_previous_targets': len((reads - targets) & prev)})
    for n in targets:
        last_write[n] = i

def stats(x):
    a = np.asarray(x, dtype=float)
    return {'mean': float(a.mean()), 'p10': float(np.quantile(a, .1)),
            'median': float(np.median(a)), 'p90': float(np.quantile(a, .9))}

chunks = []
for k in [1, 2, 4, 8, 16]:
    requested = sum(map(len, support_records))
    version_unique = sum(len(set.union(*support_records[i:i+k]))
                         for i in range(0, len(batches), k))
    node_unique = sum(len({n for rec in support_records[i:i+k] for n, _ in rec})
                      for i in range(0, len(batches), k))
    chunks.append({'chunk_batches': k, 'support_record_requests': requested,
                   'unique_node_versions': version_unique,
                   'optimistic_version_reuse_fraction': 1 - version_unique / requested,
                   'incorrect_node_only_reuse_fraction': 1 - node_unique / requested})

report = {
    'batches': len(batches), 'sampled_edges': stats([b['sampled_edges'] for b in batches]),
    'target_unique': stats([len(b['targets']) for b in batches]),
    'sampled_state_unique': stats([len(b['read_nodes']) for b in batches]),
    'duplicate_sampled_node_slot_fraction': stats(slot_dup),
    'duplicate_scope': 'PipeTGL prepare_input already uniquifies node IDs before memory_updater, then expands via searchsorted. Duplicate sampled slots do not imply redundant GRU updates or an 8.7x remaining optimization opportunity.',
    'adjacent': {'pairs': len(adjacent),
                 'raw_state_edge_fraction': sum(bool(x['read_from_previous_targets']) for x in adjacent) / len(adjacent),
                 'waw_target_edge_fraction': sum(bool(x['target_overlap']) for x in adjacent) / len(adjacent),
                 'read_coverage': stats([x['fraction_reads_from_previous_targets'] for x in adjacent])},
    'state_batch_proxy_dag': {'longest_path_batches': max(depths),
                             'predecessors': stats(list(map(len, dependencies))),
                             'scope': 'Sampled node RAW/WAW proxy. State version ownership, mail updates, and intra-batch ordering require runtime validation. Parameter receive occurs after forward/backward and before optimizer.step; this DAG does not imply whole-batch serial training.'},
    'chunk_support_version_bounds': chunks,
    'reuse_scope': 'Counts unique (support node, latest target-writing batch), excluding current targets. This is an optimistic bound for memory-record preparation, not reuse of learned activations or complete sampling descriptors. Static feature payload is zero in LastFM.',
}
(ROOT / 'analysis/dependency_breakdown.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
