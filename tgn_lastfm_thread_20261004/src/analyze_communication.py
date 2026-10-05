"""Retain per-direction completion samples; do not call host elapsed wire time."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
labels = {'same_numa': 'comm_lastfm_same_numa_v1',
          'cross_numa': 'comm_lastfm_cross_numa_v1',
          'same_numa_default': 'comm_lastfm_same_numa_default_v1'}
rows = []
for topology,label in labels.items():
    run = ROOT/'runs'/label
    assert json.loads((run/'run.json').read_text())['status'] == 'passed'
    data = [json.loads((run/'results'/f'rank{rank}.json').read_text()) for rank in range(2)]
    for pair in zip(data[0]['cases'],data[1]['cases']):
        a,b = pair
        assert a['name'] == b['name'] and a['bytes'] == b['bytes']
        # Both ranks measured the same exchange. Keep max endpoint completion,
        # rather than pretending the two endpoints are independent repetitions.
        samples = np.maximum(a['samples_ms'],b['samples_ms'])
        medians = [float(np.median(samples[parity::2])) for parity in range(2)]
        rows.append({'topology':topology,'case':a['name'],'bytes':a['bytes'],
                     'tensor_count':a['tensor_count'],'median_completion_ms':float(np.median(samples)),
                     'p90_completion_ms':float(np.quantile(samples,.9)),
                     'alternating_direction_medians_ms':medians,
                     'paired_exchange_samples_ms':samples.tolist()})
report = {'rows':rows,'scope':'Twenty alternating exchanges per case, max host completion across the two endpoints. These include API, peer arrival, and CUDA completion; this is not pure wire latency or training speedup. NUMA is a single-pair screening measurement; default NCCL passed payload correctness only.'}
(ROOT/'analysis/communication_breakdown.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows:
    print(row['topology'],row['case'],row['bytes'],round(row['median_completion_ms'],4),row['alternating_direction_medians_ms'])
