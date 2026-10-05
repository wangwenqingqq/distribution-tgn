"""Check frozen weights, full work, state equivalence and timestamp audits."""
from pathlib import Path
import json
import torch
from run_campaign import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 1e-6, 1e-5
load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
base = ROOT / 'runs/qual_lastfm_p2_baseline_s4101_e1_v2/results/rank0'
initial = load(base / 'initial_model.pt')
reference = load(base / 'final_memory.pt')
rows = []
for name, _, _, _ in CONFIGS:
    label = f'qual_lastfm_p2_{name}_s4101_e1_v2'
    results = ROOT / 'runs' / label / 'results'
    row = {'name': name, 'label': label, 'ranks': []}
    for rank in range(2):
        path = results / f'rank{rank}'
        start, final = [load(path / (tag + '_model.pt')) for tag in ['initial', 'final']]
        same_initial = start.keys() == initial.keys() and all(torch.equal(start[k], initial[k]) for k in initial)
        unchanged = start.keys() == final.keys() and all(torch.equal(start[k], final[k]) for k in start)
        states = load(path / 'final_memory.pt')
        fields = {}
        for key in reference:
            a, b = reference[key], states[key]
            fields[key] = {'bitwise_equal': torch.equal(a, b),
                           'max_abs': float((a-b).abs().max()),
                           'finite': bool(torch.isfinite(b).all()),
                           'within_tolerance': bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL))}
            if key.endswith('_ts'):
                fields[key]['within_tolerance'] = fields[key]['bitwise_equal']
        audit = json.loads((path / 'prefix_audit.json').read_text())
        future = sum(x['future_mailbox_timestamps'] + x['future_memory_timestamps'] for x in audit)
        summary = json.loads((path / 'summary.json').read_text())
        row['ranks'].append({'rank': rank, 'same_initial': same_initial, 'weights_unchanged': unchanged,
                            'memory_fields': fields, 'audit_rows': len(audit), 'future_reads': future,
                            'training_events': summary['epoch_intervals'][0]['positive_edges']})
    row['passed'] = all(x['same_initial'] and x['weights_unchanged'] and x['future_reads'] == 0
                        and all(v['finite'] and v['within_tolerance'] for v in x['memory_fields'].values())
                        for x in row['ranks']) and sum(x['training_events'] for x in row['ranks']) == 905172
    rows.append(row)
report = {'passed': all(x['passed'] for x in rows), 'rows': rows,
          'comparison': {'float_atol': ATOL, 'float_rtol': RTOL, 'timestamps': 'bitwise exact'},
          'scope': 'Frozen final state and timestamp audit; trained trajectory identity and per-read producer-version equality are not asserted.'}
(ROOT / 'analysis/frozen_qualification.json').write_text(json.dumps(report, indent=2) + '\n')
for row in rows:
    print(row['name'], 'PASS' if row['passed'] else 'FAIL',
          'future_reads', sum(x['future_reads'] for x in row['ranks']),
          'max_abs', max(v['max_abs'] for x in row['ranks'] for v in x['memory_fields'].values()), flush=True)
assert report['passed'], 'Frozen LastFM qualification failed'
