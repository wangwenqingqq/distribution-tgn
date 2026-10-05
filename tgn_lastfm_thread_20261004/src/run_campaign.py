"""Guarded, resumable LastFM qualification and paired short-training campaign."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/data/wangxuran/isaacsim6/env/bin/python'
CONFIGS = [
    ('baseline', 'pipe_prefix_bench.py', 'native', 'baseline'),
    ('loss_deferred', 'pipe_prefix_bench.py', 'native', 'loss_deferred'),
    ('state_inline', 'pipe_prefix_bench.py', 'native', 'state_inline'),
    ('direct_flash', 'pipe_prefix_bench.py', 'flash', 'baseline'),
    ('combination', 'pipe_prefix_candidate_bench.py', 'flash', None),
]


def command(config, label, epochs, seed, indices, frozen=False):
    name, entry, variant, ablation = config
    ranks = len(indices.split(','))
    cmd = [PYTHON, str(ROOT / 'src/guard.py'), '--indices', indices,
           '--label', label, '--timeout', '900']
    for item in ['DGLDEFAULTDIR=' + str(ROOT / 'cache/dgl'),
                 'TMPDIR=' + str(ROOT / 'cache/tmp'), 'NCCL_P2P_DISABLE=1',
                 'NCCL_IB_DISABLE=1', 'NCCL_SOCKET_IFNAME=lo']:
        cmd += ['--env', item]
    cmd += ['--', PYTHON, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node', str(ranks), str(ROOT / 'src' / entry),
            '--output', str(ROOT / 'runs' / label / 'results'),
            '--dataset', 'LASTFM', '--epochs', str(epochs), '--seed', str(seed),
            '--variant', variant, '--complete-batches', '--deterministic-mailbox',
            '--control-barrier', 'gloo']
    if ablation:
        cmd += ['--ablation', ablation]
    else:
        cmd += ['--schedule', 'state_first']
    if frozen:
        cmd += ['--freeze-weights', '--audit-state']
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['qualification', 'performance'], required=True)
    parser.add_argument('--indices', default='4,5')
    args = parser.parse_args()
    lock = (ROOT / 'analysis/campaign.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = args.stage == 'qualification'
    if not frozen:
        qualification = json.loads((ROOT / 'analysis/frozen_qualification.json').read_text())
        if not qualification.get('passed'):
            raise RuntimeError('Qualification has not passed')
    seeds = [4101] if frozen else [4101, 4102, 4103]
    completed = []
    status_path = ROOT / 'analysis' / (args.stage + '_campaign.json')
    def save(state, current=None):
        status_path.write_text(json.dumps({'stage': args.stage, 'state': state,
            'current': current, 'completed': completed, 'updated_unix_s': time.time(),
            'indices': args.indices}, indent=2) + '\n')
    for i, seed in enumerate(seeds):
        order = CONFIGS if i == 0 else list(reversed(CONFIGS)) if i == 1 else CONFIGS[2:] + CONFIGS[:2]
        for config in order:
            prefix = 'qual' if frozen else 'short'
            epochs = 1 if frozen else 3
            version = 'v2' if frozen else 'v1'
            label = f'{prefix}_lastfm_p2_{config[0]}_s{seed}_e{epochs}_{version}'
            run = ROOT / 'runs' / label / 'run.json'
            save('running', label)
            if run.exists():
                previous = json.loads(run.read_text())
                if previous.get('status') != 'passed':
                    raise RuntimeError(f'Existing run is not complete and passed: {label}')
                print('REUSED', label, flush=True)
            else:
                print('START', label, flush=True)
                rc = subprocess.call(command(config, label, epochs, seed, args.indices, frozen))
                if rc:
                    save('stopped', label)
                    raise RuntimeError(f'Run exited {rc}: {label}')
            summaries = [json.loads(p.read_text()) for p in
                         (ROOT / 'runs' / label / 'results').glob('rank*/summary.json')]
            assert len(summaries) == 2, (label, len(summaries))
            assert all(s['parameters_finite'] for s in summaries), label
            for epoch in range(epochs):
                assert sum(s['epoch_intervals'][epoch]['positive_edges'] for s in summaries) == 905172, label
            assert sum(x['positive_edges'] for s in summaries for x in s['training_work']
                       if x['phase'] == 'warm_up') == 905172, label
            completed.append(label)
            save('running')
            print('PASS', label, flush=True)
    save('complete')


if __name__ == '__main__':
    main()
