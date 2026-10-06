"""Fresh paired E1 runs with independent resource and semantic acceptance gates."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def command(python, indices, label, transport, seed, epochs, qualification=False):
    args = [python, str(ROOT / 'src/guard.py'), '--indices', indices,
            '--label', label, '--timeout', '900']
    for item in ['DGLDEFAULTDIR=' + str(ROOT / 'cache/dgl'),
                 'TMPDIR=' + str(ROOT / 'cache/tmp'), 'NCCL_P2P_DISABLE=1',
                 'NCCL_IB_DISABLE=1', 'NCCL_SOCKET_IFNAME=lo']:
        args += ['--env', item]
    args += ['--', python, '-m', 'torch.distributed.run', '--standalone',
             '--nproc_per_node', '2', str(ROOT / 'src/pipe_paired_bench.py'),
             '--output', str(ROOT / 'runs' / label / 'results'),
             '--dataset', 'LASTFM', '--epochs', str(epochs), '--seed', str(seed),
             '--variant', 'native', '--ablation', 'baseline',
             '--parameter-transport', transport, '--complete-batches',
             '--deterministic-mailbox', '--control-barrier', 'gloo']
    if qualification:
        args += ['--capture-optimizer-rng', '--audit-state']
    return args


def resource_acceptance(run):
    record = read(run / 'run.json')
    reasons = []
    if record.get('status') != 'passed':
        reasons.append('child_not_passed')
    for phase in ['before', 'after']:
        gpu_path = run / (phase + '_gpus.txt')
        apps_path = run / (phase + '_apps.txt')
        if not gpu_path.is_file() or not apps_path.is_file():
            reasons.append(phase + '_record_missing')
            continue
        rows = {int(x[0]): [v.strip() for v in x]
                for x in csv.reader(gpu_path.read_text().splitlines())}
        apps = apps_path.read_text()
        for index in record['indices']:
            if index not in rows:
                reasons.append(phase + '_gpu_missing')
                continue
            _, uuid, memory, utilization = rows[index]
            if int(memory) > 64 or uuid in apps:
                reasons.append(phase + '_foreign_activity')
            if phase == 'before' and int(utilization) != 0:
                reasons.append('before_not_idle')
    return {'accepted': not reasons, 'reasons': sorted(set(reasons)),
            'scope': 'Before/after observations; internal transient competition is not ruled out.'}


def compare_tree(left, right, path='$'):
    import torch
    if isinstance(left, torch.Tensor):
        return [] if isinstance(right, torch.Tensor) and torch.equal(left, right) else [path]
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return [path + '.keys']
        return [item for key in left for item in compare_tree(left[key], right[key], path + '.' + str(key))]
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return [path + '.shape']
        return [item for i, (a, b) in enumerate(zip(left, right))
                for item in compare_tree(a, b, path + '[' + str(i) + ']')]
    return [] if left == right else [path]


def summarize_run(run, epochs):
    ranks = [read(run / 'results' / ('rank' + str(rank)) / 'summary.json')
             for rank in range(2)]
    assert all(len(r['epoch_intervals']) == epochs and r['parameters_finite'] for r in ranks)
    assert sum(w['positive_edges'] for r in ranks for w in r['training_work']
               if w['phase'] == 'warm_up') == 905172
    walls = []
    for epoch in range(epochs):
        entries = [r['epoch_intervals'][epoch] for r in ranks]
        assert sum(x['positive_edges'] for x in entries) == 905172
        starts = [r['process_start_perf_s'] + x['start_s'] for r, x in zip(ranks, entries)]
        ends = [start + x['wall_seconds'] for start, x in zip(starts, entries)]
        walls.append(max(ends) - min(starts))
    return {'run_id': run.name, 'global_epoch_seconds': walls,
            'formal_train_seconds': sum(walls),
            'full_process_seconds': read(run / 'run.json')['child_process_wall_seconds'],
            'validation_ap_by_rank': [[v['ap'] for v in r['validation']] for r in ranks],
            'resources': resource_acceptance(run)}


def semantic_pair(native, arena, qualification=False):
    import torch
    fields = ['initial_model.pt', 'final_model.pt', 'final_memory.pt']
    if qualification:
        fields.append('final_optimizer_rng.pt')
    rows = []
    for rank in range(2):
        checks = {}
        for filename in fields:
            a = torch.load(native / 'results' / ('rank' + str(rank)) / filename,
                           map_location='cpu', weights_only=True)
            b = torch.load(arena / 'results' / ('rank' + str(rank)) / filename,
                           map_location='cpu', weights_only=True)
            differences = compare_tree(a, b)
            checks[filename] = {'bitwise_equal': not differences, 'first_differences': differences[:10]}
        ap_a = [v['ap'] for v in read(native / 'results' / ('rank' + str(rank)) / 'summary.json')['validation']]
        ap_b = [v['ap'] for v in read(arena / 'results' / ('rank' + str(rank)) / 'summary.json')['validation']]
        checks['ap_curve'] = {'bitwise_equal': ap_a == ap_b}
        if qualification:
            for name, path in [('native', native), ('arena', arena)]:
                audit = read(path / 'results' / ('rank' + str(rank)) / 'prefix_audit.json')
                future = sum(x['future_memory_timestamps'] + x['future_mailbox_timestamps'] for x in audit)
                checks[name + '_timestamp_audit'] = {'bitwise_equal': future == 0, 'future_reads': future}
        rows.append({'rank': rank, 'checks': checks})
    return {'passed': all(c['bitwise_equal'] for r in rows for c in r['checks'].values()),
            'ranks': rows, 'optimizer_rng_checked': qualification}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['qualification', 'performance'], required=True)
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--session-id', default='s1')
    parser.add_argument('--indices', default='4,5')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.campaign_id)
    assert re.fullmatch('[A-Za-z0-9_-]+', args.session_id)
    indices = [int(x) for x in args.indices.split(',')]
    assert len(indices) == 2 and len(set(indices)) == 2
    planned = [(4101, ['native', 'arena']), (4102, ['arena', 'native']), (4103, ['native', 'arena'])]
    if args.stage == 'qualification':
        planned = [(4101, ['native', 'arena'])]
    if args.dry_run:
        rows = []
        for seed, order in planned:
            for arm in order:
                label = f'{args.campaign_id}_{args.session_id}_{args.stage}_s{seed}_{arm}_a1'
                rows.append({'seed': seed, 'arm': arm, 'run_id': label,
                             'argv': command(args.python, args.indices, label,
                                             'list' if arm == 'native' else 'arena', seed,
                                             1 if args.stage == 'qualification' else 3,
                                             args.stage == 'qualification')})
        print(json.dumps(rows, indent=2))
        return
    import torch  # Validate the intended environment before launching children.
    (ROOT / 'runs').mkdir(exist_ok=True)
    (ROOT / 'plans').mkdir(exist_ok=True)
    lock_file = (ROOT / 'analysis/campaign.lock').open('a+')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = ROOT.parent
    for index in indices:
        matches = [p for directory in [parent, parent / '.locks'] for p in directory.iterdir()
                   if p.is_file() and p.name.endswith('.lock')
                   and re.search('gpu' + str(index) + r'(?:[_.]|$)', p.name)]
        assert matches, 'Missing cooperative locks for physical GPU ' + str(index)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'src').glob('*.py'))}
    identity = {'campaign_id': args.campaign_id, 'session_id': args.session_id,
                'source_commit': 'f9998a2a1e6645af050c826fe950cbdd97884cbe',
                'indices': indices, 'source_hashes': source_hashes,
                'seeds_and_order': [[seed, order] for seed, order in [(4101, ['native', 'arena']), (4102, ['arena', 'native']), (4103, ['native', 'arena'])]],
                'max_pair_attempts': 2, 'main_metric': 'full_process_seconds',
                'secondary_metric': 'formal_train_seconds', 'prototype_budget_reduction': 0.03,
                'second_session': 'Not automatically scheduled; requires a separately frozen confirmation plan.',
                'inference_unit': 'seed/session paired block; ranks and epochs are not independent samples'}
    contract_path = ROOT / 'plans' / (args.campaign_id + '_' + args.session_id + '_contract.json')
    if contract_path.exists():
        assert read(contract_path) == identity, 'Campaign contract changed'
    else:
        save(contract_path, identity)
    status_path = ROOT / 'analysis' / (args.stage + '_status.json')
    qualification_path = ROOT / 'analysis/paired_qualification.json'
    if args.stage == 'performance':
        qualification = read(qualification_path)
        assert qualification['semantic']['passed'] and qualification['source_hashes'] == source_hashes
    completed = []
    launched_this_invocation = 0
    for seed, order in planned:
        accepted = False
        for attempt in range(1, 3):
            pair_path = ROOT / 'analysis' / f'{args.campaign_id}_{args.session_id}_{args.stage}_s{seed}_a{attempt}.json'
            if pair_path.exists():
                previous = read(pair_path)
                if previous.get('valid_for_timing'):
                    completed.append(previous)
                    accepted = True
                    print('READ_EXISTING_PAIR', seed, 'not a new measurement', flush=True)
                    break
                if previous.get('failure_reason') or previous.get('semantic') is not None:
                    raise RuntimeError('Retained failure needs diagnosis before retry: ' + pair_path.name)
                assert previous.get('exclusion_reason') in ['resource_skipped', 'resource_observation_failed']
                continue
            pair = {'seed': seed, 'order': order, 'attempt': attempt, 'campaign_id': args.campaign_id,
                    'session_id': args.session_id, 'runs': {}, 'valid_for_timing': False,
                    'semantic': None, 'source_hashes': source_hashes}
            failed_resource = False
            for arm in order:
                label = f'{args.campaign_id}_{args.session_id}_{args.stage}_s{seed}_{arm}_a{attempt}'
                run = ROOT / 'runs' / label
                if run.exists():
                    raise RuntimeError('Unpaired/incomplete run retained; explicitly diagnose it before retry: ' + label)
                save(status_path, {'state': 'running', 'run_id': label, 'completed_pairs': len(completed)})
                print('START', label, flush=True)
                rc = subprocess.run(command(args.python, args.indices, label,
                                            'list' if arm == 'native' else 'arena', seed,
                                            1 if args.stage == 'qualification' else 3,
                                            args.stage == 'qualification')).returncode
                launched_this_invocation += 1
                if rc == 75:
                    pair['exclusion_reason'] = 'resource_skipped'
                    failed_resource = True
                    break
                if rc:
                    pair['failure_reason'] = 'child_failed'
                    save(pair_path, pair)
                    save(status_path, {'state': 'failed', 'run_id': label})
                    raise SystemExit(rc)
                row = summarize_run(run, 1 if args.stage == 'qualification' else 3)
                pair['runs'][arm] = row
                if not row['resources']['accepted']:
                    pair['exclusion_reason'] = 'resource_observation_failed'
                    failed_resource = True
                    break
            if failed_resource:
                save(pair_path, pair)
                save(status_path, {'state': 'waiting_for_idle_gpus', 'excluded_pair': pair_path.name})
                print('EXCLUDED_PAIR', seed, attempt, flush=True)
                raise SystemExit(75)
            paths = {arm: ROOT / 'runs' / row['run_id'] for arm, row in pair['runs'].items()}
            pair['semantic'] = semantic_pair(paths['native'], paths['arena'], args.stage == 'qualification')
            pair['valid_for_timing'] = pair['semantic']['passed']
            save(pair_path, pair)
            if not pair['semantic']['passed']:
                save(status_path, {'state': 'semantic_failure', 'pair': pair_path.name})
                raise SystemExit(2)
            for metric in ['formal_train_seconds', 'full_process_seconds']:
                pair[metric + '_speedup'] = pair['runs']['native'][metric] / pair['runs']['arena'][metric]
            save(pair_path, pair)
            completed.append(pair)
            accepted = True
            print('PAIR_PASS', seed, {m: pair[m + '_speedup'] for m in ['formal_train_seconds', 'full_process_seconds']}, flush=True)
            if args.stage == 'qualification':
                save(qualification_path, pair)
            break
        if not accepted:
            save(status_path, {'state': 'attempt_budget_exhausted', 'seed': seed})
            raise SystemExit(75)
    report = {'campaign_id': args.campaign_id, 'session_id': args.session_id, 'pairs': completed,
              'paired_processes_total': 2 * len(completed),
              'guard_launches_this_invocation': launched_this_invocation,
              'scope': 'Fresh paired fixed-work screening; no Flash, no time-to-target claim, no eight-GPU scaling.'}
    for metric in ['formal_train_seconds', 'full_process_seconds']:
        ratios = [pair[metric + '_speedup'] for pair in completed]
        report[metric] = {'paired_speedups': ratios,
                          'geometric_speedup': math.exp(sum(math.log(x) for x in ratios) / len(ratios))}
    save(ROOT / 'analysis' / (args.stage + '_paired_results.json'), report)
    save(status_path, {'state': 'complete', 'complete_pairs': len(completed)})
    print('COMPLETE', args.stage, flush=True)


if __name__ == '__main__':
    main()
