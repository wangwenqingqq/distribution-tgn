"""CPU-only checks for paired argv, resource exclusion, and work accounting."""
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import run_interleaved_parameter_ablation as runner

root = Path(__file__).resolve().parents[1]
for path in (root / 'src').glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
rows = json.loads(subprocess.check_output([
    sys.executable, str(root / 'src/run_interleaved_parameter_ablation.py'),
    '--stage', 'performance', '--campaign-id', 'cpu_check', '--dry-run'], text=True))
assert len(rows) == 6 and len({r['run_id'] for r in rows}) == 6
assert [r['arm'] for r in rows] == ['native', 'arena', 'arena', 'native', 'native', 'arena']
for i in range(0, 6, 2):
    normalized = []
    for row in rows[i:i+2]:
        argv = row['argv'].copy()
        for flag in ['--label', '--output', '--parameter-transport']:
            argv[argv.index(flag)+1] = '<TREATMENT_OR_ID>'
        normalized.append(argv)
    assert normalized[0] == normalized[1]
    assert rows[i]['seed'] == rows[i+1]['seed']
with tempfile.TemporaryDirectory(prefix='paired-runner-test-') as directory:
    run = Path(directory)
    (run/'run.json').write_text(json.dumps({'indices': [4, 5], 'status': 'passed',
                                         'child_process_wall_seconds': 20.0}))
    gpu = '4, GPU-FIXTURE4, 17, 0\n5, GPU-FIXTURE5, 17, 0\n'
    for phase in ['before', 'after']:
        (run/(phase+'_gpus.txt')).write_text(gpu)
        (run/(phase+'_apps.txt')).write_text('')
    assert runner.resource_acceptance(run)['accepted']
    (run/'after_apps.txt').write_text('GPU-FIXTURE4, 1, fixture, 100\n')
    assert not runner.resource_acceptance(run)['accepted']
    (run/'after_apps.txt').write_text('')
    for rank in range(2):
        target = run/'results'/f'rank{rank}'
        target.mkdir(parents=True)
        summary = {'parameters_finite': True, 'process_start_perf_s': 100.0,
                   'epoch_intervals': [{'start_s': 10+rank, 'wall_seconds': 3.0,
                                        'positive_edges': 452586}],
                   'training_work': [{'phase': 'warm_up', 'positive_edges': 452586}],
                   'validation': [{'ap': 0.7}]}
        (target/'summary.json').write_text(json.dumps(summary))
    actual = runner.summarize_run(run, 1)
    assert actual['global_epoch_seconds'] == [4.0]
    assert actual['formal_train_seconds'] == 4.0
    summary['epoch_intervals'][0]['positive_edges'] -= 1
    (target/'summary.json').write_text(json.dumps(summary))
    try:
        runner.summarize_run(run, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incomplete work was accepted')
print('CPU runner checks passed: fresh labels, single treatment, after-run exclusion, global clock and event counts.')
