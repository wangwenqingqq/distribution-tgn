"""Verify lightweight evidence without starting training or loading CUDA."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--recompute',action='store_true')
args=parser.parse_args()
read=lambda p:json.loads(p.read_text())
manifest=read(ROOT/'ARCHIVE_MANIFEST.json')
for rel,info in manifest['files'].items():
    p=ROOT/rel
    assert p.is_file() and not p.is_symlink(),rel
    assert hashlib.sha256(p.read_bytes()).hexdigest()==info['archive_sha256'],rel
source=list((ROOT/'src').glob('*.py'))+[Path(__file__)]
for p in source:ast.parse(p.read_text(),filename=str(p))
saved=read(ROOT/'analysis/paired_short_training.json')
assert len(saved['rows'])==15
epochs=0
for row in saved['rows']:
    run=ROOT/'runs'/row['label']
    assert read(run/'run.json')['status']=='passed'
    parts=[read(run/'results'/f'rank{i}/summary.json') for i in range(2)]
    for e in range(3):
        assert sum(x['epoch_intervals'][e]['positive_edges'] for x in parts)==905172
        epochs+=1
    assert sum(x['positive_edges'] for s in parts for x in s['training_work'] if x['phase']=='warm_up')==905172
assert read(ROOT/'analysis/frozen_qualification.json')['passed']
assert read(ROOT/'analysis/trained_model_checks.json')['passed']
assert not read(ROOT/'analysis/excluded_parameter_trial.json')['admissible_for_performance']
ownership=read(ROOT/'analysis/gpu_ownership_audit.json')['rows']
names={x['label'] for x in saved['rows']}
assert all(x['admissible'] for x in ownership if x['label'] in names)
result={'hashed_file_versions':len(manifest['files']),'python_files_parsed':len(source),
        'formal_runs':15,'formal_rank_summaries':30,'complete_training_epochs':epochs,
        'saved_state_qualification_passed':True,'state_reverified_from_checkpoints':False,
        'gpu_training_or_profiling_started':False,'aggregate_recomputed':False}
if args.recompute:
    with tempfile.TemporaryDirectory(prefix='lastfm-archive-check-') as directory:
        tmp=Path(directory);(tmp/'src').mkdir();(tmp/'analysis').mkdir()
        for filename in ['run_campaign.py','analyze_campaign.py']:
            shutil.copyfile(ROOT/'src'/filename,tmp/'src'/filename)
        for row in saved['rows']:
            shutil.copytree(ROOT/'runs'/row['label'],tmp/'runs'/row['label'])
        subprocess.run([sys.executable,str(tmp/'src/analyze_campaign.py')],check=True,stdout=subprocess.PIPE,text=True)
        actual=read(tmp/'analysis/paired_short_training.json')
        def compare(a,b):
            if isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a:compare(a[k],b[k])
            elif isinstance(a,list):
                assert len(a)==len(b)
                for x,y in zip(a,b):compare(x,y)
            elif isinstance(a,float):assert math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12),(a,b)
            else:assert a==b,(a,b)
        compare(saved,actual)
        result['aggregate_recomputed']=True
        result['aggregate_comparison']='all values match within 1e-12; includes bootstrap screening intervals'
print(json.dumps(result,indent=2))
