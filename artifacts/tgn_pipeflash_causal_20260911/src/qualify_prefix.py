import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tasks=[
['--tag','prefix_candidate_frozen_v1','--worlds','8','--variants','native,flash','--schedules','state_first','--epochs','2','--entry','pipe_prefix_candidate_bench.py','--freeze-weights','--audit-state'],
['--tag','prefix_probe_base','--worlds','8','--variants','native,flash','--schedules','baseline','--epochs','3','--entry','pipe_prefix_bench.py','--audit-state','--probe'],
['--tag','prefix_probe_candidate','--worlds','8','--variants','native,flash','--schedules','state_first','--epochs','3','--entry','pipe_prefix_candidate_bench.py','--audit-state','--probe']
]
for args in tasks:
 print('TASK',args,flush=True);subprocess.run([sys.executable,str(ROOT/'src/run_causal.py')]+args,check=True)
