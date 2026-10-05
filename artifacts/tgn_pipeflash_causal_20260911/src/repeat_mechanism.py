"""Repeat the mechanism tests with seeds distinct from the exploratory run."""
import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tasks=[
['--tag','repeat_probe_p8','--worlds','8','--variants','native,flash','--schedules','baseline','--epochs','3','--probe','--seeds','2027,2028'],
['--tag','repeat_probe_candidate_p8','--worlds','8','--variants','flash','--schedules','state_first','--epochs','3','--probe','--seeds','2027,2028','--entry','pipe_arena_inline_bench.py'],
]
for seed in [2027,2028]:
 stages=['zero','state','compute'] if seed%2==0 else ['compute','state','zero']
 for stage in stages:
  tasks.append(['--tag',f'repeat_gpu_delay_{stage}_p8_s{seed}','--worlds','8','--variants','flash','--schedules','baseline','--epochs','5','--entry','pipe_gpu_delay_bench.py','--seeds',str(seed),'--delay-ms','0' if stage=='zero' else '1','--delay-stage','state' if stage=='zero' else stage])
(ROOT/'analysis/repeat_mechanism_tasks.json').write_text(json.dumps(tasks,indent=2))
for args in tasks:
 print('TASK',args,flush=True);subprocess.run([sys.executable,str(ROOT/'src/run_causal.py')]+args,check=True)
