import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tasks=[
['--tag','prefix_repeat_probe_base','--worlds','8','--variants','native,flash','--schedules','baseline','--epochs','3','--entry','pipe_prefix_bench.py','--audit-state','--probe','--seeds','2027,2028'],
['--tag','prefix_repeat_probe_candidate','--worlds','8','--variants','flash','--schedules','state_first','--epochs','3','--entry','pipe_prefix_candidate_bench.py','--audit-state','--probe','--seeds','2027,2028']
]
for stage in ['zero','state','compute']:
 tasks.append(['--tag','prefix_gpu_delay_'+stage,'--worlds','8','--variants','flash','--schedules','baseline','--epochs','5','--entry','pipe_prefix_gpu_delay_bench.py','--delay-ms','0' if stage=='zero' else '1','--delay-stage','state' if stage=='zero' else stage])
(ROOT/'analysis/final_mechanism_tasks.json').write_text(json.dumps(tasks,indent=2))
for args in tasks:
 print('TASK',args,flush=True);subprocess.run([sys.executable,str(ROOT/'src/run_causal.py')]+args,check=True)
