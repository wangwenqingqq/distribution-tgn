import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tasks=[
['--tag','probe_p8','--worlds','8','--variants','native,flash','--schedules','baseline','--epochs','3','--probe'],
['--tag','probe_arena_p8','--worlds','8','--variants','native,flash','--schedules','baseline,state_first','--epochs','3','--probe','--entry','pipe_arena_bench.py'],

['--tag','probe_arena_state_first_p4','--worlds','4','--variants','native,flash','--schedules','state_first','--epochs','3','--probe','--entry','pipe_arena_bench.py'],
['--tag','target_arena_state_first_p4','--worlds','4','--variants','native,flash','--schedules','state_first','--seeds','2026,2027,2028','--epochs','20','--target-ap','0.97','--entry','pipe_arena_bench.py'],
['--tag','sensitivity_arena_zero','--worlds','4','--variants','flash','--schedules','baseline,state_first','--epochs','5','--entry','pipe_arena_bench.py'],
]
for stage in ['state','compute']:
 tasks.append(['--tag','sensitivity_arena_'+stage+'_ms1','--worlds','4','--variants','flash','--schedules','baseline,state_first','--epochs','5','--delay-ms','1','--delay-stage',stage,'--entry','pipe_arena_bench.py'])
(ROOT/'analysis/phase3_tasks.json').write_text(json.dumps(tasks,indent=2))
for args in tasks:
 print('TASK',args,flush=True)
 subprocess.run([sys.executable,str(ROOT/'src/run_causal.py')]+args,check=True)
