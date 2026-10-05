import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tasks=[
['--tag','probe_arena_p4','--worlds','4','--variants','native,flash','--schedules','baseline','--epochs','3','--probe','--entry','pipe_arena_bench.py'],
['--tag','target_arena_p4','--worlds','4','--variants','native,flash','--schedules','baseline','--seeds','2026,2027,2028','--epochs','20','--target-ap','0.97','--entry','pipe_arena_bench.py'],
['--tag','sensitivity_zero','--worlds','4','--variants','native,flash','--schedules','baseline','--epochs','5'],
]
for ms in ['1','4']:
 for stage in ['state','compute']:
  tasks.append(['--tag','sensitivity_'+stage+'_ms'+ms,'--worlds','4','--variants','native,flash','--schedules','baseline','--epochs','5','--delay-ms',ms,'--delay-stage',stage])
(ROOT/'analysis/phase2_tasks.json').write_text(json.dumps(tasks,indent=2))
for args in tasks:
 print('TASK',args,flush=True)
 subprocess.run([sys.executable,str(ROOT/'src/run_causal.py')]+args,check=True)
