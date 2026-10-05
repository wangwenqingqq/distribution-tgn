import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tasks=[
['--tag','probe_inline_p8','--worlds','8','--variants','flash','--schedules','baseline,state_first','--epochs','3','--probe','--entry','pipe_inline_bench.py'],
['--tag','probe_arena_inline_p8','--worlds','8','--variants','flash','--schedules','baseline,state_first','--epochs','3','--probe','--entry','pipe_arena_inline_bench.py'],
['--tag','gpu_delay_zero_p8','--worlds','8','--variants','native,flash','--schedules','baseline','--epochs','5','--entry','pipe_gpu_delay_bench.py'],
]
for stage in ['state','compute']:
 tasks.append(['--tag','gpu_delay_'+stage+'_ms1_p8','--worlds','8','--variants','native,flash','--schedules','baseline','--epochs','5','--delay-ms','1','--delay-stage',stage,'--entry','pipe_gpu_delay_bench.py'])
tasks.append(['--tag','gpu_delay_arena_zero_p4','--worlds','4','--variants','flash','--schedules','baseline','--epochs','5','--entry','pipe_arena_gpu_delay_bench.py'])
for stage in ['state','compute']:
 tasks.append(['--tag','gpu_delay_arena_'+stage+'_ms1_p4','--worlds','4','--variants','flash','--schedules','baseline','--epochs','5','--delay-ms','1','--delay-stage',stage,'--entry','pipe_arena_gpu_delay_bench.py'])
(ROOT/'analysis/phase4_tasks.json').write_text(json.dumps(tasks,indent=2))
for args in tasks:
 print('TASK',args,flush=True)
 subprocess.run([sys.executable,str(ROOT/'src/run_causal.py')]+args,check=True)
