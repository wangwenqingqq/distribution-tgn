import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for tag,entry,schedule in [('audit_base_p8','pipe_audit_bench.py','baseline'),('audit_candidate_p8','pipe_arena_inline_audit_bench.py','state_first')]:
 subprocess.run([sys.executable,str(ROOT/'src/run_causal.py'),'--tag',tag,'--worlds','8','--variants','flash','--schedules',schedule,'--epochs','3','--entry',entry],check=True)
