import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for tag,entry,variants,schedule in [('prefix_native_frozen_v2','pipe_prefix_bench.py','native','baseline'),('prefix_candidate_frozen_v2','pipe_prefix_candidate_bench.py','native,flash','state_first')]:
 subprocess.run([sys.executable,str(ROOT/'src/run_causal.py'),'--tag',tag,'--worlds','8','--variants',variants,'--schedules',schedule,'--epochs','2','--entry',entry,'--freeze-weights','--audit-state'],check=True)
