import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for tag,entry,variants,schedule in [('prefix_v2_probe_base','pipe_prefix_bench.py','native,flash','baseline'),('prefix_v2_probe_candidate','pipe_prefix_candidate_bench.py','flash','state_first')]:
 subprocess.run([sys.executable,str(ROOT/'src/run_causal.py'),'--tag',tag,'--worlds','8','--variants',variants,'--schedules',schedule,'--epochs','3','--entry',entry,'--audit-state','--probe'],check=True)
