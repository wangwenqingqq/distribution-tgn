"""Controlled CUDA-stream latency perturbation, calibrated and measured with CUDA events."""
import torch,json,statistics
class CudaDelay:
    def __init__(self,out,phase):
        self.out=out;self.phase=phase;self.records=[]
        samples=[];cycles=1000000
        for _ in range(6):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();torch.cuda._sleep(cycles);b.record();b.synchronize()
            samples.append(a.elapsed_time(b))
        self.calibration_ms=samples
        self.cycles_per_ms=cycles/statistics.median(samples[1:])
    def __call__(self,milliseconds):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();torch.cuda._sleep(max(1,round(milliseconds*self.cycles_per_ms)));b.record()
        self.records.append((self.phase['name'],milliseconds,a,b))
    def finish(self):
        torch.cuda.synchronize()
        result={'calibration_ms':self.calibration_ms,'cycles_per_ms':self.cycles_per_ms,
                'records':[{'phase':phase,'requested_ms':ms,'observed_gpu_ms':a.elapsed_time(b)} for phase,ms,a,b in self.records]}
        (self.out/'cuda_delay.json').write_text(json.dumps(result,indent=2))
