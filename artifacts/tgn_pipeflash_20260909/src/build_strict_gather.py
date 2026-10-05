from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0')
os.environ.setdefault('MAX_JOBS','2')
from torch.utils.cpp_extension import load
load(name='_pipeflash_gather_strict',sources=[str(ROOT/'src/csrc/fused_gather_strict.cu')],
    extra_cuda_cflags=['-O3','--fmad=false','-std=c++17'],
    extra_cflags=['-O3','-std=c++17'],build_directory=str(ROOT/'cache/strict_gather'),verbose=True)
