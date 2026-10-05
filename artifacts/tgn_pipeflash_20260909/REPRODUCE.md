# Reproduce the PipeTGL + Flash-kernel prototype

This directory is on **pro6000-8**, under /home/data/wangxuran. The experiment modifies only its own source snapshot, adapters, data layouts and build outputs. Shared environments, drivers and other experiments are unchanged.

## Environment

- Python: /home/data/wangxuran/isaacsim6/env/bin/python
- PyTorch: 2.11.0+cu130; CUDA compiler: /usr/local/cuda-13.1/bin/nvcc
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition; architecture 12.0
- DGL 1.1.x and GNNFlow were built from pinned source for Python 3.12/CUDA 13.
- Additional Python dependencies are isolated in pydeps/.
- Compiler commands, source commits and compatibility patches are in analysis/. Build failures and successful builds remain in logs/.
- FlashTGN is the archived local artifact identified by file hashes in analysis/source_manifest.json, not a newly authenticated upstream release.

## Numerical contract

The adapter keeps PipeTGL's model parameters, target/support memory semantics, optimizer and parameter-staleness protocol. It replaces gather/time encoding and attention with adapted fused kernels. The gather uses precise trigonometric functions and an explicit fused multiply-add matching the tested original FP32 linear time encoder. Attention dropout is supported only at zero in this prototype. The tested configuration is one layer, fanout 10, dimension/time dimension 100, two heads and FP32 highest precision.

Original Flash fast-math kernels failed a large-time-difference test. Disabling fast math while separately rounding multiplication and addition also failed with learned bias. These versions and their timing records are retained as diagnostic evidence, not qualified equivalent speedups. Fixed-input output/gradient agreement does not imply identical complete training trajectories.

## Rebuild the adapted gather

Run on pro6000-8, from this directory:

```bash
CUDA_VISIBLE_DEVICES='' CUDA_HOME=/usr/local/cuda-13.1 TORCH_CUDA_ARCH_LIST=12.0 PYTHONPATH="$PWD/pydeps" MAX_JOBS=2 /home/data/wangxuran/isaacsim6/env/bin/python src/build_strict_gather.py
```

The build uses src/csrc/fused_gather_strict.cu and writes cache/strict_gather/. It does not modify the original FlashTGN extension.

For GNNFlow and DGL, apply the archived compatibility patches to the pinned sources before using the archived CMake configurations. The successful incremental build targets are vendor/GNNFlow/build and vendor/dgl/build (target dgl). The command text files also record earlier configuration attempts; successful build logs are gnnflow_build_v5.log and dgl_build_v4.log. No GPU probing is needed for these explicit architecture builds.

## Numerical checks

Use a fresh label and an actually idle GPU. The guard checks processes, memory, utilization and existing advisory locks. The following example selects GPU 6 and refuses to run if it is busy.

```bash
/home/data/wangxuran/isaacsim6/env/bin/python src/guard.py --indices 6 --label reproduce_adapter --timeout 90 --env DGLDEFAULTDIR="$PWD/cache/dgl" -- /home/data/wangxuran/isaacsim6/env/bin/python src/qualify_adapter.py --device cuda --output "$PWD/runs/reproduce_adapter/results"
```

The learned-weight check is src/qualify_learned_adapter.py. It loads the archived trained checkpoints named in that script. Both initial-parameter and learned-parameter output/gradient checks must pass before updating analysis/qualified_implementation.json. run_campaign.py rejects missing/failed qualification or changed qualified source/binary hashes.

## Training runs

```bash
/home/data/wangxuran/isaacsim6/env/bin/python src/run_campaign.py --tag reproduce --gpu-pool 1,3,4,5 --world-sizes 1,2,4 --seeds 2026,2027,2028 --epochs 3 --deterministic-mailbox
```

This runs sequentially and checks every rank's work counts. Each measured epoch processes 110,232 positive training edges, including the final 432-edge batch, and the native warm-up performs one additional optimization pass. The sampler's native next-batch exit, partial-batch overlap indexing and I/O process length are adapted together by --complete-batches. The executed source and overlap function are saved per rank. The controlled comparison also uses --deterministic-mailbox in both variants: repeated node positions select the maximum event index, avoiding ambiguous CUDA scatter writes. This is an explicit last-event policy, and the original selector remains available without that flag. Native dropout-zero model, per-rank Adam, parameter staleness, and sqrt(world-size) learning-rate scaling remain.

--control-barrier gloo uses a CPU control group for phase barriers. Memory and parameter transfers still use NCCL. This was necessary to make the archived multi-GPU program terminate on this software stack. The failed native barrier and other diagnostic attempts remain archived. The experiments set NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1 and NCCL_SOCKET_IFNAME=lo because earlier transport qualification found direct-P2P corruption on this host. Scaling results do not represent a qualified GPUDirect/NVLink configuration.

## Timing and artifacts

- guard.py records complete child-process wall time: launcher, imports, preparation, warm-up, training, validation, checkpoint artifacts and exit.
- Per-rank epoch intervals include their batch loop's setup and final CUDA synchronization. Stable runs record the absolute common-host monotonic start for every rank. Aggregation reconstructs each global epoch from the earliest rank start to the latest synchronized completion, and also retains maximum-rank durations. Older runs without that clock field only report maximum-rank durations.
- The training-driver interval includes the additional warm-up optimization pass and validation.
- Per rank: initial/final weights, final memory/mailbox snapshot, actual positive-edge counts, phase/epoch intervals, numerical-finiteness result and executed entrypoint.
- Per run: exact command/environment, GPU snapshots, stdout/stderr and exit status.
- Aggregation: src/aggregate_results.py; JSON contains per-seed values and sample standard deviations.

Nsight traces are separate diagnostic runs. Use nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none before the guarded torchrun command, append --profile to pipe_bench.py, and set TMPDIR to this directory's cache/. Export with nsys export --type sqlite, then use src/trace_analysis.py. GPU-activity interval unions, dependency waiting and SM utilization are distinct metrics. CPU NVTX launch attribution does not automatically cover autograd worker threads; unattributed kernels remain explicitly unattributed.

The current measurements are a small Wikipedia prototype study. They do not establish two-layer behavior, other datasets, convergence, time to target, 8-GPU behavior, multi-host scaling or a publishable speedup over all current systems.

Final qualifying kernel checks: runs/adapter_initial_fma_v4 and runs/adapter_learned_fma_v2. Final controlled benchmark tag: stable. Earlier initial, qualified and final tags are diagnostic/superseded as explained by their status files. Full CUDA tracing of fused two-GPU execution stalled; do not use that failed trace as a performance result.

Stable host dispatch traces: src/run_host_profiles.py (no CUDA interception). Single-GPU full CUDA profiles: src/run_single_cuda_profiles.py. Both are diagnostic-only runs with deterministic last-event semantics. Shared-memory objects from finished runs were cleaned after confirming no owned experiment processes remained; serialized outputs are retained.
