# PipeTGL + FlashTGN: initial verification
Authorized after the user accepted the critical-path and resource-aware distributed TGN direction on 2026-09-09.

## Objective
Determine whether faster intra-batch kernels expose state-update, transfer, and prefetch bottlenecks in PipeTGL, and build a semantically qualified combined baseline before evaluating a new scheduler.

## Sources and controls
- PipeTGL official commit c79fbb6e0e39668dd0de541a2c2b0735140099dc.
- GNNFlow official commit ea525b3d9db13d85c0bde1727a7ab306d7125c71.
- FlashTGN existing local artifact snapshot, files hashed in analysis/source_manifest.json. Its public anonymous API returned 401. This snapshot has no git metadata.
- Existing experiment sources and results remain separate. All new writes stay under this directory.
- Shared server: fresh GPU snapshots and advisory locks before every GPU job. No other jobs are stopped. Use NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1 due to the corruption detected in earlier transport qualification; disclose this transport limitation in all scaling results.

## Qualification before comparison
1. Check matching model dimensions, heads, layer count, time encoder, attention normalization, padding/empty-neighbor behavior, dropout, negative sampling, and memory/mailbox write sets.
2. Freeze descriptors and weights for forward/backward comparisons. Treat parameter staleness separately from node-memory consistency.
3. Keep the original PipeTGL state and optimizer semantics when substituting kernels. A whole-model swap is not automatically equivalent.
4. Account for all processed edges and partial batches. Source compatibility fixes and behavior fixes must be separately recorded.
5. FlashTGN native performance and the combined baseline may have different memory semantics; label them separately until qualified.

## First measurements
- Wikipedia real edges with actual edge features, common fixed batch sizes 600 and 2000, fanout 10, initial one-layer D=100/time=100/heads=2, dropout 0 for numerical qualification.
- Then test two layers and additional datasets only after the first configuration runs correctly.
- Compare 1/2/4 GPUs when idle; 8 GPUs only when all eight are available.
- Record complete process wall time, setup and topology preparation, epoch wall time, steady-state training loop, final synchronization and cleanup.
- Diagnostic trace separately records CPU work, state reads/updates, transfer, attention, backward, parameter communication, optimizer, and mailbox commit.
- Use non-synchronizing NVTX/CUDA-event instrumentation for representative traces. Synchronized phase profiles are diagnostic only, never reported as normal throughput.
- Sampled GPU utilization, dependency waiting, kernel execution time, and achieved memory bandwidth are different quantities.

## Decision gates
- Gate A: source/environment and correct single-GPU kernels.
- Gate B: executable PipeTGL and qualified kernel adapter; complete edge coverage.
- Gate C: directly combined baseline on 1/2/4 GPUs with measured post-fusion stages.
- Gate D: implement only the optimization supported by Gate C: shorten state critical path, change pipeline width, or distribute work within one semantic batch.
- STC is optional and requires measured end-to-end improvement including compression and backward. Irregular graph sparsity alone is not sufficient.

## Interpretation
The Table 2 REDDIT calculation gives 27.1 waiting units and 25.09% of the ideal GPU cycle. It does not reproduce Figure 15's 7.14%; post-optimization stage measurements and matching normalization are needed.
A finite-event model and arithmetic examples are sensitivity analyses, not measurements of either framework.
