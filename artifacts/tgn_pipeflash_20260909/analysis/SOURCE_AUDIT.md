# Source audit before baseline measurements

These observations apply to the archived source revisions and have not yet been demonstrated as runtime failures on this machine.

## Native entrypoint and counting
- PipeTGL run_single_mechine.sh invokes scripts/pipeTrain3.py.
- The entrypoint requires LOCAL_RANK/LOCAL_WORLD_SIZE and initializes distributed training even for world size one. Its shell launcher uses plain Python for one GPU, so an explicit launch adapter is required.
- warm_up() traverses its loader and calls backward/optimizer.step; model/optimizer state is not restored afterwards. Complete wall timing must include this training pass and report its processed edge count.
- Both warm_up() and train() fetch the *next* batch before processing the current one, and break on StopIteration. The final rank-local batch is therefore skipped in those loops. Count actual processed positive edges rather than infer work solely from the dataset length.
- train() starts a FetchClient thread whose q_input.get() has no active corresponding q_input.put() in the current entrypoint (the producers are commented out). Qualify process termination and record any cleanup adaptation.
- The native one-GPU ring paths contain self-send/receive assumptions and a cached/uncached memory branch that does not initialize receive buffers for world size one. A separate qualified single-GPU path may be necessary.

## Combining model implementations
- PipeTGL's target-only persistent memory update and support-node temporary update differ from this FlashTGN snapshot, which persists updates for all_compute_nodes (targets, negatives, sampled neighbors).
- Native FlashTGN's attention module declares dropout but the examined one-layer forward does not apply it. Initial numerical comparisons therefore use dropout zero; the adapter preserves PipeTGL's output dropout and explicitly rejects training-time attention dropout.
- PipeTGL returns all-zero embeddings when the entire sampled block has no edges. This must be preserved by the adapter.
- Force no random feature fallback: provide real Wikipedia edge features and explicit zero node features for FlashTGN.
- FlashTGN native train() builds schedules outside its reported epoch elapsed time in gpu/cpu precompute modes. The new driver reports complete wall time including this work independently of the native loop time.
- FlashTGN CPU prefetch loop overwrites next_item with batch i+2 before assigning the next current batch. Validate batch sequence before using CPU mode; initial runs use online mode.

## Initial adapter boundary
src/pipe_flash_adapter.py reuses FlashTGN fused gather/time-encoding and packed segmented attention. It preserves the original PipeTGL parameters, support/target memory handling, dropout-zero numerical contract, and optimizer state. CPU topology conversion and tensor transfer must be included in the timing. No claim of end-to-end equivalence is made until the qualification tests pass.

## Runtime verification and final adaptations (2026-09-09)

The observations above are static source findings; the following have now been exercised:
- Native single-GPU launch adapter passed; new GNNFlow Python compatibility replaces the removed torch._six string type tuple.
- Native two-GPU phase barrier stalled at warm-up completion, with rank 0 in NCCL barrier and rank 1 in final parameter-to-CPU writeback. Waiting at the warm-up exit alone did not resolve it. Gloo phase barriers with CUDA completion before the control barrier made the two- and four-GPU runs finish. Exact failed and successful runs retain thread stacks and logs. This establishes behavior on this stack, not the behavior of the paper's original environment.
- Complete-batch mode processes the final local batch, adjusts its prefetch/I/O count, and computes positive target overlap using actual len(eid). For Wikipedia all 110,232 positive training events are counted in every epoch and warm-up across 1/2/4 GPUs.
- Original Flash fast math failed the initial large-delta test (max output error about 0.03828). Precise trig with separate multiply/add passed initial time parameters but failed learned parameters. A GPU probe found the original linear time encoder matches fused multiply-add at tested row counts 16, 1280 and 18000; separate rounding differs by up to 0.25 before cosine.
- The final separate gather extension uses precise trig and explicit FMA. Initial and learned-parameter output/input-gradient/edge-gradient/all-parameter-gradient cases pass unchanged atol=5e-5, rtol=4e-4. Maximum learned-case output error is about 1.31e-6. See adapter_initial_fma_v4 and adapter_learned_fma_v2.
- Native FlashTGN is only a short execution smoke test here. Its memory-update set and native splitting differ from PipeTGL. No direct system-level equivalence or cross-system speedup is claimed.
- Original per-rank Adam, parameter staleness, extra warm-up optimization and sqrt(world-size) learning-rate scaling are retained. Three-epoch validation AP is an observation, not a convergence or time-to-target result.

## Duplicate mailbox events

Final memory and mailbox timestamps differed even between same-seed single-GPU native and fused runs (e.g. 484/599 node timestamp entries for seed 2026). They were all finite, so this was not NaN comparison. All three memory update methods select a duplicate node's position with inv.new_empty(...).scatter_(0, inv, perm), which does not define the final occurrence when indices repeat on CUDA. The opt-in --deterministic-mailbox mode replaces these six selectors with an integer amax scatter reduction of event position. Both implementations then use an explicit last-event policy. The single-GPU full-pass smoke tests have exactly equal final memory and mailbox timestamps. Original selectors and runs are retained; this is an explicit behavior contract for the controlled baseline, not a claim about results in the original paper.

Full CUDA Nsight tracing of the original-scatter fused two-GPU run stalled in sampling/parameter send versus state receive; it was stopped as a diagnostic failure. It does not yield a valid bubble or throughput measurement. Non-profiled multi-GPU runs completed. Profiling limitations must remain visible in interpretation.
