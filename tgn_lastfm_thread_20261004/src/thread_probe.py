"""Optional host-thread diagnostics and a logging-only loss synchronization ablation."""
from contextlib import contextmanager
from collections import defaultdict
import functools
import inspect
import json
import os
import re
import threading
import time
import textwrap


def adapt_source(source, args):
    if args.ablation == 'loss_deferred':
        assert source.count('total_loss += float(loss) * num_target_nodes') == 2
        source = source.replace('total_loss += float(loss) * num_target_nodes',
                                'deferred_losses.append((loss.detach(), num_target_nodes))')
        a, b = source.index('def train('), source.index('def send(')
        train = source[a:b]
        assert train.count('        total_loss = 0\n') == 1
        train = train.replace('        total_loss = 0\n', '        total_loss = 0\n        deferred_losses = []\n')
        marker = '        torch.cuda.synchronize()\n'
        assert train.count(marker) == 1
        train = train.replace(marker, marker +
                              '        _loss_values = torch.stack([x for x, _ in deferred_losses]).cpu().tolist()\n'
                              '        total_loss = sum(x * n for x, (_, n) in zip(_loss_values, deferred_losses))\n')
        source = source[:a] + train + source[b:]
        a = source.index('def warm_up(')
        warm = source[a:]
        assert warm.count('    total_loss = 0\n') == 1
        warm = warm.replace('    total_loss = 0\n', '    total_loss = 0\n    deferred_losses = []\n')
        source = source[:a] + warm
    if args.thread_probe:
        source, n = re.subn(r'(?m)^([ ]+)mfgs, eid = next_data$',
                           lambda m: m.group(0) + '\n' + m.group(1) + 'thread_batch(iteration_now)', source)
        assert n == 2, ('batch markers', n)
        patterns = [(r'fetchClient\.event2\.wait\(\)', 'io_wait'),
                    (r'fetchClient\.event1\.set\(\)', 'io_request'),
                    (r'(?:sends_thread[12]?|sampling_thread)\.join\(\)', 'thread_join'),
                    (r'total_loss \+= float\(loss\) \* num_target_nodes', 'loss_scalar_read'),
                    (r'params = \[param\.data\.clone\(\) for param in model\.parameters\(\)\]', 'parameter_clone')]
        for pattern, label in patterns:
            source = re.sub(r'(?m)^([ ]+)(' + pattern + ')$',
                            lambda m: m.group(1) + "with thread_span('" + label + "'):\n" +
                            m.group(1) + '    ' + m.group(2), source)
    return source


class ThreadProbe:
    def __init__(self, out, rank, world, phase, enabled):
        self.out, self.rank, self.world = out, rank, world
        self.phase, self.enabled = phase, enabled
        self.epoch = -1
        self.batch = -1
        self.rows = []
        self.samples = []
        self.payloads = []
        self.ordinal = defaultdict(int)
        self.local = threading.local()
        self.torch = None

    def epoch_begin(self, epoch):
        self.epoch = epoch

    def batch_begin(self, batch):
        self.batch = batch

    def metadata(self):
        return getattr(self.local, 'metadata', {'phase': self.phase['name'], 'epoch': self.epoch,
                                                'batch': self.batch})

    @contextmanager
    def span(self, label):
        if not self.enabled:
            yield
            return
        meta = dict(self.metadata())
        start = time.perf_counter_ns()
        if self.torch is not None:
            self.torch.cuda.nvtx.range_push(f"{label}|r={self.rank}|e={meta['epoch']}|b={meta['batch']}")
        try:
            yield
        finally:
            if self.torch is not None:
                self.torch.cuda.nvtx.range_pop()
            self.rows.append(dict(meta, label=label, start_ns=start, end_ns=time.perf_counter_ns(),
                                  rank=self.rank, pid=os.getpid(), native_tid=threading.get_native_id()))

    def install(self, ns, TGNN, Memory, protocol):
        import numpy as np
        import torch
        from gnnflow.temporal_sampler import TemporalSampler
        self.torch = torch
        ns['thread_batch'] = self.batch_begin
        owner = self
        def wrap(obj, name, label):
            original = getattr(obj, name) if not isinstance(obj, dict) else obj[name]
            @functools.wraps(original)
            def traced(*a, **kw):
                if label in ['parameter_send_api', 'state_send_api']:
                    tensors = a[0] if a else None
                    if isinstance(tensors, (list, tuple)):
                        owner.payloads.append(dict(owner.metadata(), rank=owner.rank,
                            label=label, native_tid=threading.get_native_id(),
                            bytes=sum(t.numel()*t.element_size() for t in tensors),
                            shapes=[list(t.shape) for t in tensors],
                            dtypes=[str(t.dtype) for t in tensors]))
                with owner.span(label):
                    return original(*a, **kw)
            if isinstance(obj, dict): obj[name] = traced
            else: setattr(obj, name, traced)
        original_sample = TemporalSampler.sample
        @functools.wraps(original_sample)
        def sample(sampler, nodes, times, *a, **kw):
            key = (owner.phase['name'], owner.epoch)
            ordinal = owner.ordinal[key]
            owner.ordinal[key] += 1
            with owner.span('sampling_full'):
                result = original_sample(sampler, nodes, times, *a, **kw)
            if key[0] == 'train':
                b = result[0][0]
                ids = b.srcdata['ID'].numpy()
                edge_src, edge_dst = b.edges()
                # IDs are recorded while the sampled block is already on CPU.
                targets = np.unique(np.asarray(nodes[:2 * (len(nodes) // 3)], dtype=np.int64))
                reads = np.unique(ids)
                global_batch = owner.rank + owner.world * ordinal
                row = {'rank': owner.rank, 'epoch': owner.epoch,
                                      'global_batch': global_batch,
                                      'input_nodes': np.asarray(nodes, dtype=np.int64).tolist(),
                                      'targets': targets.tolist(), 'read_nodes': reads.tolist(),
                                      'node_slots': len(ids), 'sampled_edges': b.num_edges(),
                                      'ts_max': float(np.max(times))}
                if global_batch < 32:
                    row.update(target_slot_ids=ids[:b.num_dst_nodes()].tolist(),
                               edge_src_slots=edge_src.numpy().tolist(),
                               edge_dst_slots=edge_dst.numpy().tolist(),
                               all_node_ids=ids.tolist())
                owner.samples.append(row)
            return result
        TemporalSampler.sample = sample
        for obj, name, label in [(TGNN, 'update_memory_and_send', 'target_state'),
                                 (TGNN, 'prepare_input', 'support_snapshot_and_compute'),
                                 (TGNN, 'forward', 'forward'), (Memory, 'recv_mem', 'state_receive'),
                                 (torch.Tensor, 'backward', 'backward_dispatch'),
                                 (torch.optim.Adam, 'step', 'optimizer'),
                                 (ns, 'send', 'parameter_send_api'), (ns, 'recv', 'parameter_recv_api')]:
            wrap(obj, name, label)
        wrap(protocol.mm, 'send', 'state_send_api')
        source = textwrap.dedent(inspect.getsource(type(protocol).receive_current_frontier))
        assert source.count('ready.synchronize()') == 1
        source = source.replace('ready.synchronize()', '_thread_ready_wait(ready)')
        def ready_wait(event):
            with owner.span('state_gpu_ready_cpu_wait'):
                event.synchronize()
        scope = dict(type(protocol).receive_current_frontier.__globals__, _thread_ready_wait=ready_wait)
        exec(compile(source, '<thread_probe_state_ready>', 'exec'), scope)
        protocol.receive_current_frontier = scope['receive_current_frontier'].__get__(protocol)

        original_start = threading.Thread.start
        def start(worker, *a, **kw):
            target = getattr(worker, '_target', None)
            arguments = getattr(worker, '_args', ())
            payload = arguments[0] if arguments else None
            tensor_payload = isinstance(payload, (list, tuple)) and payload and all(isinstance(x, torch.Tensor) for x in payload)
            control_target = payload is None and getattr(target, '__name__', '') in ['send', 'stage_send', 'profiled', 'traced']
            if not tensor_payload and not control_target:
                return original_start(worker, *a, **kw)
            count = len(payload) if isinstance(payload, (list, tuple)) else 0
            role = 'parameter' if count and len(arguments) == 3 else 'state' if count else 'control'
            meta = dict(owner.metadata())
            requested = time.perf_counter_ns()
            original_run = worker.run
            worker.name = f'pipe_{role}_r{owner.rank}'
            def run():
                owner.local.metadata = meta
                entered = time.perf_counter_ns()
                owner.rows.append(dict(meta, label='worker_first_instruction/' + role,
                                       rank=owner.rank, pid=os.getpid(), native_tid=threading.get_native_id(),
                                       start_ns=requested, end_ns=entered))
                with owner.span('worker_run/' + role):
                    return original_run()
            worker.run = run
            with owner.span('thread_start/' + role):
                return original_start(worker, *a, **kw)
        threading.Thread.start = start

    def finish(self):
        if not self.enabled:
            return
        (self.out / 'thread_intervals.json').write_text(json.dumps({
            'clock': 'host perf_counter_ns; shared on this machine',
            'scope': 'diagnostic wall intervals, not on-core CPU time', 'events': self.rows}) + '\n')
        (self.out / 'dependency_sets.json').write_text(json.dumps(self.samples) + '\n')
        (self.out / 'communication_payloads.json').write_text(json.dumps(self.payloads) + '\n')
