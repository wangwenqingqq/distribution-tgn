"""Diagnostic endpoints and storage-write provenance; never a timing benchmark.

CUDA timestamps are mapped through a synchronized per-device anchor with an
explicit host bracket. Work.wait return is not used as a GPU completion time.
Only selected windows retain immutable payloads / copy received parameters.
"""
import hashlib
import inspect
import json
import threading
import time
import types
from collections import defaultdict

WINDOWS = [(16, 47), (128, 159), (784, 815), (1472, 1503)]


class EndpointProbe:
    def __init__(self, out, rank, phase):
        self.out, self.rank, self.phase = out, rank, phase
        self.epoch = -1
        self.batch = -1
        self.rows = []
        self.seq = defaultdict(int)
        self.groups = {}
        self.payloads = {}
        self.cuda_points = []
        self.retained = []
        self.anchor = None
        self.anchor_bracket = None
        self.theta = 'rank%d:initial' % rank
        self.step_number = 0
        self.forward_theta = None
        self.received_theta = None
        self.lock = threading.Lock()

    def context(self):
        return {'phase': self.phase['name'], 'epoch': self.epoch,
                'processing_batch': self.batch}

    def selected(self, batch=None):
        b = self.batch if batch is None else batch
        return self.phase['name'] == 'train' and any(a-2 <= b <= z+2 for a, z in WINDOWS)

    def emit(self, kind, context=None, **fields):
        with self.lock:
            row = {'event_id': 'r%d:%d' % (self.rank, len(self.rows)),
                   'rank': self.rank, 'kind': kind, 'host_ns': time.perf_counter_ns(),
                   **(context or self.context()), **fields}
            self.rows.append(row)
        return row

    def point(self, row, field):
        import torch
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.rank))
        self.cuda_points.append((row, field, event))

    def begin_epoch(self, epoch):
        self.epoch = epoch

    def begin_batch(self, batch):
        import torch
        self.batch = batch
        if self.phase['name'] == 'train' and self.anchor is None:
            self.anchor = torch.cuda.Event(enable_timing=True)
            before = time.perf_counter_ns()
            self.anchor.record(torch.cuda.current_stream(self.rank))
            self.anchor.synchronize()
            after = time.perf_counter_ns()
            self.anchor_bracket = [before, after]
        if self.selected():
            self.emit('batch_begin', theta_at_batch_entry=self.theta)

    def parameter_snapshot(self, tensors):
        if self.selected():
            row = self.emit('parameter_payload', payload_batch=self.batch,
                            producer_version=self.theta,
                            tensor_count=len(tensors),
                            payload_bytes=sum(t.numel()*t.element_size() for t in tensors))
            self.point(row, 'payload_ready_gpu_ns')
            self.payloads[id(tensors[0])] = row
            self.retained.append((row, [t.detach() for t in tensors]))
        return tensors

    def target_read(self, nodes, cpu_values):
        if self.selected():
            self.emit('target_cpu_read_begin', nodes=nodes.tolist(),
                      fields=['node_memory_ts', 'mailbox_ts'] +
                      (['node_memory', 'mailbox'] if cpu_values else []),
                      value_source='shared_cpu' if cpu_values else 'GPU frontier mixture',
                      theta_at_target_update=self.theta)

    def cpu_write(self, nodes, push, send, edge):
        if self.phase['name'] == 'train':
            self.emit('cpu_write_' + edge, value_nodes=(nodes if send is None else push).tolist(),
                      timestamp_nodes=nodes.tolist(),
                      producer_version='train:e%d:b%d:r%d' % (self.epoch, self.batch, self.rank))

    def support_begin(self, nodes):
        if self.selected():
            self.support_row = self.emit('support_cpu_read_begin', nodes=nodes.tolist(),
                                         fields=['node_memory', 'mailbox', 'node_memory_ts', 'mailbox_ts'])

    def support_ready(self, model, b, nodes, mem, mail, mem_ts, mail_ts, release):
        import torch
        if self.selected():
            fields = dict(node_memory=mem, mailbox=mail,
                          node_memory_ts=mem_ts, mailbox_ts=mail_ts)
            equal = {name: bool(torch.equal(value, getattr(model.memory, name)[nodes]))
                     for name, value in fields.items()}
            self.emit('support_cpu_snapshot', parent_event_id=self.support_row['event_id'],
                      nodes=nodes.tolist(), same_as_current_shared_storage=equal,
                      private_cpu_copies=True)
        return release(model, b, nodes, mem_ts, mail_ts)

    def cached_state_ready(self, memory, idx, mem, mail):
        if self.selected():
            self.last_state_recv['nodes'] = idx.tolist()
            self.emit('state_cached_consumer_ready', nodes=idx.tolist(),
                      parent_event_id=self.last_state_recv['event_id'],
                      consumer='target_state_update', readiness='existing CUDA event synchronized')

    def uncached_read(self, memory, nodes):
        if self.selected():
            self.emit('target_value_cpu_read_begin', nodes=nodes.tolist(),
                      fields=['node_memory', 'mailbox'])

    def install_update(self, TGNN):
        # The qualification adapter emits the exact deterministic source. Read
        # that source rather than relying on inspect for a synthetic filename.
        method = TGNN.update_memory_and_send
        source = (self.out / 'update_memory_and_send_last_event.py').read_text()
        marker = '    if mem is None:'
        assert source.count(marker) == 1
        source = source.replace(marker, '    _ep.target_read(all_nodes_unique, mem is None and mail is None)\n' + marker)
        marker = '        if self.memory.partition:'
        assert source.count(marker) == 1
        source = source.replace(marker, '        _ep.cpu_write(all_nodes_unique, push_msg, send_msg, "begin")\n' + marker)
        marker = '        t5 = time.time()'
        assert source.count(marker) == 1
        source = source.replace(marker, marker + '\n        _ep.cpu_write(all_nodes_unique, push_msg, send_msg, "end")')
        scope = dict(method.__globals__, _ep=self)
        exec(compile(source, '<endpoint_target_update>', 'exec'), scope)
        TGNN.update_memory_and_send = scope['update_memory_and_send']
        (self.out / 'endpoint_target_update.py').write_text(source)

    def install(self, ns, TGNN, Memory, prefix):
        import torch
        import torch.distributed as dist
        import modules.memory as mm
        original_group = dist.new_group
        def group(*a, **kw):
            result = original_group(*a, **kw)
            self.groups[id(result)] = len(self.groups)
            return result
        dist.new_group = group

        # The compiled support method captured prefix.release already.
        source = (self.out / 'prefix_snapshot_prepare_input.py').read_text()
        marker = '        mem = self.memory.node_memory[pull_nodes]'
        assert source.count(marker) == 1
        source = source.replace(marker, '        _ep.support_begin(pull_nodes)\n' + marker)
        marker = '    _release_prefix_snapshot(self,b,pull_nodes,mem_ts,mail_ts)'
        assert source.count(marker) == 1
        source = source.replace(marker, '    _ep.support_ready(self,b,pull_nodes,mem,mail,mem_ts,mail_ts,_release_prefix_snapshot)')
        scope = dict(TGNN.prepare_input.__globals__, _ep=self)
        exec(compile(source, '<endpoint_prepare_input>', 'exec'), scope)
        TGNN.prepare_input = scope['prepare_input']
        (self.out / 'endpoint_prepare_input.py').write_text(source)

        source = inspect.getsource(type(prefix).receive_current_frontier)
        import textwrap
        source = textwrap.dedent(source)
        marker = '            ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(device));ready.synchronize()'
        assert source.count(marker) == 1
        source = source.replace(marker, marker + '\n            _ep.cached_state_ready(memory,cached_idx,cached_mem,cached_mail)')
        marker = '    uncached_mem=memory.node_memory[uncached_idx].to(device)'
        assert source.count(marker) == 1
        source = source.replace(marker, '    _ep.uncached_read(memory,uncached_idx)\n' + marker)
        scope = dict(type(prefix).receive_current_frontier.__globals__, _ep=self)
        exec(compile(source, '<endpoint_receive_frontier>', 'exec'), scope)
        prefix.receive_current_frontier = types.MethodType(scope['receive_current_frontier'], prefix)
        (self.out / 'endpoint_receive_frontier.py').write_text(source)

        original_reset = TGNN.reset
        def reset(model, *a, **kw):
            result = original_reset(model, *a, **kw)
            self.emit('memory_reset', note='Shared CPU reset; model and Adam continue')
            return result
        TGNN.reset = reset

        original_forward = TGNN.forward
        def forward(model, *a, **kw):
            self.forward_theta = self.theta
            row = self.emit('forward_issue_begin', theta_forward_version=self.theta) if self.selected() else None
            if row: self.point(row, 'prior_stream_work_done_gpu_ns')
            result = original_forward(model, *a, **kw)
            if row:
                row['host_issue_end_ns'] = time.perf_counter_ns()
                self.point(row, 'forward_stream_end_gpu_ns')
            return result
        TGNN.forward = forward

        original_step = torch.optim.Adam.step
        def step(optimizer, *a, **kw):
            input_version = self.theta
            result = original_step(optimizer, *a, **kw)
            self.step_number += 1
            self.theta = 'r%d:%s:e%d:b%d:step%d' % (self.rank, self.phase['name'], self.epoch, self.batch, self.step_number)
            if self.selected():
                self.emit('optimizer_step', optimizer_owner_rank=self.rank,
                          theta_forward_version=self.forward_theta,
                          theta_step_input=input_version, theta_step_output=self.theta)
            return result
        torch.optim.Adam.step = step

        original_pull, original_push = ns['pull_model'], ns['push_model']
        def pull(model, *a, **kw):
            result = original_pull(model, *a, **kw)
            self.theta = 'shared_model:%s:e%d' % (self.phase['name'], self.epoch)
            if self.selected(): self.emit('parameter_shared_pull', producer_version=self.theta)
            return result
        def push(model, *a, **kw):
            result = original_push(model, *a, **kw)
            if self.selected(): self.emit('parameter_shared_push', producer_version=self.theta)
            return result
        ns['pull_model'], ns['push_model'] = pull, push

        original_stage = Memory.send_mem
        def stage(memory, mem, mail, rank, world_size, group=None, dst=-1):
            if mem is not None and self.selected():
                nodes = memory.send_msg[self.batch//world_size]
                row = self.emit('state_payload', payload_batch=self.batch,
                                producer_version='train:e%d:b%d:r%d' % (self.epoch, self.batch, self.rank),
                                nodes=nodes.tolist(), tensor_count=2,
                                payload_bytes=sum(t.numel()*t.element_size() for t in [mem,mail]))
                self.point(row, 'payload_ready_gpu_ns')
                if mem.shape[0]:
                    self.payloads[id(mem)] = row
                    self.retained.append((row, [mem.detach(),mail.detach()]))
                else:
                    self.empty_state_payload = row
            return original_stage(memory, mem, mail, rank, world_size, group, dst)
        Memory.send_mem = stage

        def message(channel, direction, peer, group, payload=None):
            ctx = {k:payload[k] for k in ['phase','epoch','processing_batch']} if payload else self.context()
            src, dst = (self.rank, peer) if direction == 'send' else (peer, self.rank)
            gid = self.groups.get(id(group), 'unknown')
            key = (ctx['phase'], ctx['epoch'], channel, gid, src, dst, direction)
            sequence = self.seq[key]; self.seq[key] += 1
            mid = '%s:e%d:%s:g%s:%d>%d:m%d' % (*key[:4], src, dst, sequence)
            row = self.emit(channel + '_' + direction, context=ctx, message_id=mid,
                            source_rank=src, destination_rank=dst, group_id=gid, message_sequence=sequence,
                            payload_event_id=payload['event_id'] if payload else None,
                            payload_batch=payload.get('payload_batch') if payload else None)
            if direction == 'send' and ctx['phase'] == 'train' and any(a-2 <= ctx['processing_batch'] <= z+2 for a,z in WINDOWS):
                self.point(row, 'sender_stream_ready_gpu_ns')
            return row

        original_send, original_recv = prefix.native_send, prefix.native_recv
        def send(tensors, peer, group=None):
            channel = 'parameter' if tensors is not None else 'parameter_order'
            payload = self.payloads.pop(id(tensors[0]), None) if tensors else None
            row = message(channel, 'send', peer, group, payload)
            if payload: payload['message_id'] = row['message_id']
            result = original_send(tensors, peer, group)
            row['host_api_return_ns'] = time.perf_counter_ns()
            return result
        def recv(tensors, peer, group=None):
            channel = 'parameter' if tensors is not None else 'parameter_order'
            row = message(channel, 'recv', peer, group)
            result = original_recv(tensors, peer, group)
            row['host_wait_return_ns'] = time.perf_counter_ns()
            if tensors is not None:
                self.theta = 'message:' + row['message_id']
                if self.selected():
                    self.point(row, 'receive_ready_gpu_ns')
                    # The receive destination is overwritten by Adam. This one
                    # diagnostic copy is explicitly outside performance E1.
                    copy = torch.cat([t.detach().reshape(-1) for t in tensors])
                    row['diagnostic_copy_bytes'] = copy.numel()*copy.element_size()
                    self.retained.append((row, [copy]))
            return result
        prefix.native_send, prefix.native_recv = send, recv

        original_state_send, original_state_recv, original_req = mm.send, mm.recv, mm.recv_req
        def state_send(tensors, rank, peer, group=None):
            payload = self.payloads.pop(id(tensors[0]), None) if tensors else getattr(self, 'empty_state_payload', None)
            if tensors is None: self.empty_state_payload = None
            row = message('state' if tensors is not None else 'state_order', 'send', peer, group, payload)
            if payload: payload['message_id'] = row['message_id']
            result = original_state_send(tensors, rank, peer, group)
            row['host_api_return_ns'] = time.perf_counter_ns()
            return result
        def state_recv(tensors, rank, peer, group=None):
            row = message('state' if tensors is not None else 'state_order', 'recv', peer, group)
            result = original_state_recv(tensors, rank, peer, group)
            row['host_wait_return_ns'] = time.perf_counter_ns()
            return result
        class WaitProxy:
            def __init__(work_self, inner, tracker):
                work_self.inner, work_self.tracker = inner, tracker
            def wait(work_self, *a, **kw):
                result = work_self.inner.wait(*a, **kw)
                work_self.tracker['remaining'] -= 1
                if work_self.tracker['remaining'] == 0:
                    row = work_self.tracker['row']
                    row['host_wait_return_ns'] = time.perf_counter_ns()
                    if self.selected(): self.point(row, 'receive_ready_gpu_ns')
                return result
            def __getattr__(work_self, name): return getattr(work_self.inner, name)
        def state_req(tensors, rank, peer, group=None):
            row = message('state', 'recv', peer, group)
            self.last_state_recv = row
            result = original_req(tensors, rank, peer, group)
            row['host_api_return_ns'] = time.perf_counter_ns()
            if self.selected(): self.retained.append((row, [t.detach() for t in tensors]))
            tracker = {'row': row, 'remaining': len(result)}
            return [WaitProxy(req, tracker) for req in result]
        mm.send, mm.recv, mm.recv_req = state_send, state_recv, state_req

    def finish(self):
        import torch
        torch.cuda.synchronize()
        assert self.anchor is not None
        midpoint = sum(self.anchor_bracket)//2
        for row, field, event in self.cuda_points:
            row[field] = midpoint + round(self.anchor.elapsed_time(event)*1e6)
        for row, tensors in self.retained:
            digest = hashlib.sha256()
            for tensor in tensors:
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            row['payload_sha256'] = digest.hexdigest()
        with (self.out / 'endpoint_events.jsonl').open('w') as file:
            for row in self.rows: file.write(json.dumps(row) + '\n')
        (self.out / 'endpoint_contract.json').write_text(json.dumps({
            'rank': self.rank, 'windows': WINDOWS, 'boundary_context_batches': 2,
            'clock': 'perf_counter_ns, same host; per-device CUDA event mapping',
            'anchor_host_bracket_ns': self.anchor_bracket,
            'clock_mapping_half_bracket_ns': (self.anchor_bracket[1]-self.anchor_bracket[0])//2,
            'events': len(self.rows), 'retained_payloads': len(self.retained),
            'diagnostic_only': True,
            'limitations': ['CUDA event boundaries are stream fences, not actual kernel start/end',
                'Send completion is unobserved; native send does not wait for Work',
                'Parameter receive copy and CPU audits perturb this diagnostic',
                'CPU field versions describe storage provenance, not temporal correctness',
                'Logical message pairing and producer consistency require offline validation']}, indent=2) + '\n')
