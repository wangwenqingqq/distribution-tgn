"""Validate logical endpoints, parameter lineage and CPU storage provenance."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


WINDOWS = [(16,47), (128,159), (140,147), (784,815), (1472,1503)]
FIELDS = ['node_memory', 'mailbox', 'node_memory_ts', 'mailbox_ts']


def distribution(values):
    if not values:
        return {'n': 0}
    ordered = sorted(values)
    return {'n': len(values), 'median': statistics.median(values),
            'p95_nearest_rank': ordered[min(len(ordered)-1, max(0, __import__('math').ceil(.95*len(ordered))-1))],
            'min': min(values), 'max': max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    args = parser.parse_args()
    run = args.root/'runs/critical_20261006_e2_native_s4101_a1'
    records = [json.loads(line) for rank in range(2)
               for line in (run/'results'/f'rank{rank}'/'endpoint_events.jsonl').read_text().splitlines()]
    contracts = [json.loads((run/'results'/f'rank{rank}'/'endpoint_contract.json').read_text()) for rank in range(2)]
    events = {r['event_id']:r for r in records}
    messages = defaultdict(lambda: {'send':[], 'recv':[]})
    for r in records:
        if r['kind'].endswith(('_send','_recv')) and 'message_id' in r:
            messages[r['message_id']][r['kind'].rsplit('_',1)[1]].append(r)
    sends = {}
    bad = []
    for mid, pair in messages.items():
        if len(pair['send']) != 1 or len(pair['recv']) != 1:
            bad.append({'message_id':mid, 'send_count':len(pair['send']), 'receive_count':len(pair['recv'])})
        else:
            sends[mid] = pair['send'][0]
    matched = []
    hash_bad = []
    for mid, send in sends.items():
        recv = messages[mid]['recv'][0]
        payload = events.get(send['payload_event_id'])
        if payload is None or payload['phase'] != 'train':
            continue
        row = {'message_id':mid, 'channel':send['kind'].removesuffix('_send'),
               'source_rank':send['rank'], 'destination_rank':recv['rank'],
               'payload_batch':payload['payload_batch'], 'consumer_batch':recv['processing_batch'],
               'producer_version':payload['producer_version'],
               'payload_bytes':payload['payload_bytes'],
               'sender_payload_event':payload['event_id'], 'send_event':send['event_id'], 'receive_event':recv['event_id'],
               'payload_ready_gpu_ns':payload.get('payload_ready_gpu_ns'),
               'sender_stream_ready_gpu_ns':send.get('sender_stream_ready_gpu_ns'),
               'receiver_host_call_begin_ns':recv['host_ns'],
               'receiver_host_wait_return_ns':recv.get('host_wait_return_ns'),
               'receive_ready_gpu_ns':recv.get('receive_ready_gpu_ns'),
               'clock_pair_error_bound_ns':sum(c['clock_mapping_half_bracket_ns'] for c in contracts),
               'actual_cuda_kernel_start_ns':None, 'actual_send_complete_ns':None,
               'pure_transport_latency_ns':None,
               'sender_payload_sha256':payload.get('payload_sha256'),
               'receiver_payload_sha256':recv.get('payload_sha256')}
        row['receiver_outside_capture_window'] = recv.get('payload_sha256') is None
        row['hash_verified'] = (payload.get('payload_sha256') == recv.get('payload_sha256')) if recv.get('payload_sha256') is not None else None
        row['node_order_verified'] = (payload.get('nodes') == recv.get('nodes')) if 'nodes' in payload and 'nodes' in recv else None
        row['batch_adjacency_verified'] = row['consumer_batch'] == row['payload_batch']+1
        if row['hash_verified'] is False and row['channel'] in ['parameter','state']:
            hash_bad.append({'message_id':mid, 'sender_hash':payload.get('payload_sha256'), 'receiver_hash':recv.get('payload_sha256')})
        matched.append(row)

    def resolve_theta(theta, seen=None):
        if not theta.startswith('message:'):
            return theta
        mid = theta.removeprefix('message:')
        send = sends.get(mid)
        payload = events.get(send['payload_event_id']) if send else None
        return payload.get('producer_version') if payload else None

    parameter_rows = []
    for row in records:
        if row['kind'] != 'optimizer_step' or row['phase'] != 'train':
            continue
        forward = resolve_theta(row['theta_forward_version'])
        step_input = resolve_theta(row['theta_step_input'])
        parameter_rows.append({'rank':row['rank'], 'processing_batch':row['processing_batch'],
                               'theta_forward_version':forward, 'theta_step_input':step_input,
                               'theta_step_output':row['theta_step_output'], 'optimizer_owner_rank':row['optimizer_owner_rank'],
                               'observed_distinct_versions':None if forward is None or step_input is None else forward != step_input})

    # Trace the actual write sites, independently for values and timestamps.
    # A read overlapping a write is unresolved, even when the final snapshot
    # happens to match the current shared table.
    beginnings = {}
    writes = []
    for row in records:
        if row['kind'] == 'cpu_write_begin': beginnings[row['producer_version']] = row
    for row in records:
        if row['kind'] == 'cpu_write_end':
            begin = beginnings[row['producer_version']]
            writes.append({'version':row['producer_version'], 'batch':row['processing_batch'],
                           'start':begin['host_ns'], 'end':row['host_ns'],
                           'value_nodes':row['value_nodes'], 'timestamp_nodes':row['timestamp_nodes']})
    by_field_node = defaultdict(list)
    for write in writes:
        for field in FIELDS:
            for node in write['timestamp_nodes' if field.endswith('_ts') else 'value_nodes']:
                by_field_node[(field,node)].append(write)
    starts = {}
    for key, entries in by_field_node.items():
        entries.sort(key=lambda w:w['start'])
        starts[key] = [w['start'] for w in entries]
    provenance = []
    totals = Counter()
    for row in records:
        if row['kind'] not in ['support_cpu_read_begin','target_cpu_read_begin','target_value_cpu_read_begin'] or row['phase'] != 'train':
            continue
        begin = row['host_ns']
        snapshots = [s for s in records if s['kind'] == 'support_cpu_snapshot' and s.get('parent_event_id') == row['event_id']]
        # Only the support snapshot has an observed end enclosing all four CPU
        # gathers. Target read calls are bounds only, not atomic snapshots.
        end = snapshots[0]['host_ns'] if snapshots else None
        versions = defaultdict(dict)
        counts = Counter()
        for node in row['nodes']:
            for field in row['fields']:
                if end is None:
                    version = None; status = 'read_end_unobserved'
                else:
                    entries = by_field_node.get((field,node), [])
                    cutoff = bisect_right(starts.get((field,node), []), end)
                    relevant = entries[:cutoff]
                    overlapping = [w for w in relevant if w['start'] < end and w['end'] > begin]
                    completed = [w for w in relevant if w['end'] <= begin]
                    if overlapping:
                        version = None; status = 'write_overlap_unresolved'
                    elif completed:
                        writer = max(completed,key=lambda w:w['end'])
                        version = writer['version']; status = 'observed_cpu_storage_writer'
                        if writer['batch'] > row['processing_batch']: counts['writer_batch_ahead_of_consumer'] += 1
                    else:
                        version = 'epoch0:shared_memory_reset'; status = 'initial_reset'
                versions[str(node)][field] = {'version':version, 'status':status}
                counts[status] += 1
        totals.update(counts)
        provenance.append({'read_event_id':row['event_id'], 'rank':row['rank'],
                           'processing_batch':row['processing_batch'], 'kind':row['kind'],
                           'read_begin_ns':begin, 'read_end_ns':end,
                           'counts':dict(counts), 'node_field_versions':dict(versions)})

    original = args.original_root/'runs/nsys_lastfm_p2_native_threads_s4101_e1_v2/results'
    old_events = [json.loads((original/f'rank{rank}'/'thread_intervals.json').read_text())['events'] for rank in range(2)]
    windows = []
    for first,last in WINDOWS:
        m = [r for r in matched if first <= r['consumer_batch'] <= last]
        theta = [r for r in parameter_rows if first <= r['processing_batch'] <= last]
        cpu = [r for r in provenance if first <= r['processing_batch'] <= last and r['kind']=='support_cpu_read_begin']
        timing = {}
        for channel in ['parameter','state']:
            selected = [r for r in m if r['channel'] == channel]
            fields = {
                'payload_ready_to_receive_ready_us':[ (r['receive_ready_gpu_ns']-r['payload_ready_gpu_ns'])/1e3 for r in selected if r['receive_ready_gpu_ns'] is not None],
                'producer_gpu_ready_minus_receiver_host_call_us':[ (r['payload_ready_gpu_ns']-r['receiver_host_call_begin_ns'])/1e3 for r in selected],
                'producer_stream_fence_minus_payload_ready_us':[ (r['sender_stream_ready_gpu_ns']-r['payload_ready_gpu_ns'])/1e3 for r in selected if r['sender_stream_ready_gpu_ns'] is not None],
                'receiver_host_call_to_wait_return_us':[ (r['receiver_host_wait_return_ns']-r['receiver_host_call_begin_ns'])/1e3 for r in selected if r['receiver_host_wait_return_ns'] is not None]}
            timing[channel] = {name:distribution(values) for name,values in fields.items()}
        threads = []
        for rank, entries in enumerate(old_events):
            chosen = [r for r in entries if r['phase']=='train' and r['epoch']==0 and first <= r['batch'] <= last]
            labels = ['worker_first_instruction/parameter','worker_first_instruction/state',
                      'thread_start/parameter','thread_start/state','thread_join','loss_scalar_read',
                      'state_gpu_ready_cpu_wait','parameter_recv_api','state_receive']
            threads.append({'rank':rank,'historical_host_interval_us':{
                label:distribution([(r['end_ns']-r['start_ns'])/1e3 for r in chosen if r['label']==label]) for label in labels}})
        counts = Counter()
        for row in cpu: counts.update(row['counts'])
        windows.append({'global_batches':[first,last], 'independent_window': (first,last)!=(140,147),
                        'matched_selected_messages':len(m), 'hash_verified':sum(r['hash_verified'] is True for r in m),
                        'parameter_steps':len(theta), 'forward_vs_step_input_distinct':sum(r['observed_distinct_versions'] is True for r in theta),
                        'parameter_versions_unresolved':sum(r['observed_distinct_versions'] is None for r in theta),
                        'support_node_field_provenance':dict(counts),
                        'new_diagnostic_endpoint_us':timing, 'historical_thread_intervals':threads})
    report = {
        'run_id':run.name, 'diagnostic_only':True, 'raw_events':len(records),
        'logical_messages':len(messages), 'unpaired_or_ambiguous_messages':bad,
        'selected_payload_messages':len(matched), 'payload_hash_failures':hash_bad,
        'payload_hash_verified':sum(r['hash_verified'] is True for r in matched),
        'payload_hash_unobserved_outside_capture':sum(r['hash_verified'] is None for r in matched),
        'batch_adjacency_failures':[r['message_id'] for r in matched if not r['batch_adjacency_verified']],
        'state_node_order_failures':[r['message_id'] for r in matched if r['node_order_verified'] is False],
        'cpu_writes_observed':len(writes), 'cpu_provenance_total':dict(totals),
        'support_current_storage_mismatch_count':sum(not all(r['same_as_current_shared_storage'].values()) for r in records if r['kind']=='support_cpu_snapshot'),
        'parameter_tensor_counts':sorted(set(r['tensor_count'] for r in records if r['kind']=='parameter_payload')),
        'parameter_payload_bytes':sorted(set(r['payload_bytes'] for r in matched if r['channel']=='parameter')),
        'cuda_clock_contracts':contracts,
        'windows':windows,
        'whole_batch_critical_path_verified':False,
        'limitations':['Old kernel/thread trace and new version diagnostic are separate runs; their clocks are not joined',
                      'Endpoint intervals include queueing, peer readiness and NCCL; no pure wire-time estimate',
                      'CPU provenance is resolved only for bounded support snapshots without overlapping writes',
                      'Target CPU gather ends and actual GPU consumer kernel boundaries remain unobserved',
                      'No state/parameter versions establish that changing native staleness is semantically safe',
                      'GPU compute union is observed kernel presence, not SM idle or occupancy']}
    output = args.root/'analysis/e2_endpoints'
    output.mkdir(exist_ok=True)
    for name, rows in [('matched_messages',matched),('parameter_versions',parameter_rows),('cpu_provenance',provenance)]:
        with (output/(name+'.jsonl')).open('w') as file:
            for row in rows: file.write(json.dumps(row)+'\n')
    (output/'endpoint_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['logical_messages','unpaired_or_ambiguous_messages','selected_payload_messages',
              'payload_hash_failures','batch_adjacency_failures','state_node_order_failures','cpu_writes_observed',
              'cpu_provenance_total','support_current_storage_mismatch_count']},indent=2))


if __name__=='__main__': main()
