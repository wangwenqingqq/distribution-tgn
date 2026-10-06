"""Verify the lightweight archive and recompute paired E1 / E2 checks offline."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path): return json.loads((ROOT/path).read_text())


def main():
    manifest=read('ARCHIVE_MANIFEST.json')
    for row in manifest['files']:
        path=ROOT/row['path']
        assert path.is_file() and not path.is_symlink(),row['path']
        assert len(path.read_bytes())==row['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
    contract=read('plans/critical_20261006_s1_contract.json')
    for name,digest in contract['source_hashes'].items():
        assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==digest,name
    assert read('analysis/source_inventory_local.json')['new_source']==contract['source_hashes']
    assert read('analysis/source_inventory_remote.json')['base_source_hashes_match']
    assert read('analysis/paired_qualification.json')['semantic']['passed']
    assert read('analysis/historical_native_lineage.json')['passed']
    performance=read('analysis/performance_paired_results.json')
    assert len(performance['pairs'])==3 and performance['paired_processes_total']==6
    ratios={name:[] for name in ['full_process_seconds','formal_train_seconds']}
    for pair in performance['pairs']:
        assert pair['valid_for_timing'] and pair['semantic']['passed']
        assert pair['source_hashes']==contract['source_hashes']
        for arm in ['native','arena']:
            run=pair['runs'][arm]; label=run['run_id']
            assert run['resources']['accepted']
            raw=read('runs/'+label+'/run.json')
            assert raw['status']=='passed' and raw['exit_code']==0
            assert raw['child_process_wall_seconds']==run['full_process_seconds']
            ranks=[read('runs/'+label+'/rank'+str(rank)+'/summary.json') for rank in range(2)]
            assert sum(r['work_totals_by_phase']['warm_up'] for r in ranks)==905172
            assert sum(r['work_totals_by_phase']['train'] for r in ranks)==3*905172
            epochs=[]
            for epoch in range(3):
                intervals=[r['epoch_intervals'][epoch] for r in ranks]
                assert sum(x['positive_edges'] for x in intervals)==905172
                epochs.append(max(x['global_host_end_offset_s'] for x in intervals)-min(x['global_host_start_offset_s'] for x in intervals))
            assert all(math.isclose(a,b,rel_tol=0,abs_tol=1e-9) for a,b in zip(epochs,run['global_epoch_seconds']))
            assert math.isclose(sum(epochs),run['formal_train_seconds'],rel_tol=0,abs_tol=1e-9)
            assert all(r['parameters_finite'] for r in ranks)
        for name in ratios:
            ratio=pair['runs']['native'][name]/pair['runs']['arena'][name]
            ratios[name].append(ratio)
            assert ratio==pair[name+'_speedup']
    for name, values in ratios.items():
        assert math.isclose(math.exp(sum(map(math.log,values))/len(values)),performance[name]['geometric_speedup'],abs_tol=1e-12)
    diag=read('analysis/e2_semantic_qualification.json')
    assert diag['semantic']['passed'] and diag['summary']['resources']['accepted']
    e2=read('analysis/e2_endpoints/endpoint_analysis.json')
    assert not e2['unpaired_or_ambiguous_messages'] and not e2['payload_hash_failures']
    assert not e2['batch_adjacency_failures'] and not e2['state_node_order_failures']
    assert e2['payload_hash_verified']==280 and e2['payload_hash_unobserved_outside_capture']==8
    assert e2['cpu_writes_observed']==1509 and e2['support_current_storage_mismatch_count']==0
    messages=[json.loads(x) for x in (ROOT/'analysis/e2_endpoints/matched_messages.jsonl').read_text().splitlines()]
    assert len(messages)==288
    assert sum(r['sender_payload_sha256']==r['receiver_payload_sha256'] for r in messages)==280
    for window in e2['windows']:
        first,last=window['global_batches']
        rows=[r for r in messages if first<=r['consumer_batch']<=last]
        assert len(rows)==window['matched_selected_messages']
        assert all(r['hash_verified'] and r['batch_adjacency_verified'] for r in rows)
        assert window['parameter_steps']==window['forward_vs_step_input_distinct']==last-first+1
        assert window['parameter_versions_unresolved']==0
    provenance=[json.loads(x) for x in (ROOT/'analysis/e2_endpoints/cpu_provenance_counts.jsonl').read_text().splitlines()]
    total=Counter()
    for row in provenance:total.update(row['counts'])
    assert dict(total)==e2['cpu_provenance_total']
    sources=read('analysis/e2_source_hashes.json')
    for key in ['executed_diagnostic_sources','offline_analysis_sources']:
        for name,digest in sources[key].items():
            assert hashlib.sha256((ROOT/'e2'/name).read_bytes()).hexdigest()==digest,name
    print(json.dumps({'files_verified':len(manifest['files']),'paired_runs':6,'global_formal_epochs':18,
        'full_process_speedup':performance['full_process_seconds']['geometric_speedup'],
        'formal_train_speedup':performance['formal_train_seconds']['geometric_speedup'],
        'selected_messages_hash_verified':280,
        'scope':'Offline file/statistics validation; checkpoint contents and full raw provenance remain server-only.'},indent=2))


if __name__=='__main__':main()
