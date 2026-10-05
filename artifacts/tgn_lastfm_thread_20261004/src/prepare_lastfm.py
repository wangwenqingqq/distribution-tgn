"""Prepare canonical JODIE LastFM, retaining real features and chronological order."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pydeps'))
import numpy as np
import pandas as pd

raw = ROOT / 'raw/lastfm.csv'
assert raw.stat().st_size == 38770679, 'Incomplete official download'
df = pd.read_csv(raw, skiprows=1, header=None)
assert len(df) == 1293103, len(df)
users, user_names = pd.factorize(df.iloc[:, 0], sort=True)
items, item_names = pd.factorize(df.iloc[:, 1], sort=True)
items += len(user_names)
times = df.iloc[:, 2].to_numpy(dtype=np.float64)
assert np.isfinite(times).all() and (np.diff(times) >= 0).all()
features = df.iloc[:, 4:].to_numpy(dtype=np.float32)
assert features.shape[1] > 0 and np.isfinite(features).all()
cuts = np.quantile(times, [.70, .85])
train_end, val_end = [int(np.searchsorted(times, t, side='right')) for t in cuts]
split = np.zeros(len(df), dtype=np.int8)
split[train_end:val_end] = 1
split[val_end:] = 2
out = ROOT / 'data/pipe/LASTFM'
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'eid': np.arange(len(df)), 'src': users, 'dst': items,
              'time': times, 'ext_roll': split}).to_csv(out / 'edges.csv', index=False)
np.save(out / 'edge_features.npy', features)
np.savez_compressed(ROOT / 'data/events.npz', src=users.astype(np.int32),
                    dst=items.astype(np.int32), ts=times, split=split)
manifest = {'dataset': 'JODIE LastFM temporal user-item interactions',
            'source_url': 'https://snap.stanford.edu/jodie/lastfm.csv',
            'citation_url': 'https://snap.stanford.edu/jodie/',
            'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
            'events': len(df), 'users': len(user_names), 'items': len(item_names),
            'nodes': len(user_names) + len(item_names),
            'edge_feature_dim': features.shape[1], 'edge_features_all_zero': bool((features == 0).all()),
            'node_features': 'absent, matching dataset; no artificial features',
            'train_events': train_end, 'validation_events': val_end - train_end,
            'test_events': len(df) - val_end, 'split_timestamp_quantiles': cuts.tolist(),
            'split_rule': '70/85% timestamp quantiles; <= boundary retained in earlier split; equal timestamps not split',
            'node_ids': 'zero-based disjoint user/item namespaces; sorted original IDs',
            'event_ids': 'zero-based original CSV row order; no padding feature row',
            'timestamp_rule': 'original relative time; split in float64, sampler uses its native float32',
            'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in out.iterdir() if p.is_file()}}
(ROOT / 'analysis/data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
