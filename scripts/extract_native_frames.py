#!/usr/bin/env python3
"""Extract real native post-step states without rerunning the solver.

Run where the archived binaries live; stdout is a compact NPZ, not a figure.
The 12-byte little-endian header precedes [step, (u,a), local_node] float64.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('history_directory', type=Path)
p.add_argument('manifest', type=Path)
args = p.parse_args()
expected = json.loads(args.manifest.read_text())
indices = np.unique(np.r_[np.arange(0, 800, 20), [199, 399, 599, 799]])
arrays = {'step_index_zero_based': indices}
sources = []
for rank in range(2):
    name = f'ds_rank{rank:06d}_history.bin'
    path = args.history_directory / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = next(row for row in expected if Path(row['file']).name == name)
    assert digest == record['sha256'], (name, digest, record)
    assert path.stat().st_size == record['bytes']
    header = np.fromfile(path, dtype='<i4', count=3)
    assert header[1] == 800
    history = np.memmap(path, dtype='<f8', mode='r', offset=12,
                        shape=(int(header[1]), 2, int(header[0])))
    arrays[f'rank{rank}_header'] = header
    arrays[f'rank{rank}_states'] = np.asarray(history[indices])
    sources.append({'file': name, 'bytes': path.stat().st_size, 'sha256': digest,
                    'header_nglob_nstep_nrecloc': header.tolist()})
arrays['provenance_json'] = np.array(json.dumps({
    'sources': sources,
    'selection': 'union(range(0,800,20), [199,399,599,799]); no spatial subsampling',
    'layout': 'native post-step [frame, field=(displacement,acceleration), local_node]',
    'dt_s': 0.001, 'user_t0_s': 0.22,
    'native_elapsed_s': ((indices + 1) * 0.001).tolist(),
    'upstream_trace_label_s': (indices * 0.001 - 0.22).tolist(),
}))
np.savez_compressed(sys.stdout.buffer, **arrays)
