#!/usr/bin/env python3
"""Export a whitelist of actual P-SV results, without redistributing input models.

SPDX-License-Identifier: GPL-3.0-or-later
Full native histories are temporary gradient storage; validation retains their
hashes and selected states before reusing the run directory. Private input,
coefficient and perturbation arrays are not exported. This command runs no simulation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def package(source, destination, ranks):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Use a new output directory; keep the original run intact')
    validation = json.loads((source / 'validation.json').read_text())
    if not validation.get('passed'):
        raise ValueError('Only passed runs can be exported as validated evidence')
    destination.mkdir(parents=True)
    manifest = dict(
        scope='actual native coupled P-SV results; no input material arrays',
        source_case=validation['case'], source_validation_sha256=digest(source / 'validation.json'),
        excluded=['tomography and prepared material arrays', 'native databases',
                  'full six-field history', 'physical perturbation directions (model times known patterns)',
                  'rho/lambda/mu columns from native derivative tables'], sources=[],
        raw_derivative_columns=['element', 'i', 'j', 'node', 'x_m', 'z_m',
                                'dJ_drho', 'dJ_dlambda', 'dJ_dmu', 'solid_flag'],
        wavefield='unchanged native compute_vector_one_element, UX/UZ in both phases; element-local values, no phase averaging')

    def record(path, output):
        manifest['sources'].append(dict(source=str(path.relative_to(source)),
                                        bytes=path.stat().st_size, sha256=digest(path),
                                        output=str(output.relative_to(destination))))

    def copy(path, output):
        if not path.exists():
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
        record(path, output)

    for name in ['validation.json', 'native_calls.jsonl', 'target.npy', 'native_reference.npy']:
        copy(source / name, destination / name)
    for kind in ['l2', 'processed']:
        path = source / f'gradient_{kind}.npz'
        with np.load(path, allow_pickle=False) as original:
            result = {key: original[key] for key in ['gradient', 'data', 'cotangent', 'sample_time']}
            result['raw_derivatives'] = original['raw'][:, [0, 1, 2, 3, 4, 5, 9, 10, 11, 12]]
            if not all(np.isfinite(array).all() for array in result.values()):
                raise ValueError('Nonfinite result array')
        output = destination / path.name
        np.savez_compressed(output, **result)
        record(path, output)
    for archive in ['native_baseline', 'target_native', 'gradient_l2', 'gradient_processed']:
        parent = source / archive
        for path in parent.glob('*.log'):
            copy(path, destination / archive / path.name)
        for path in parent.glob('*.resources.txt'):
            copy(path, destination / archive / path.name)
        for name in ['Par_file', 'SOURCE', 'STATIONS', 'interfaces.dat', 'interfaces_elastic_analytic.dat']:
            copy(parent / 'DATA' / name, destination / archive / 'DATA' / name)
        copy(parent / 'history_manifest.json', destination / archive / 'history_manifest.json')
        for rank in range(ranks):
            prefix = f'psv_rank{rank:06d}_'
            for suffix in ['coupling_probe.txt', 'coupling_geometry.txt', 'sampling.txt',
                           'sampling_native_nodes.txt', 'sampling_native_receivers.txt',
                           'gll_geometry.txt', 'nodes.txt', 'route.txt']:
                copy(parent / 'OUTPUT_FILES' / (prefix + suffix),
                     destination / archive / 'OUTPUT_FILES' / (prefix + suffix))
    # Native snapshots contain physical displacement, not input coefficients.
    # Export every saved step, all GLL points and both components, per rank.
    steps_reference = None
    for rank in range(ranks):
        path = source / 'gradient_l2' / 'OUTPUT_FILES' / f'psv_rank{rank:06d}_displacement_snapshots.bin'
        with path.open('rb') as stream:
            nspec, nstep, nx, nz, ndim = map(int, np.fromfile(stream, dtype='<i4', count=5))
            if nx != 5 or nz != 5 or ndim != 2:
                raise ValueError('Unexpected native vector snapshot header')
            stride = 4 + ndim * nx * nz * nspec * 8
            if (path.stat().st_size - 20) % stride:
                raise ValueError('Incomplete native snapshot record')
            count = (path.stat().st_size - 20) // stride
            steps = np.empty(count, dtype=np.int32)
            displacement = np.empty((count, ndim, nspec * nx * nz))
            for frame in range(count):
                steps[frame] = np.fromfile(stream, dtype='<i4', count=1)[0]
                # Native Fortran layout: component, i, j, element.
                values = np.fromfile(stream, dtype='<f8', count=ndim * nx * nz * nspec)
                displacement[frame] = values.reshape(-1, ndim).T
        if not np.isfinite(displacement).all() or np.any(np.diff(steps) <= 0):
            raise ValueError('Invalid native snapshots')
        if steps_reference is not None and not np.array_equal(steps, steps_reference):
            raise ValueError('Rank snapshot schedules differ')
        steps_reference = steps
        geometry_path = path.with_name(f'psv_rank{rank:06d}_gll_geometry.txt')
        geometry = np.loadtxt(geometry_path, ndmin=2)
        if geometry.shape != (nspec * nx * nz, 6):
            raise ValueError('Native geometry/vector shape mismatch')
        output = destination / f'wavefield_rank{rank:06d}.npz'
        np.savez_compressed(output, step_one_based=steps, displacement=displacement,
                            geometry=geometry, nstep=np.array(nstep),
                            component_names=np.array(['UX', 'UZ']))
        record(path, output)
        record(geometry_path, output)
    manifest['outputs'] = [dict(path=str(path.relative_to(destination)), bytes=path.stat().st_size,
                                sha256=digest(path)) for path in sorted(destination.rglob('*')) if path.is_file()]
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--ranks', type=int, default=2)
    arguments = parser.parse_args()
    result = package(arguments.source, arguments.destination, arguments.ranks)
    print(json.dumps(dict(files=len(result['outputs']),
                          bytes=sum(item['bytes'] for item in result['outputs'])), indent=2))
