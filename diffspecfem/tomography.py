"""Static native tomography interpolation and its physical-coordinate transpose.

SPDX-License-Identifier: GPL-3.0-or-later
No wave propagation or temporal AD is implemented in this module.
"""
from pathlib import Path
import numpy as np

class BilinearMap:
    """Sparse interpolation and its exact Euclidean transpose.

    Mirrors ``interpolation:interpolate/searchInf`` in the pinned native
    ``define_external_model_from_tomo_file.f90`` for increasing rectilinear
    axes, including constant border extension. TINYVAL is 1.e-9 in this pin.
    """
    def __init__(self, x, z, query_x, query_z, tiny=1.e-9):
        self.x, self.z = np.asarray(x, float), np.asarray(z, float)
        self.shape = (len(self.z), len(self.x))
        if min(self.shape) < 2 or np.any(np.diff(self.x) <= 0) or np.any(np.diff(self.z) <= 0):
            raise ValueError('Tomography axes must be strictly increasing with at least two samples')
        qx, qz = np.broadcast_arrays(np.asarray(query_x, float), np.asarray(query_z, float))
        self.query_shape = qx.shape
        def axis_weights(a, q):
            q = q.ravel().copy()
            q[(q < a[0]) | (np.abs(q-a[0]) < tiny)] = a[0]
            q[(q > a[-1]) | (np.abs(q-a[-1]) < tiny)] = a[-1]
            upper = np.searchsorted(a, q, side='left')
            upper = np.clip(upper, 0, len(a)-1)
            lower = np.where(np.abs(a[upper]-q) > tiny, upper-1, upper)
            lower = np.clip(lower, 0, len(a)-2)
            fraction = (q-a[lower])/(a[lower+1]-a[lower])
            return lower, fraction
        i, tx = axis_weights(self.x, qx)
        j, tz = axis_weights(self.z, qz)
        self.indices = np.stack([j*len(self.x)+i, j*len(self.x)+i+1,
                                 (j+1)*len(self.x)+i, (j+1)*len(self.x)+i+1], axis=1)
        self.weights = np.stack([(1-tx)*(1-tz), tx*(1-tz), (1-tx)*tz, tx*tz], axis=1)

    def apply(self, values):
        values = np.asarray(values, float)
        if values.shape != self.shape:
            raise ValueError(f'Expected model grid {self.shape}, got {values.shape}')
        return np.sum(values.ravel()[self.indices]*self.weights, axis=1).reshape(self.query_shape)

    def transpose(self, cotangent):
        cotangent = np.asarray(cotangent, float)
        if cotangent.shape != self.query_shape:
            raise ValueError('Interpolation cotangent shape mismatch')
        out = np.zeros(np.prod(self.shape))
        np.add.at(out, self.indices.ravel(), (cotangent.ravel()[:, None]*self.weights).ravel())
        return out.reshape(self.shape)


def write_tomography(path, x, z, vp, vs, rho):
    """Write the native five-column regular-grid tomography format."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    vp, vs, rho = (np.asarray(v, float) for v in (vp, vs, rho))
    shape = (len(z), len(x))
    if any(a.shape != shape for a in (vp, vs, rho)):
        raise ValueError('Tomography material arrays must match (nz,nx)')
    if not all(np.isfinite(a).all() and np.min(a) > 0 for a in (vp, vs, rho)):
        raise ValueError('Solid tomography requires finite strictly positive Vp, Vs and rho')
    if not np.allclose(np.diff(x), x[1]-x[0], rtol=1.e-12, atol=1.e-9) or not np.allclose(np.diff(z), z[1]-z[0], rtol=1.e-12, atol=1.e-9):
        raise ValueError('Native tomography requires regular increasing axes')
    if x[1] <= x[0] or z[1] <= z[0]:
        raise ValueError('Native tomography axes must increase')
    xx, zz = np.meshgrid(x, z)
    with Path(path).open('w') as f:
        f.write(f'{x[0]:.17g} {z[0]:.17g} {x[-1]:.17g} {z[-1]:.17g}\n')
        f.write(f'{x[1]-x[0]:.17g} {z[1]-z[0]:.17g}\n{len(x)} {len(z)}\n')
        f.write(' '.join(f'{v:.17g}' for a in (vp, vs, rho) for v in (a.min(), a.max()))+'\n')
        np.savetxt(f, np.column_stack([xx.ravel(), zz.ravel(), vp.ravel(), vs.ravel(), rho.ravel()]), fmt='%.17g')
