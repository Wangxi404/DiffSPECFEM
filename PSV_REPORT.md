# Native P–SV: method and reproduction

This extension starts from the **official SPECFEM2D source**, pinned to
[`abf3196e3192af3343f2b402c82f34205577e672`](https://github.com/SPECFEM/specfem2d/tree/abf3196e3192af3343f2b402c82f34205577e672).
The source tree was clean before instrumentation; its original MPI/double
executable was saved before the extension was compiled. This application did
not use an earlier private DiffSpecfem implementation. The previous public SH
adapter remains a separate regression, documented in [SH_REGRESSION.md](SH_REGRESSION.md).

## Numerical contract

| Quantity | Contract |
|---|---|
| Physics | Planar, isotropic, coupled acoustic water and elastic P–SV |
| Propagation | Official CPU/MPI force, coupling and Newmark kernels, float64 |
| Active model | Physical solid-grid Vp, Vs and density, each 67×200 |
| Fixed quantities | Water coefficients, mesh, source, receivers, DT and boundaries |
| Water | Vp = 1500 m/s; Vs = 0; density = 2450 kg/m³, matching the common prepared comparison inputs |
| Boundaries | Native pressure-free water surface and reflecting/free exterior |
| Observation | Both native Cartesian solid displacement components, full time sampling |
| Storage | Six native state arrays per step on disk; one-frame reverse reads; no replay |
| Production gradient | One native forward → data-only Torch VJP → one native coupled adjoint → static material/interpolation transpose |

Two MPI ranks, one thread per rank and `nice -n 10` are used in the recorded
coupled validations. GPU execution, attenuation, PML/Stacey absorption,
anisotropy, poroelasticity, axisymmetry, active water/source/geometry and
acoustic receivers are outside this interface. These are fixed-discretization
derivative tests; no boundary absorption or continuum-convergence claim is made.

## Reversing the actual coupled step

With `u` the elastic displacement and `chi` the acoustic displacement potential,
the official acoustic stage precedes the elastic stage:

```
chi_ddot = M_a^-1 (-K_a chi + B u)
a        = M_e^-1 (-K_e u - B^T chi_ddot + f)
```

The extension reverses the native predictor, the two force stages, velocity
correctors and both acoustic free-surface projections in their actual order.
It calls the unchanged native stiffness and coupling routines. It does not
use a PyTorch wave propagator, a model finite-difference sweep or checkpoint
replay to produce gradients. Native saved continuum alpha/beta kernels are
not substituted for derivatives of this discrete plane-strain update.

At each elastic GLL occurrence, the extension accumulates raw coefficient
derivatives in `(rho, lambda, mu)`. Quadrature and geometric Jacobians are
already included. With `w = M_e^-1 bar_a`, the mass derivative is
`g_rho -= W J (w · a)`; the two stiffness derivatives follow the native
plane-strain bilinear form. The physical map is

```
lambda = rho (Vp² - 2 Vs²)
mu     = rho Vs²

g_Vp  = P^T [2 rho Vp g_lambda]
g_Vs  = P^T [2 rho Vs (g_mu - 2 g_lambda)]
g_rho = P^T [g_rho_mass + (Vp² - 2 Vs²) g_lambda + Vs² g_mu]
```

`P` is the official tomography reader's bilinear interpolation, including its
constant border extension. The physical variables are interpolated **before**
forming lambda/mu. Each element-local GLL contribution is counted once across
ranks; model kernels are not divided by shared-node multiplicity. Only newly
created nodal force/source contributions are assembled with native MPI routines.

The receiver data become a new float64 Torch leaf. AD differentiates the
chosen objective with respect to that leaf. Both L2 and an FIR-filtered,
`tanh`-transformed, receiver/component-mixed, time-weighted objective are tested.
The data cotangent is scattered with the transpose of the actual native
off-grid receiver interpolation, at the matching post-step state. There is no
implicit DT weighting, sign change or file-order time reversal.

## Full Marmousi-II mapping

The input is the same **78×200, 45 m prepared Marmousi-II** model used by the
HybridADFWI comparison: original x = 5000…13955 m, depth = 0…3465 m, with
sigma = 150 m Gaussian initial-model smoothing within the solid. All six
true/initial material arrays were checked byte-for-byte against those inputs.
All 11 water sample rows are retained; no water-to-rock replacement or positive
Vs floor is applied. Density is deliberately constant at 2450 kg/m³ in both
models, including the water, following that comparison setup.

The native conforming fluid/solid interface is at depth 450 m. Solid samples
start at depth 495 m, so the official reader extends the first solid row
one-sided through this 45 m interval. Water coefficients belong to the separate
acoustic material. This explicit SEM representation differs from the adjacent
water/solid samples of an FD grid; sharing prepared input values does not make
the two simulators' discretizations or waveforms identical.

The force source is at original `(x, depth) = (9500, 540) m`. All 41 solid
receivers are at depth 540 m, span original x = 5100…13855 m and record both
UX and UZ. Each mesh/source/time configuration is stated separately in the
[visual report](VISUAL_REPORT.md). The input-sample model image uses nearest
sample cells; its 450 m interface line is the native physical boundary, not a
claim that a nearest-pixel water cell edge defines the SEM interface.

## Reproduce

On Linux with GCC/Gfortran and OpenMPI:

```bash
python -m pip install -r requirements-marmousi.txt
bash scripts/build_psv.sh

# Small reproducible nonuniform coupled fixture; no third-party model needed.
python examples/psv_coupled_unit/DATA/prepare_fixture.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/validate_psv.py \
  --model examples/psv_coupled_unit/DATA/unit_model.npz \
  --case examples/psv_coupled_unit --par-file Par_file_tomography \
  --evidence evidence/my_psv_unit

# Download with checksums and prepare the full water-bearing Marmousi-II model.
python scripts/prepare_marmousi.py --mode full-psv
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/validate_psv.py \
  --model data/prepared/marmousi2_full_psv_solid_domain.npz \
  --case examples/psv_marmousi2_validation64 --evidence evidence/my_marmousi64
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/validate_psv.py \
  --model data/prepared/marmousi2_full_psv_solid_domain.npz \
  --case examples/psv_marmousi2_full --evidence evidence/my_marmousi128 \
  --mode illustration --directions mixed localized
```

The clean-source build uses `make -j1`. The build script rejects an already
modified vendor checkout; retain the baseline, or use a fresh directory for
an independent rebuild. Validation refuses to overwrite an existing result
namespace. Full histories are disk-intensive and deliberately remain outside
Git. Model finite differences appear only in independent validation calls.
Validation hashes the full histories and saves selected native states before
the next forward call reuses its work directory. The complete multi-gigabyte
history is temporary gradient storage, not a permanently retained archive.

For a different differentiable objective:

```python
import numpy as np
import torch
from diffspecfem import PSVSolver

m = np.load("data/prepared/marmousi2_full_psv_solid_domain.npz")
initial = np.stack([m[key + "_initial"] for key in ("vp", "vs", "rho")])
truth = np.stack([m[key + "_true"] for key in ("vp", "vs", "rho")])
s = PSVSolver(m["x_native"], m["z_native"], "examples/psv_marmousi2_full")
observed = s.forward(truth)
data = s.forward(initial, record=True)
leaf = torch.tensor(data, dtype=torch.float64, requires_grad=True)
loss = ((torch.tanh(leaf / .02) - torch.tanh(torch.tensor(observed) / .02)) ** 2).sum()
cotangent, = torch.autograd.grad(loss, leaf)
gradient, raw = s.adjoint(cotangent.detach().numpy())
print(gradient.shape)  # [Vp, Vs, rho], each 67 x 200 solid input samples
```

The interface returns an explicit NumPy model gradient for inversion. It does
not advertise model double-backward or concurrent objectives sharing one work
directory. Use separate work directories for independent solver contexts.

## Evidence and application history

[psv_application_record.json](psv_application_record.json) identifies the frozen
generic skill v0.2.0, exact source/build/patch, attempts, repairs and supported
scope. The immutable skill packet remains unchanged. A P–SV-specific integrity
check was omitted before implementation; the record explicitly reports this
omission and the later integrity checks, without inventing a pre-check.
This follow-up inherited an existing SH adapter and is not a blinded skill
effectiveness experiment or a measurement of universal conversion effort.

The first unit attempt failed native source-frequency guidance; the next
reached an overly restrictive per-rank domain guard. The repair pads histories
on ranks lacking a local domain while preserving official native allocations.
Both failures and the successful rerun are retained. Numerical thresholds were
not relaxed. Build/source provenance is in
[evidence/psv_build/provenance.json](evidence/psv_build/provenance.json).

Public Marmousi-II bundles are whitelist exports: traces, AD sources, raw
derivatives, native vector fields, geometry, logs and verification. They omit
tomography, prepared input model arrays, native databases and physical
perturbation directions that would reproduce the input arrays. The raw
coefficient derivative tables omit rho/lambda/mu values. Original inputs remain
available through the checksum-verified preparation recipe and are not
relicensed by this software. See
[model provenance](evidence/marmousi2_full_psv_preparation.json) and the
[AGL download page](https://www.agl.uh.edu/downloads/downloads.htm).
