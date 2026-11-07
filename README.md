# DiffSPECFEM

A runnable differentiation extension of **official SPECFEM2D**, pinned to
[`abf3196e3192af3343f2b402c82f34205577e672`](https://github.com/SPECFEM/specfem2d/tree/abf3196e3192af3343f2b402c82f34205577e672).
It computes a PyTorch data-objective VJP and sends that cotangent to a compiled
native exact discrete SH/Newmark adjoint. Production gradients use **one native
forward, one native adjoint, zero model finite differences and zero temporal AD**.
The new extension is copyright 2026 Sid and licensed GPL-3.0-or-later.
Upstream SPECFEM2D author notices are retained under GPL-3.0-or-later; see
`NOTICE` and the complete license text in `LICENSE`.

## See the native evidence

The animation is extracted from the **actual validated 32×32, two-rank native
SH histories**, with a fixed displacement scale and the complete 2 km square.
It uses the heterogeneous reference/starting model. All plots below use real
native arrays; the target changes four regional log parameters.

![Actual native SH wavefield, 44 saved post-step states](figures/native_sh_wavefield.gif)

| Physical contrast | Spatial native gradient |
|---|---|
| [![Starting and target density/Vs maps](figures/01_models.png)](figures/01_models.pdf) | [![Raw native coefficient and log-coordinate gradients](figures/04_spatial_gradients.png)](figures/04_spatial_gradients.pdf) |
| Exact two-layer starting/target models and differences; shared scales. | Raw SEM coefficient derivatives summed within each element; no smoothing. |
| **Data → AD cotangent** | **Independent native perturbation checks** |
| [![Target, starting, residual and AD data source](figures/03_data_and_adjoint_source.png)](figures/03_data_and_adjoint_source.pdf) | [![Every declared Taylor epsilon and centered derivative error](figures/05_taylor_evidence.png)](figures/05_taylor_evidence.pdf) |
| All four receivers and 800 native samples, including the duplicate station. | Both objectives, three parameter directions and all eight epsilons. |

Full-size plates, legends, exact sampling rules and QA are in the
[visual report](VISUAL_REPORT.md). Each PNG has an editable PDF/SVG companion;
[native snapshots](figures/02_native_wavefield.pdf),
[source data and hashes](evidence/visuals/manifest.json) and
[rendering code](scripts/visualize_evidence.py) are included.
Rebuild with `python scripts/visualize_evidence.py`; the alignment helper and
its [upstream attribution/license](scripts/figure_qa/README.md) are bundled,
so rendering requires only NumPy, matplotlib and Pillow, with no installed skill.

| Validated quantity | 32×32 native SH case |
|---|---:|
| Native MPI ranks / unique physical nodes | 2 / 16,641 |
| Time steps / DT / source peak frequency | 800 / 0.001 s / 6 Hz |
| Native source-frequency recommendation | ≤ 8.192 Hz |
| Original-native vs extension forward traces | Bitwise identical |
| L2 / processed model-dot relative error | 1.113×10⁻⁹ / 1.215×10⁻⁹ |
| Native sampling/scatter dot relative error | 1.214×10⁻¹⁶ |
| Full-history sampling-to-trace relative error | 1.568×10⁻¹⁶ |
| Selected plotted histories → saved traces | Exact in the visualization check |
| Production propagation per gradient | 1 native forward + 1 native adjoint |

This is a fixed discrete differentiation test with four regional coordinates.
The reference is a real two-layer model, not a Gaussian-blurred full-field
starting model. The 32×32 case meets native grid/source guidance; this is not a
continuum-convergence or full-field inversion claim. The original coarse and
failed validation attempts remain documented below and in `REPORT.md`.

This is a deliberately bounded application of the frozen generic skill
`differentiate-seismic-solver` v0.1.0. The original frozen packet is unchanged;
its SHA256 is `b09e0ee57324f715b80178c2a30cb26a2140b1ea3d5545ad8b15dc5680dca49f`
and its frozen Git source revision is `89d135f75f3d8521737645cab2e0c6ec49136f91`.
See `application_record.json`, `evidence/validation.json`,
`evidence/mpi2/validation.json`, `evidence/acceptance.json`, and `REPORT.md`.

## Supported numerical contract

- Planar, isotropic, lossless elastic **SH** waves, zero initial state,
  fixed point-force source and fixed off-grid displacement receivers.
- Native quadrilateral spectral elements, five GLL points per direction,
  native Newmark stepping, native natural reflecting/free boundaries.
- Native CPU execution with **one or two MPI ranks**, one thread per rank.
  The Python interface rejects other rank counts. GPU, P–SV, acoustic,
  attenuation, PML/Stacey absorption, anisotropy, periodicity, moving/noise
  sources and higher-order derivatives are outside this interface.
- The example is a heterogeneous 2 km square with 8×8 elements, 1089 unique
  nodes, two material regions, 800 time steps at 0.001 s, a 6 Hz force source,
  and four receivers including a duplicate and off-grid interpolation.
- The four active coordinates are relative log density in the bottom/top
  regions, then relative log shear speed in the bottom/top regions. Reference
  densities are `[2500,2800] kg/m³`, shear speeds `[1600,1900] m/s`. Compressional
  speed, mesh, time step, source, receiver positions, boundaries and all other
  settings remain fixed. The density/shear-speed map is `mu=rho*Vs**2`.
- Internal arithmetic, receiver export, native cotangent and kernels are
  double precision. Exporting float64 is not used to disguise single arithmetic.

An additional `examples/sh_two_layer_32` case refines only the mesh to 32×32
and its material element ranges. Source/receiver files, physical geometry,
material values, time step and step count are identical. Its native diagnostics
recommend `f0 <= 8.192 Hz` and `DT <= 0.0015417 s`; the actual 6 Hz and 0.001 s
meet both recommendations (CFL 0.3243). The complete two-rank suite also passes
for this case, with 16,641 unique nodes. This is a grid-guidance check, not a
multi-grid convergence study. Full history totals 214,860,824 bytes and is kept
outside Git with checksums. See `evidence/mpi2_32/validation.json`.

Native sample `n=1…800` is displacement **after** the native predictor/force/
corrector step. Upstream labels it `(n-1)*DT - USER_T0`, where `USER_T0=0.22 s`.
Receiver files are sorted by station name by Python. Native local receiver
ownership is `recloc`; each MPI rank selects its cotangent rows accordingly.
The native data source is the additive Lagrange interpolation transpose applied
to the AD cotangent at that same post-step state. There is no implicit `dt`,
force normalization, sign reversal, or array reversal at the file boundary.

## Exact discrete native adjoint

The numerical forward updates and force assembly are unchanged. A module and
three hooks are inserted into `src/specfem2D/iterate_time.F90`; the explicit patch
is `native/iterate_time.patch`, applied by `scripts/patch_native.py`.
The original native solver is built and copied **before** patching.

For the supported configuration, the native step is

```
u_n = u_old + dt*v_old + dt²/2*a_old
v_half = v_old + dt/2*a_old
a_n = M⁻¹(-K*u_n + f_n)
v_n = v_half + dt/2*a_n
```

The native module reverses these actual operations. With accumulated state
cotangents it first adds `dt/2*bar_v` to `bar_a`, forms
`w=M⁻¹*bar_a`, adds `-K*w` and `Cᵀ*bar_d_n` to `bar_u`, then reverses the
predictor. `-K*w` calls the **unchanged native elastic force routine**, rather
than a Python/AD wave solver. Local density and shear-modulus sensitivities are
accumulated as

```
dJ/drho_GLL -= W*Jgeo*w*a_n
dJ/dmu_GLL  -= W*Jgeo*(grad(u_n)·grad(w))
```

These are already discrete GLL coefficient derivatives including spatial
quadrature/Jacobian, rather than unweighted continuous kernel densities.
The raw rows contain `element,i,j,node,x,z,rho,mu,dJ/drho,dJ/dmu,material_id`.
No smoothing, illumination division, clipping or fitted scaling is applied.
The region map gives `g_logrho=sum(rho*g_rho+mu*g_mu)` and
`g_logVs=sum(2*mu*g_mu)` over unique local elements.

On two ranks, adjoint nodal states represent globally assembled quantities and
are replicated consistently at shared nodes. Only each **new** local receiver
source and each local stiffness force are summed with the native
`assemble_MPI_vector_el_blocking` routine. Existing replicated state cotangents
are not summed again. Element-local coefficient sensitivities are written once
per element and summed in Python by native material ID. The native sampling dot
probe uses inverse interface multiplicities to count each nodal product once.

Full post-step displacement and acceleration histories are written to disk and
loaded for reverse traversal. The adjoint reads both fields, uses no replay and
retains only its live adjoint state in addition. The serial history is
13,939,212 bytes including a 12-byte shape header. MPI ranks write disjoint
history files; shared interface nodes are stored on each participating rank.
Memory/storage grows with `NSTEP`; this implementation makes no checkpoint
scaling claim.

## Reproduction

On Linux with GCC/Gfortran and, for two ranks, OpenMPI:

```bash
python -m pip install -r requirements.txt
bash scripts/build_remote.sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/validate.py
python scripts/performance.py
bash scripts/build_mpi.sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/validate_mpi.py
python scripts/performance_mpi.py
python scripts/validate_mpi.py --case examples/sh_two_layer_32 --evidence evidence/mpi2_32
python scripts/performance_mpi.py --case examples/sh_two_layer_32 --evidence evidence/mpi2_32
```

The build scripts expect fresh, unmodified upstream source directories and pin
them explicitly. On the recorded host use
`/home/xiwang/anaconda3/envs/pytorch26/bin/python`; builds run `nice -n 10 make -j1`.
All propagation is native compiled Fortran. `scripts/validate*.py` uses model
finite differences **only as independent correctness checks**, never to return
production gradients.

A minimal objective-to-model-gradient call is:

```python
import numpy as np
from diffspecfem import Solver

solver = Solver(ranks=2, case="examples/sh_two_layer_32")  # original: omit case; ranks=1 also supported for original
observed = solver.forward(np.array([.025, -.02, .035, -.03]))
result = solver.value_gradient(np.zeros(4), observed, kind="processed")
print(result["value"], result["gradient"])
```

For another differentiable objective, call `forward(model, record=True)`, make
its data a float64 PyTorch leaf, obtain `torch.autograd.grad(loss, leaf)`, and
pass that cotangent to `adjoint`. The example processed objective contains a
fixed FIR convolution, `tanh`, receiver mixing (including the duplicated
receiver), and time-dependent weights. The returned model gradient is an
explicit NumPy vector; this wrapper does not claim PyTorch model double-backward
or batched concurrent solver contexts. Use separate work directories for
concurrent independent objectives.

## Evidence and limits

The coarse mesh is a discrete differentiation test, not a converged waveform
accuracy study. Upstream diagnostics recommend `f0 <= 2.048 Hz` for this mesh;
the chosen 6 Hz Ricker source has an underresolved high-frequency tail. The
additional 32×32 case above meets the native resolution recommendation. Every
forward/gradient check holds its declared mesh/source fixed.

Numerical tolerances were declared before validation. Every epsilon, objective,
first-order remainder and centered derivative is retained, including a failed
initial sweep that needed smaller epsilons to obtain two consecutive central
matches. Tolerances were not relaxed. Independent review identified an
algebra-only sampling probe; this was retained and superseded by actual compiled
native observation/scatter and history-to-trace tests. The source kernel hashes,
compiler output, native call commands, failed attempts, MPI scope extension and
review interventions are recorded outside the frozen skill packet.

Runtime/resource figures are observations on a shared host for this small
supported case. Host RSS is reported separately from history bytes; no GPU was
used. One successful application is not evidence of universal adaptation effort
or a causal benefit over a no-skill baseline. The application agent was fresh
with respect to private adapters, but the study is not author-blinded.
