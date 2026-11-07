# Independent dense Marmousi-II review

Final status: **all delivered claims passed review in their declared scopes; ready for release.** Numerical raw-artifact review passed for the small synthetic coupled P-SV test, complete coarse Marmousi-II verification and fine-mesh illustration. Public numerical sources passed exact numeric-hash comparison against accepted private native outputs, the public tree excludes base model arrays, and root plus the visual QA agent closed final rendered QA. The initial SH source audit below is preserved as history and does not satisfy the clarified P-SV request. This reviewer has not built or run the native solver and has not changed producer implementation, inputs, plots, or README. Full five-direction, six-step Taylor acceptance applies to the coarse case; fine acceptance retains its mixed/localized directional scope.

## Sources inspected

- Installed differentiation skill and `references/derivative-contract.md`, `references/numerical-acceptance.md` (read only).
- Local `DiffSPECFEM/native/discrete_adjoint.f90`, `diffspecfem/adapter.py`, existing example inputs, previous independent review.
- Read-only SSH to `hpnewpc3-via-vps:/home/xiwang/DiffSPECFEM/vendor/specfem2d`, source pin `abf3196e3192af3343f2b402c82f34205577e672`: `src/specfem2D/define_external_model_from_tomo_file.f90`, `read_external_model.f90`, `setup_mesh.F90`, `invert_mass_matrix.F90`, `compute_forces_viscoelastic_calling_routine.F90`, `compute_add_sources_viscoelastic.f90`, `iterate_time.F90`. Remote ripgrep was unavailable; source search used find/grep.

## Exact forward material map

For MODEL='tomo', `define_external_model_from_tomo_file.f90:407–422` loops over every element-local GLL entry. When `kmato(ispec)==tomo_material`, it obtains `xmesh=coord(1,ibool(i,j,ispec))` and `zmesh=coord(2,...)` and independently interpolates **physical rho, Vp, Vs** from the tomography grid. It does not interpolate mu or log parameters. `setup_mesh.F90:676–686` then reads these interpolated values and sets `mul=rhol*vs*vs`; lines 777–779 save rho/mu/kappa. In planar isotropic SH without attenuation/PML/Stacey, rho and mu are the relevant active coefficients, with Vp held fixed.

Let P be the actual fixed grid-to-element-GLL interpolation map, r=P rho_grid, s=P Vs_grid, and mu=r*s². The raw exported per-GLL coefficient derivatives are gr=dJ/dr and gm=dJ/dmu. Then the physical grid derivatives are:

    g_rho_grid = P^T (gr + s² gm)
    g_Vs_grid  = P^T (2 r s gm)

Only after this transpose, if active coordinates are independent grid log ratios, multiply by the corresponding **grid** physical rho and Vs. Equivalently use static AD of the exact material map. The simpler regional expression `sum(r_GLL gr + mu_GLL gm)` cannot be scattered directly as a grid-log derivative when log parameters are exponentiated before interpolation; multiplying by interpolated rho is not interchangeable with multiplying the grid nodes after P^T. Interpolating mu_grid=rho_grid*Vs_grid² defines a different forward model and would require a different VJP.

Native interpolation must be verified from actual exported rhostore/mustore: reconstruct rho_GLL and sqrt(mu_GLL/rho_GLL) from the declared grid and compare. A Python-only dot product of a reused interpolation matrix against its own transpose proves algebra but not agreement with the native material map. A static material-map FD/VJP test should test rho-only, Vs-only, and mixed dense directions, including boundary grid nodes.

## File coordinates, axes, and edges

- `read_tomo_file` documents the file at lines 549–589; first four records are `(origin_x,origin_z,end_x,end_z)`, `(dx,dz)`, `(NX,NZ)`, `(vpmin,vpmax,vsmin,vsmax,rhomin,rhomax)`.
- Data columns are **x z Vp Vs rho**, x runs fastest for each increasing z. The implementation reads x from the first NX entries (`:730`), z from every NX-th flat entry (`:751`), and assigns each field by `NX*(j-1)+i` (`:753–758`). For a NumPy `(nz,nx)` field, C-order flattening gives the documented order; a NumPy `(nx,nz)` field needs the corresponding transpose or Fortran flattening. Do not infer axes solely from shape.
- Source comments specify meters, m/s and kg/m³ (`:572–585`). If paper rho is in g/cm³, the declared conversion is ×1000; if speeds are km/s, conversion is ×1000. Mu is in Pa under SI. Explicitly report the actual data range before and after conversion rather than relying on convention.
- Native coordinate 2 is increasing z; the existing rectangular example has its top at z=0 and bottom at z=-2000. A paper depth-positive-down array must be explicitly mapped, for example z=-depth after a documented crop-origin shift. Increasing file z typically requires reversing the depth rows. Source/receiver coordinates and plotted depth axes must use the same map.
- Interpolation module `:146–335` uses bilinear interior interpolation and extends **border values**, rather than linear extrapolation, outside the sampled grid (corners constant, side regions 1D). The code has TINYVAL boundary branches and `searchInf` chooses the lower cell except that a point within TINYVAL of an interior grid node may use that node as the lower index. At the maximum endpoint it uses the final cell. For an interior grid exactly on a node either valid cell gives the same value; coordinate-level comparison remains the best evidence.
- The current source reads header min/max into variables but does **not clamp rho/Vp/Vs to them** in `define_external_model_from_tomo_file`. Do not implement an unsupported clipping derivative based on header prose. The only limits in the inspected interpolator routine are optional Q limits, irrelevant when attenuation is disabled.
- Raw-row coordinates must all fall within the declared model rectangle if border extension is not intended. Preserve the original full paper arrays plus crop indices, spacing, origin, axis/flip/transpose rules and hashes. A solid-only crop must be labeled as such, including original-depth offset. It is not the complete water-bearing Marmousi-II physics case.

## Water and domain flags

`read_external_model.f90` checks tomography-provided Vs against the initially meshed element type: an elastic element redefined with Vs<TINYVAL aborts. The existing custom reverse rejects any acoustic elements. Therefore zero-Vs water is unsupported by this native SH adjoint. A documented water-free solid crop is a valid restricted test; replacing Vs=0 with invented positive values would alter the model and must not be called native Marmousi-II water.

## Element duplication, MPI, and weights

Raw kernels are derivatives for **element-local GLL coefficient entries**, not unique global-node densities. Native `invert_mass_matrix.F90:255–256` accumulates each local element contribution `wxgll*wzgll*rho*jacobian` to the node mass. Local native reverse already includes the same quadrature/Jacobian in both gr and gm. Exported rows on shared element faces may have the same physical coordinates but represent distinct element integrals. Both are needed.

Consequently:

- Sum/scatter **all** rows from all rank raw files exactly once through the material-map transpose.
- Do not divide raw rows by global-node/MPI replica multiplicity.
- Do not multiply raw gr/gm by an additional SEM weight or by grid-cell area.
- Do not deduplicate by `(x,z)` before computing the model VJP.
- Node multiplicity weighting belongs only in unique-node state/probe inner products. It does not apply to element coefficient cotangents.
- A full-grid raw model gradient has one derivative per active input grid coordinate. A support or water mask must be fixed and included in the actual model map and in all perturbations. Display-only masking, illumination scaling, smoothing or clipping cannot be used in the quantitative raw-gradient gate.

The existing custom reverse maintains assembled state replicas, assembles only a fresh local receiver source before adding to bu, and separately assembles the native force transpose. This preserves the established two-rank route; the dense material map should not change that communication convention.

## Density includes source mass dependence

For the supported fixed force source, native `compute_forces_viscoelastic_calling_routine.F90:62–64` computes internal forces, `:89–100` adds the fixed source in the same acceleration/force accumulator, `:141–153` assembles MPI forces, and `:172–181` multiplies the whole result by the inverse mass. `compute_add_sources_viscoelastic.f90:82–88` adds the SH source without a density factor. `iterate_time.F90:605–607` writes seismograms then saves u and the complete post-mass acceleration. Thus `gr -= weight*(M^-1 bar_a)*saved_a` differentiates mass scaling of the **full** acceleration, including source acceleration. An additional separate source-density term would double-count this dependence. Replacing saved_a with stiffness-only acceleration would omit it. This conclusion is scoped to the current fixed force source and supported configuration.

## Pending acceptance evidence

Before accepting dense Marmousi derivatives, inspect the actual independent uninstrumented native baseline and matching extension traces, not just JSON flags. Require declared ranks=2, identical model/source/receivers/mesh/time/build/precision, raw full-grid gradients and raw native kernel files, nonzero dense-family Taylor/dot tests over all retained step sizes, and a nontrivial processed-objective VJP. At least one dense perturbation should vary within geological regions and across the mesh so that a four-region reduction cannot accidentally satisfy it. Use informative rho-only, Vs-only, and mixed directions and compare unprocessed gradients. A random data cotangent dot probe adds coverage beyond the objective's cotangent.

Plot acceptance separately from physical accuracy. Preserve native resolution/CFL diagnostics; complex model geometry does not establish continuum convergence. Report actual support, all-zero/near-zero gradient fractions, gradient dimensions and units, native frame provenance, coordinate extents and raw color scale. A complex true model and starting model, actual native wavefield and gathers, AD data cotangent, and actual full-grid gradients must be shown. One single shot can provide a valid derivative check but may poorly illuminate distant/deep regions.

Current decision: **pending runtime/raw-artifact evidence**. The source map above is a source-grounded contract, not a claim that the dense implementation or its run has passed.

## Scope correction: full P-SV with water is now the primary task

The user clarified that P-SV was expected throughout. The SH crop above is retained as a source audit of an abandoned primary route; it is not an acceptable substitute for the requested native coupled water/solid experiment. No Marmousi SH simulation had begun at this pivot. The current work must use actual planar P-SV coupled acoustic/elastic propagation and verify its actual transpose, including the water interface.

### Native coupled forward block and feasible exact reverse

Additional exact-pin sources inspected: `compute_coupling_acoustic_el.f90`, `compute_coupling_viscoelastic_ac.f90`, `compute_forces_viscoacoustic_calling_routine.F90`, `compute_forces_viscoelastic.F90`, `compute_kernels.f90`, `save_adjoint_kernels.f90`, `compute_vector_field.f90`, `compute_interpolated_dva.f90`, `enforce_acoustic_free_surface.f90`, `compute_add_sources_acoustic.f90`, `shared/read_material_table.f90`, `meshfem2D/save_databases.f90`, `specfem2D/read_materials.f90`.

For zero-init planar no-attenuation Newmark with no absorber and fixed geometry, native SIMULATION_TYPE=1 iterates acoustic force stage **before** elastic force stage (`iterate_time.F90:490–510`). Both predictor states have already been formed. Let B map elastic displacement to the acoustic RHS by native interface normal/line quadrature, D_A and D_E be inverse assembled mass matrices, and F_A/F_E be the negative symmetric native stiffness force operators. Ignoring the explicit free-surface projections only for this displayed block:

    a_A = D_A [ F_A(chi) + B u + f_A ]
    a_E = D_E [ F_E(u) - B^T a_A + f_E ]

This is source-grounded: `compute_coupling_acoustic_el.f90:263–269` adds `weight*(n_x*u_x+n_z*u_z)` to acoustic RHS. `compute_coupling_viscoelastic_ac.f90:88–102` chooses `pressure=-potential_dot_dot_acoustic` for forward simulation, and `:248–249` adds `weight*n*pressure` to elastic RHS. Both routines construct their normals/line weights on the **acoustic** element edge (`:193–225` of the first; `:171–203` of the second), with opposite point-order index maps as needed. Thus the sign is the explicit minus shown above; it should not be inferred by recomputing an outward solid normal.

A sound exact extension reverses elastic stage first, including Newmark corrector and data observation, forms w_E=D_E*bar_a_E, adds `-B*w_E` to the acoustic acceleration cotangent **before** reversing the acoustic mass stage, then w_A=D_A*bar_a_A and adds `B^T*w_A` to elastic displacement cotangent. Native F_E/F_A may supply their own symmetric interior transpose actions while edge transpose routines use the native index/normal/weight definitions. Full saved u and chi plus full accelerations are needed for active mass sensitivities; no temporal AD or parameter-FD production is necessary. This is a feasible derivation plan, not an implemented/accepted operator.

Acoustic free surface is an explicit projection of chi, chi_dot, chi_ddot to zero (`enforce_acoustic_free_surface.f90:74–89`), called **both before** acoustic force assembly and **after** its velocity corrector (`compute_forces_viscoacoustic_calling_routine.F90:52–53,225–226`). Each assignment has its own transpose zeroing at the reverse location. Merely using the forward acoustic force call without these transpose projections is insufficient. PML, Stacey, attenuation, forcing, moving sources, or periodic conditions need their own full state transposes if enabled; the above block does not certify them.

Native MPI acoustic calling routine explicitly warns (`:127–130`) that coupling interfaces can be missed when a domain exists only on another slice. Inspect partition/domain/interface ownership and compare total geometric interface edge measure with serial geometry. A two-rank native/extension waveform match alone will not reveal an interface omitted identically by both arms. Preserve actual parallel execution; do not silently make the coupled test serial to avoid it.

### P-SV coefficient VJP from the actual constitutive law

Actual elastic force computes `Vp=rho_vpstore/rho`, `lambda=rho*Vp²-2*mu` (`compute_forces_viscoelastic.F90:287–294`) and planar stresses (`:345–357`):

    sigma_xx = (lambda+2mu) ux,x + lambda uz,z
    sigma_zz = (lambda+2mu) uz,z + lambda ux,x
    sigma_xz = mu (ux,z+uz,x)

For local weight W=Jacobian*wx*wz and elastic test state w_E, coefficient derivatives of negative force are therefore:

    g_lambda = -W div(u) div(w_E)
    g_mu_at_fixed_lambda = -W [2 ux,x wx,x + 2 uz,z wz,z
                               + (ux,z+uz,x)(wx,z+wz,x)]
    g_rho_mass = -W w_E dot a_E_full

The dense physical coefficient map is rho=P rho_grid, Vp=P Vp_grid, Vs=P Vs_grid, lambda=rho(Vp²-2Vs²), mu=rho Vs². Before interpolation transpose:

    g_rho = g_rho_mass + (Vp²-2Vs²) g_lambda + Vs² g_mu
    g_Vp  = 2 rho Vp g_lambda
    g_Vs  = 2 rho Vs (g_mu-2g_lambda)

Then apply each exact P^T, subject to fixed-domain/mask parameterization. At active acoustic points, mu=0 and kappa=rho*Vp², acoustic stiffness depends on 1/rho and mass on 1/kappa. Their derivatives and any active source/sampling terms must be included; a Vs=0 water mask is fixed and should not be differentiated through a changing acoustic/elastic domain classifier.

### Why blindly reusing the upstream kernel files is unsafe

`compute_kernels_el.f90:192–209` uses a 1/3 deviatoric strain expression; `save_adjoint_kernels.f90:124–142` declares exported rho/mu/kappa values to be **log-coordinate kernel densities** under a volume integral and adds material factors, sign, dt, and kernel stride. They do not already carry SEM spatial quadrature, unlike the custom discrete gr/gm in the SH extension. They are different artifacts and must not be mixed.

More seriously, `save_adjoint_kernels.f90:147–166` explicitly carries TODO/warnings that Vp/Vs conversion is possibly false for plane strain. Planar setup stores kappa2D=rho*Vp²-mu, but alpha/beta conversion there uses 4/3*mu/kappa. Thus direct reuse of exported alpha/beta as exact planar physical gradients is unverified. Upstream kernel field presence is not a correctness gate. Deriving raw lambda/mu VJPs from the actual force constitutive code avoids this ambiguity, and multi-epsilon raw native checks must still pass. SIMULATION_TYPE=3 also changes coupling variables: acoustic coupling reads negative elastic acceleration while elastic coupling reads acoustic potential rather than its acceleration. It is a continuum-adjoint formulation, not a trivial sign/flag variation of the SH discrete reverse.

### Active-water source and observation traps

The earlier SH source-density conclusion does **not** transfer unchanged to acoustic sources. `compute_add_sources_acoustic.f90:65–74` adds `sourcearrays*stf/kappastore` to acoustic force **before** mass inversion. If source-domain kappa is active, its explicit 1/kappa derivative is additional to the full-acceleration mass derivative. Freezing water/source coefficients removes that parameter dependence only when the actual tested model map also freezes them.

Native acoustic displacement observation is `(1/rho)*grad(chi)` at every element GLL entry (`compute_vector_field.f90:143–205`), followed by receiver interpolation (`compute_interpolated_dva.f90:58–72`). Exact sampling reverse must transpose both gradient and interpolation, and active receiver-density requires the direct derivative of 1/rho. Pressure observation uses `-chi_ddot` and needs an acceleration cotangent at the exact observation position. An elastic displacement receiver uses the ordinary two-component interpolation transpose. An interface receiver's native element selection determines which formula applies; do not assume it belongs to solid by coordinate alone.

### Tomography domain input trap

Negative material code selects tomography, and setup rewrites MODEL='default' to 'tomo' when `tomo_material>0` (`setup_mesh.F90:553–556`). However the placeholder A in column 5 is **not unused**: `shared/read_material_table.f90:236–247` sets cs=A and initial acoustic/elastic flags from it, mesh writer saves that value (`meshfem2D/save_databases.f90:521–526`), and solver `read_materials.f90:400–404` uses positive A for elastic versus zero for acoustic. Current source supports one tomography-defined material; a coupled regular mesh can use fixed water material plus one solid tomography material. Use a positive elastic seed for the solid tomo region. Full-grid ASCII/GLL external assignment is another existing native input route if both material domains must vary, with its own explicit map/provenance. Do not put zero-seeded tomo values into solid elements and assume native later reclassifies them: `read_external_model` checks prevent that redefinition.

Primary P-SV decision remains **pending design, implementation, independent baseline and raw derivative evidence**. No claim of a completed coupled gradient follows from this feasibility audit.

### Independent preparation inspection (local, no simulation)

Loaded `data/prepared/marmousi2_full_psv.npz` with local Anaconda NumPy (system python3 had no NumPy; no installation was performed). All six physical fields are float64 `(78,200)`; water mask is bool `(78,200)`; native x spans 0..8955 and z 0..3465 at exact 45 m spacing. All six full-array first 67 native rows equal the previously prepared solid arrays exactly. Last 11 native rows (67..77) are water and retain Vp=1500, Vs=0, rho=2450 in both true and initial models. Full true Vp range is 1500..4699.999809 m/s; Vs range 0..2751.999855 m/s; initial ranges are 1500..4351.191074 and 0..2476.365198, respectively; rho is constant2450 throughout.

Inspected `examples/psv_marmousi2_full/CASE_PROVENANCE.json` and actual DATA inputs. They explicitly separate a fixed acoustic water material1 from positive-seed solid tomo material2, use 128x(48 solid+8 water) elements, source and two-component displacement receivers in solid at original depth540, physical interface450, and reflecting/free boundaries including acoustic pressure-free surface. Solid tomography border extension from first solid sample495 up to interface450 is explicitly recorded; this differs from bilinear interpolation across the paper grid's water/solid sample interval and must remain in the effective native model disclosure. The input is presently labeled proposed and does not claim accepted gradients.

### First P-SV exact-extension source review

Read initial `native/psv_adjoint.f90`, `scripts/patch_psv.py`, `scripts/build_psv.sh` after backend handoff. The extension is separate from the old SH vendor route, builds a clean official pinned MPI/double baseline before patching, and instruments only `iterate_time` begin/post-observation history hooks.

The elastic rho/lambda/mu contractions match the direct constitutive VJP above. The reverse calls unchanged native elastic and acoustic force kernels in their two element phases. It reuses native coupling kernels by setting global scratch arrays: `compute_coupling_acoustic_el` evaluates B*we and its negative is added to acoustic acceleration cotangent; `compute_coupling_viscoelastic_ac` with acoustic acceleration -wa evaluates B^T*wa. Each newly generated local force/coupling/source contribution is separately assembled before being added to an already assembled cotangent. Cartesian solid displacement receivers are explicitly required, avoiding unimplemented fluid observation/source-material derivatives in the active scope. Full acoustic state still propagates in forward/reverse; water coefficients are fixed.

The reverse applies the post-acoustic free-surface projection to all three acoustic cotangents, then the velocity corrector, mass, acoustic force transpose, solid-coupling transpose, then the pre-force projection of the acoustic potential cotangent before reversing predictors. In the advertised zero-init surface state the pre-force velocity cotangent has already been zeroed by the post-force projection, and the old acceleration accumulator is reset by the predictor; source ordering is consistent. The Fortran call that aliases the same zero scratch array in the two unused projection arguments is unnecessary but does not introduce a nonzero mathematical term in this case.

History writes actual native elastic displacement/acceleration and acoustic potential/acceleration, while reverse assumes exactly 6*nglob doubles per step. I requested that producer confirm/enforce actual array dimensions for every rank: absent-domain native arrays may be size1 rather than nglob. A fully coupled rank where both domains' arrays span nglob is consistent, but a generic pure-domain claim or a slice containing only one domain needs guarding or encoded domain sizes. No compiled/run acceptance yet.

Producer added per-rank coupled-domain and actual-array-size guards, closing this shape contract issue for the advertised coupled-only extension. A native B/B^T probe now evaluates both real native coupling actions on independent coordinate-defined fields, with separately assembled outputs and inverse interface-multiplicity state products. I recommended a constant-vertical-field action as an additional physical-interface completeness check: for this flat interface, negative unique-node sum of B*(0,1) should equal8955 m, and physical coupling edge count should equal128 unless an ownership duplication convention is explicitly accounted for.

Read `diffspecfem/psv.py`: production is one recorded native coupled forward, two-component `[receiver,component,time]` Torch data leaf, actual L2 or FIR/tanh/receiver-and-component-mixing/time-weighted objective VJP, one native reverse, and exact physical solid tomography transpose. File cotangent order `(time,component,receiver)` in C order matches Fortran `(receiver,component,time)` memory. Raw row parsing uses explicit elastic flags and every element-local occurrence. Its Vp/Vs/rho chain rule matches the lambda/mu formulas above; no illumination scaling or fitted factors are inserted. Active coordinates are strictly the67x200 solid grid for all three physical families; fixed water is outside that API. It currently requires positive lambda via Vp²>2Vs², which is a declared restricted material contract rather than a general stability criterion. Source-level route is credible; numerical evidence is still pending.

The producer reports that the small SCOTCH test has a rank with no local acoustic elements; the conservative coupled-on-every-rank guard was correctly rejected and retained as attempt2. Source was repaired to retain native size-one unused arrays and pad **only history** to the declared6*nglob format. I reviewed the new conditional assignments: when a domain is absent, its physical state is recorded as zero, reverse full-size temporaries are zeroed, native coupling calls are gated by local interface existence, and fresh full-size contributions are assembled with native MPI routines before accumulation. Local elastic coefficient loops remain restricted to elastic elements. This is consistent with the fixed layout and native local-domain allocation contract. It broadens rank partition handling without changing native propagation. Actual native kernels have zero local phase elements when a domain is absent; no-op behavior must still be confirmed by the small MPI runtime checks.

The producer also added sparse actual native **vector-displacement** snapshots via `compute_vector_one_element` for all elements. This correctly represents water as grad(chi)/rho rather than mixing scalar potential units with solid displacement units in a plot. The snapshot hook exports native GLL geometry and acoustic/elastic flags. Save/history/snapshot hooks retain raw units; plotting should use these arrays directly and disclose any display normalization.

Prepared independent audit utility `/private/tmp/audit_psv_native.py` to recompute saved ordinary and processed objective/data VJP in NumPy, independently reconstruct physical grid coefficient VJPs, compare native raw/cotangent files, check actual native baseline equality, sampling/coupling probes and interface length, sample native frames back to receiver data, and inspect MPI replicas. The NumPy convolution transpose passed its own independent dot sanity check exactly; a random processed-objective directional FD error decreased from3.63e-8 to3.64e-10 to6.59e-12 at h=.01,.001,.0001. These are audit-utility sanity checks only and do not validate any native solver result.

## Small native coupled P-SV acceptance: independently passed

Retrieved17MB of actual remote synthetic-unit artifacts from `/home/xiwang/DiffSPECFEM-PSV/evidence/psv_unit_attempt3` plus its public/synthetic9x9 model into `/private/tmp/psv-independent-unit-audit`. No native builds or simulations were run by this reviewer. Raw audit output is `/private/tmp/diffspecfem-psv-unit-raw-audit.json`.

The validator itself was independently inspected. Its sampling `lambda` functions only return previously emitted **actual native** forward/scatter action results to a generic inner-product helper: the inputs are real native coordinate-defined probe vf/y, forward sampling calls official `compute_vector_one_element` and `compute_interpolated_dva`, and transpose calls the same `psv_scatter` used by production. This is not the earlier tautological NumPy C/C^T test. Coupling probe likewise calls the actual two native edge operators. The separate geometric constant-field action checks native interface completeness. History frame sampling is another independent source/time/order gate.

Independent artifact recomputations:

- Actual original baseline `.semd` files, read in STATIONS/x,z order, are **bitwise equal** to extension data `(4,2,2400)`. Nonzero native norm0.07164243181; x/z component norms0.05538870672 and0.04543929141.
- Recomputed L2 value1.9795798073894655e-5 and cotangent agree to floating precision/exactly. Recomputed processed value0.04850697905066396 and independent convolution/tanh/mixing/time-weight transpose cotangent agree at relative1.586e-15 (max absolute1.832e-15). Archived binary cotangent agrees with the saved data path. This confirms real processed-objective AD reaches the native source.
- Native raw text arrays equal saved NPZ raw arrays exactly. There are2000 element-GLL rows:1600 elastic,400 acoustic. Acoustic coefficient derivatives are exactly zero, as required by fixed-water scope. Independent interpolation reconstruction matches native rho/lambda/mu at relative norms9.85e-17,6.64e-16,2.63e-16. Independent raw-to-physical-grid VJP matches gradient `(3,9,9)` at relative1.736e-16(L2) and1.406e-16(processed). No coefficient-family gradient is all zero; no grid entries are exactly zero in this synthetic test.
- Independent native sampling products differ by4.441e-16; coupling products differ by1.137e-13 on magnitude643.82, relative1.77e-16. Native constant-field coupling has8 edges and exactly2000m total interface width. This checks both transpose algebra and physical interface coverage for the small case.
- Re-sampling49 actual full-space native history frames gives extension data **bitwise exactly** at those steps. All six stored fields agree exactly at all37 repeated MPI coordinates in these frames, including the absent-acoustic-rank padded layout.
- Actual native vector snapshot headers confirm2 in-plane components,5x5GLL,2400steps. Rank0 water displacement norm is0.149727; both rank0 and rank1 solid x/z norms are nonzero. Water displacement was generated by the native gradient/potential conversion, not plotted from scalar chi with incompatible units.
- Independently recomputed every directional slope and positive Taylor remainder from the saved raw gradient, direction arrays and reported objective samples. Both objectives, all5 dense directions (Vp,Vs,rho,mixed,localized), all6 retained epsilons are informative and show observed order tending to2. L2 slopes span roughly7.0e-5..1.10e-3 in magnitude; processed slopes0.166..2.641, far above1e-9 absolute tolerance. Smaller h central errors improve proportionally to h². The helper requires two consecutive observable order1.8..2.2 intervals and two consecutive central matches, and rejects nearzero directions. This is not a pass caused solely by small absolute objective/gradient values.

**Decision:** the restricted exact coupled P-SV extension is numerically credible on the small native two-rank synthetic case. It is reasonable to start coordinated coarse/full Marmousi-II stages. This does **not** accept Marmousi-II gradients or extend active parameters to water, PML/Stacey, arbitrary receivers or unsupported settings. Those actual target artifacts still require review.

## Public evidence without redistributing source model arrays

Use an explicit release allowlist. Removing rho/lambda/mu columns from raw text is necessary but insufficient: `gradient_*.npz` contains the same model columns in its `raw` member; archived DATA tomography XYZ, prepared NPZ, native `Database*.bin`, and some VTK outputs can also contain the model. Physical direction arrays are particularly easy to overlook: `direction_vp/vs/mixed/localized.npy = initial_model * known_pattern`, and therefore can directly reveal initial model values. Publish the dimensionless direction formulas/patterns and physical-direction hashes, and regenerate physical directions after users retrieve source inputs themselves.

Publicly reproducible evidence can include:

- Complete synthetic-unit inputs and raw evidence, since those are authored fixture data.
- Public code, exact official source/build pin/hash/configuration, download URLs and source archive checksums, model preparation recipe and expected prepared-array hashes, without source-model arrays.
- Marmousi validation JSON, retained failed attempts/logs, geometry/probe outputs, receiver traces/residuals/cotangents, raw grid-gradient arrays, and coefficient-derivative rows with columns `[e,i,j,k,x,z,gr,g_lambda,g_mu,elastic_flag]` only.
- Figure scripts and derived plotted numerical evidence. Model panels are regenerated from users' own downloaded AGL inputs. An independent user can reconstruct material values from their regenerated model and native point coordinates, apply the physical lambda/mu chain and P^T to the published coefficient derivatives, and reproduce the dense gradient/verification plots without receiving base model columns from this repository.
- Separate internal full-precision archives for reviewer/private material-map checks and an exported public schema; publish hashes of internal native material columns rather than their values. Avoid broad archive copies that silently bring model DATA/databases back into the public tree.

## Coarse complete Marmousi-II P-SV acceptance: independently passed

Reviewed immutable remote `evidence/psv_marmousi64_attempt1` and exact `examples/marmousi2_solid_model.npz`. To avoid a slow large transfer, streamed the independent NumPy audit script to the already installed `/home/xiwang/anaconda3/envs/pytorch26/bin/python` with one numerical thread/nice10. It read completed artifacts and wrote JSON to stdout only; no remote file writes, native builds, native solver runs or changes to running fine tests. Local result: `/private/tmp/diffspecfem-psv-marmousi64-raw-audit.json`. A redundant partial bulk evidence transfer was canceled.

This is the complete original78x200 paper grid with fixed acoustic water, active three-family67x200 solid map and64x28 SEM elements,2 MPI ranks,41 Cartesian x/z displacement receivers,6s/4000steps/dt=.0015, f0=.5Hz and t0=2.4s. Delay scaling preserves f0*t0=1.2 relative to the fine1Hz/t0=1.2 case. Native guidance reported maxf0=.88954(initial)/.74085(true), maxDT=.002493/.002308, so this frequency/time step lie within native guidance. It is still not a mesh-convergence claim and is a separate case from the fine1Hz run.

Independent findings:

- Actual original native baseline `.semd` files are bitwise equal to extension `(41,2,4000)` data; nonzero native norm2.17874426937. This comparison uses the raw native text files, not only producer's saved baseline array.
- Independent L2 objective0.8985601253988494; processed objective1148.0269980311264. L2 cotangent equals independent residual exactly. Processed independent NumPy cotangent relative error2.847e-16/max absolute2.132e-14. Archived actual binary cotangent follows that output; raw native kernel text equals NPZ raw exactly.
- 44,800 native element-GLL rows are 38,400 solid / 6,400 water. Water raw derivatives are exactly zero. Independent physical rho/lambda/mu interpolation-map relative errors are 8.11e-17 / 6.35e-16 / 3.01e-16. Raw coefficient derivatives reduced independently to dense physical gradient `(3,67,200)` at relative 1.107e-16 (L2) and 1.109e-16 (processed). All three grid-gradient fields have no exactly-zero entries; their L2 norms are `[3.4868e-5,.00185157,.000127199]` and `[.0420941,2.16420,.152207]`, respectively.
- Native sampling dot absolute discrepancy1.776e-15; native coupling dot exactly equal at value-2350.77282655. Actual native constant-field probe gives64 coupling edges and exactly8955m full interface width. No missing physical interface is evident on this partition.
- 50 actual native frames (steps 1, 84, …, 3985, 4000) sample to recorded traces bitwise exactly. All six native history fields agree exactly at all 113 shared MPI coordinates in those frames.
- Both objectives pass all5 dense directions and all6 retained epsilons. Independent slopes equal reported slopes; L2 slopes are0.3925..8.7666, processed572.96..11177.18, so1e-9 absolute tolerance cannot create a vacuous pass. Independently recomputed remainder orders range1.9366..2.0011(L2) and1.9280..1.9998(processed), tending to2. At the final step the relative centered derivative errors lie1.68e-10..4.61e-8. No direction, epsilon, floor or scale was removed/fitted.

**Decision:** accepted for the declared complete Marmousi-II **coarse discrete coupled P-SV** experiment with fixed water and active solid Vp/Vs/rho. This is a genuine40200-coordinate solid gradient, not a regional four-value kernel. Fine128x56 waveform/gradient illustration, final visual rendering and public export remain separately pending.

Reviewed the current `package_psv_evidence.py` and `visualize_psv_marmousi.py` source. The exporter implements a model-data exclusion allowlist, including NPZ raw model columns and physical direction arrays. The visualizer uses element-local explicit GLL triangles, no cross-phase averaging, official native vector displacement, a fixed full-record signed-log wavefield scale, and raw physical grid gradients. Its Taylor plot explicitly consumes the full correctness case rather than implying fine-run multi-epsilon Taylor acceptance. Actual rendered figures still require visual QA. A model-panel sampling pitfall was sent to root: paper water samples through450 and first solid sample495 produce a472.5 pixel-edge boundary under naive nearest-cell plotting; native interface is450. Overlay the actual interface and disclose the495→450 solid border extension, or display actual native material coefficients.

## Fine Marmousi-II P-SV illustration: independently passed in declared scope

Read-only NumPy audit of completed remote `evidence/psv_marmousi128_illustration` used the same independently authored audit utility and exact prepared solid model. It wrote only local `/private/tmp/diffspecfem-psv-marmousi128-raw-audit.json` via stdout, with no remote writes or native reruns. This case uses 128x56 SEM elements, 2 ranks, 1 Hz source, dt=.00075 s, t0=1.2 s and 8000 steps / 6 s. Native source/time-step guidance passed separately. Its validation scope is **mixed and localized directions at two epsilons**, plus the final model-dot probe and physical/sampling/coupling/data gates; the full five-direction six-epsilon Taylor sweeps belong to the coarse case.

Independent findings:

- Actual original native baseline `.semd` files and saved reference are bitwise equal to extension `(41,2,8000)` traces, nonzero norm 1.94884414425. L2 and processed native traces are also bitwise identical to one another, so data processing changes only the actual objective/VJP.
- Independent objective values are 2.5861453123132727 (L2) and 4215.867109798609 (processed). L2 cotangent equals the residual exactly; processed independent NumPy VJP differs by relative 2.026e-16 / max absolute 2.842e-14. Actual binary cotangent follows the saved path.
- All 179,200 native element-GLL rows match NPZ raw exactly: 153,600 solid and 25,600 water. Fixed-water coefficient derivatives are exactly zero. Native rho/lambda/mu maps match independent interpolation at relative 8.31e-17 / 6.32e-16 / 2.94e-16. Raw-to-physical-grid gradient `(3,67,200)` agrees at relative 9.424e-17 (L2) and 1.143e-16 (processed). No advertised gradient family is zero or manufactured from regional values; each has no exactly-zero grid entries.
- Native sampling dot absolute discrepancy is 1.776e-15; native coupling discrepancy 4.547e-13 on magnitude 2350.77 (relative about 1.9e-16). Actual native geometric action gives 128 edges and exactly 8955 m full water/solid interface width.
- The actual frame schedule is **50 frames**, steps `[1,167,333,…,7969,8000]`. Exact schedule is stored in the audit JSON. Re-sampling these native history frames reproduces corresponding traces bitwise exactly. All six fields at 233 repeated MPI coordinates agree exactly.
- Independently rebuilt elementwise Lagrange differentiation matrices from actual rectangular GLL coordinates, applied them to saved native acoustic chi, divided by actual fixed water rho, and compared with official native vector snapshots. Relative water-displacement discrepancies are 5.376e-14 and 3.880e-14 on ranks 0 and 1; maxima are below 4.0e-15 m. Both ranks' actual water displacement norms are nonzero (1.4523, 1.4190). Thus water vector images have a verified physical origin and common displacement units with the solid.
- Independent gradient contractions match all reported fine mixed/localized dot RHS values exactly. At h=3e-4 / 1e-4, L2 mixed relative errors are 1.019e-6 / 1.133e-7, L2 localized 1.558e-7 / 1.731e-8; processed mixed 1.272e-6 / 1.413e-7, processed localized 1.774e-7 / 1.971e-8. Their approximately ninefold drop when h shrinks threefold is consistent with centered O(h²) truncation. These are not full fine-mesh Taylor sweeps and must not be labeled as such.
- The final mixed h=6e-5 native model-dot probe has relative errors 4.0774e-8 (L2) and 5.0873e-8 (processed), using independently reconstructed gradient RHS values 20.16570484156 and 32362.93673804. The public validation JSON SHA matches the accepted private source JSON; this final probe was checked against that immutable reported native FD action, without a reviewer solver rerun.

**Fine numerical decision:** accepted as the declared native coupled P-SV fine illustration, including actual water, full dense solid Vp/Vs/rho gradients and the stated checks. Combined with the coarse complete six-step family sweeps, this is credible evidence for the requested dense application. Fixed water, solid Cartesian displacement receivers, reflecting/free boundaries, no absorbers/attenuation and the documented solid boundary extension remain material scope limits. Public numerical export provenance and final rendered QA passed as detailed below.

Reviewer wording correction: an earlier message accidentally reused the unit's49-frame count when describing coarse evidence. The program always dynamically read the actual schedule; the coarse data have50 frames `[1,84,…,3985,4000]`, the fine data have50 frames `[1,167,…,7969,8000]`, and the unit has49. Audit JSONs now explicitly record actual coarse/fine frame counts and schedules. Numerical frame-sampling results were unaffected.

## Public numerical source identity: passed

Independently computed numeric SHA256 hashes from the accepted private fine native source arrays, then compared them against the completed local public bundle. All 16 checked arrays match byte for byte, including dtype and shape: gradient/data/cotangent/sample-time/sanitized coefficient derivatives for both objectives, and native step/displacement/geometry for both rank wavefields. The public source-validation SHA also matches the accepted private JSON (`dbab60641a8f1ebc04915f514b9ee1d395cce7b6e492bcb254496a847b6daf09`).

Both public vector bundles contain exactly 50 native frames and two physical displacement components. Rank 0 has `(50,2,90000)` values; rank 1 has `(50,2,89200)`. Their arrays equal the official native snapshot stream after only its declared serialization reshape. No display rescaling, averaging, interpolation, frame synthesis or model coefficient insertion occurs in the public data. Public coefficient derivative tables have `(179200,10)` entries and match precisely the accepted native subset that excludes rho/lambda/mu model columns. The public tree contains no physical direction arrays, tomography XYZ, native databases or model/prepared input files.

Machine-readable record: `evidence/psv_audits/fine_public_identity_audit.json`, alongside the three raw numerical audit JSONs. Numerical and public-source review are complete.

## Final public-tree and visual closure

Inspected the tracked plus unignored candidate public tree. No AGL source SEG-Y, prepared model arrays, Marmousi tomography XYZ, native material databases, physical model-scaled direction arrays or model-bearing raw kernel tables are included. All eight coarse/fine Marmousi NPZ files obey their public schemas; sanitized derivative tables have ten columns. The local full/prepared/solid-tomography inputs are Git-ignored. Record: `evidence/psv_audits/public_tree_audit.json`.

Root independently verified all 234 files across the three public manifests by byte size and SHA256. Root and the visual QA agent then completed the final fine rendering: five PDFs and the GIF rendered successfully; all PDF collision reports have zero failures and zero warnings, all alignment reports pass, minimum font sizes are 7 / 7 / 7 / 5.6 / 8 pt, and all 50 real GIF frames are present. Root visually inspected the fine wavefield and gradient panels; this reviewer previously inspected the full model-input panel. The final visual QA record is `figures/marmousi2/qa/native_visual_qa.json`, with `pending=[]`; its hash is recorded in the public-tree audit JSON.

**Release decision: yes for the delivered restricted coupled P-SV application and its explicitly separated verification scopes.** This is not acceptance of active-water gradients, absorbing/attenuating configurations, arbitrary observations, a mesh-convergence claim or an inversion result. No reviewer native rerun, implementation edit or source-model redistribution was used to obtain this decision.
