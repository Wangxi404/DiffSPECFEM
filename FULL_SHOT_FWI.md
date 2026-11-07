# Native P–SV full-shot Vp inversion

Every iteration evaluates **all five independent shots at the same current
Vp**, accumulates their native model derivatives, then makes one accepted
update. The data objective is differentiated by PyTorch; the existing compiled
SPECFEM2D program supplies the propagation adjoint. Native numerical kernels
and two-rank execution are preserved.

This is a **new boxed Marmousi-II ROI**, introduced to make a multi-epoch
experiment tractable. Its reflecting boundaries and source/receiver geometry
are explicit experimental choices. It is not the old full-domain derivative
case and is not a matched performance comparison with the FD experiment.

| Item | Declared protocol |
| --- | --- |
| Domain | Original x = 8.015–10.490 km; surface-to-bottom depth = 2.025 km |
| Model coordinates | 35 × 56 solid input points; 45 m spacing; depth = 0.495–2.025 km |
| Unknown | Solid Vp; inherited sigma = 150 m blurred initial model |
| Fixed priors | Known true solid Vs and density; separate native water material fixed |
| Water | Top 450 m; native Vp = 1500 m/s, Vs = 0, density = 2450 kg/m³ |
| Sources | Five separate water sources, depth = 150 m |
| Receivers | 161 solid receivers at depth = 540 m; UX and UZ retained |
| Coarse stage | 2 Hz; 56 × (36 solid + 10 water) elements; 5000 × 0.5 ms; at most 8 accepted updates |
| Fine stage | 4 Hz; 112 × (72 solid + 20 water) elements; 10000 × 0.25 ms; at most 32 accepted updates |
| Objective | Per-shot mean squared normalized displacement residual, then global shot mean |
| Normalization | Observed RMS fixed per stage/shot; data-size factor once, shot-count factor once |
| Optimizer | Projected L-BFGS memory 5; full-shot Armijo; history reset at stage changes |
| Step limit | Actual accepted Vp change at most 60 m/s |
| Constraints | Positive isotropic bulk from fixed Vs; Vp at most 6000 m/s |
| Resources | MPI 2, one thread per rank, nice 10, low I/O priority; one temporary native history |

The optimizer's training NPZ **physically omits unknown true Vp**. Observation
preparation and offline true-model evaluation are separate actions. True-model
error cannot choose an epoch count, step length or stopping decision. The
strong known-Vs prior must remain visible when interpreting recovered detail.

## Verification before a long run

The native three-shot qualification uses actual original and extended
binaries, uneven grouping (2 + 1), data AD cotangents and separate native
adjoints. It checks the full objective's directional derivative, untouched
native traces, partial-cache resume and exactly one actual full-shot update.
Separate tiny tests attack optimizer control flow and cache provenance; they
are not substitutes for the native wave tests.

The new ROI's first five-shot coarse-model derivative has also been checked
from **all raw plus/minus receiver arrays**, independently of the controller's
reported scalars:

| First coarse anchor | Result |
| --- | --- |
| Independent shots / total data samples | 5 / 8,050,000 |
| Maximum physical directional steps | 0.10 / 0.05 m/s |
| Full-objective central-difference relative error | 8.0065e−8 / 2.0016e−8 |
| Waveform-JVP / AD-cotangent relative error | 1.4668e−8 / 3.6664e−9 |
| Original summed gradient vs per-shot reconstruction | Exact array equality before division by 5 |
| First accepted full-shot update | One update; 60 m/s maximum change; data loss decreases 49.6042% |

The [independent raw-array audit](evidence/multishot_fwi_independent_review/native_roi_raw_fd_audit.json)
retains source/target/array hashes and the declared scope. This qualifies the
first coarse anchor; subsequent stages and the completed inversion are
separate evidence.

For each new material model and stage, a short original-binary diagnostic
checks the solver's own suggested frequency/time-step limits before expensive
history recording. The diagnostic uses ten steps and restores the declared
record length. Both stages must pass on their actual mesh and material model;
planned mesh sizes alone are not numerical acceptance evidence.

The run binds observations, source/case files, adapter/runtime sources and
native binary hashes. Changed inputs reject cache reuse. Complete evaluations
have one fixed model fingerprint, all source IDs once, and one global mean.
Accepted checkpoints and model pointers support replay after interruption.
Native history is deleted after its verified contribution has been retained;
the fine-stage temporary history is substantial and is not a RAM-saving claim.

## Entry points and outputs

```bash
python scripts/invert_vponly_psv.py prepare --name MY_NEW_EXPERIMENT
python scripts/invert_vponly_psv.py observe --name MY_NEW_EXPERIMENT
python scripts/invert_vponly_psv.py invert --name MY_NEW_EXPERIMENT
```

Use a fresh namespace for changed physics or optimization settings. The
experiment also accepts an independently verified frozen skill packet; see
the entry point's help for the packet path/hash arguments.

| Artifact | Meaning |
| --- | --- |
| `ledger.jsonl` | Per-evaluation model/shot/reduction evidence and accepted update chain |
| `epochs/*/gradient/aggregate.npz` | Actual raw shot-sum derivative and declared shot-mean derivative |
| `epochs/*/direction.npz` | Raw derivative and optimizer search direction at that anchor |
| `epochs/*/accepted.json` | Full-shot accepted candidate, loss change and checkpoint pointer |
| `states/state_*.npz` | Accepted Vp and replay state |
| `state.json` | Actual progress and stopping status |

[Offline figures](FULL_SHOT_FIGURES.md) show physical models, raw derivatives,
directions, actual accepted updates and fit/recovery trajectories. An epoch
cap, interrupted run or small data residual does not establish convergence or
geological resolution.
