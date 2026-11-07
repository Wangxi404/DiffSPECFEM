# Full-shot inversion figures

The new inversion experiment is a **separate reflecting ROI**, with its own
source geometry, observation data and coarse/fine discretizations. It does not
replace the previous full-domain P–SV derivative experiments.

Every accepted iteration must use **all declared independent sources at the
same current model**, accumulate the full gradient, then make one accepted
update. Unknown true Vp is absent from the optimizer's training container.
Known Vs and density are fixed. True Vp may be read by the offline evaluation
program; it cannot select optimization steps or stopping decisions.

| Plate | Evidence |
| --- | --- |
| `model_recovery` | True, blurred initial and current physical Vp; initial/current errors and accumulated model change. Shared physical scales. |
| `gradient_direction_update` | All-shot raw derivatives at the first/last accepted-update anchors, optimizer direction, actual accepted Vp change. They remain distinct quantities. |
| `data_fit_and_model_error` | Full-shot data loss within each frequency stage and post-hoc Vp RMSE for every accepted model. |

After at least one accepted update:

```bash
python scripts/plot_multishot_fwi.py --format native \
  --run RUN_DIRECTORY --prepared PREPARED_EXPERIMENT_DIRECTORY \
  --truth-posthoc PREPARED_EXPERIMENT_DIRECTORY/truth_for_observations_only.npz \
  --output figures/fullshot
```

The offline reader needs no PyTorch. `requirements-fwi-visualization.txt`
records the tested Python 3.9 plotting stack; a separate plotting interpreter
can be used without changing the solver's numerical environment.

The reader checks model/observation provenance and accepted checkpoint hashes.
Native upward z coordinates are converted to depth by an exact row reversal;
the physical horizontal origin is retained. It reads every accepted model for
the trajectory. It does not run a wave solver or alter numerical fields.

The derivative plates display the **sum** of all shot contributions; the
optimizer uses the declared mean. Initial and latest raw derivatives share a
linear P99 scale, with saturation fractions recorded. Optimizer directions
and actual accepted updates use their own stated units and scales. There is
no gradient smoothing, depth gain or sharpening. Different frequency stages
have different objectives and native meshes, so data losses are normalized
within a stage and not connected across stage boundaries.

The gradient used for accepted update K belongs to model K−1. It is not the
gradient of the displayed current model K. Stage labels make this distinction
visible; a shared colour scale does not equate different-stage objectives.
Post-hoc spatial-derivative errors/cosines are retained alongside Vp RMSE.

Exports include editable PDF/SVG, 300 dpi PNG previews, a trajectory CSV and
hash/scale provenance. Preserve source preflight, 1.5 pt rendered alignment
checks, final-PDF glyph and collision audits, and inspect each exported panel.
These report-sized plates are not final journal production files. Label any
unfinished run with its actual accepted-update count and status; an epoch cap
does not establish convergence or resolved geological detail.
