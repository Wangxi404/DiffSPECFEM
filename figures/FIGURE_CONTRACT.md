# Visual evidence contract

The figures explain the already validated 32×32, two-rank, lossless SH case.
They make no new solver, full-field inversion, convergence or performance claim.
The scientific inputs are native data and coefficient kernels from the accepted
validation run and actual native displacement/acceleration histories whose
SHA256 values match its retained manifests. Python/matplotlib is the established
backend. Image generation is not used.

| Plate | Scientific question and bounded claim | Evidence roles |
|---|---|---|
| 01 Models | What physical contrast produces the target? Two heterogeneous regions receive the four declared relative log perturbations. | Reference/target control; difference isolates the actual density and Vs contrast. |
| 02 Wavefield | What does the native wave evolution look like? Actual saved post-step displacement propagates through the fixed case. | Four distinct native times show onset, transmission and later propagation; fixed geometry and amplitude scale allow comparison. |
| 03 Data and source | Which native data enter the objective and which source returns from AD? Native target/start traces produce the shown residual; the processed data cotangent differs from L2. | Data contrast, residual, L2 control, processed VJP, one waveform detail; all receivers and samples remain visible. |
| 04 Spatial gradients | What spatial information is reduced to four regional coordinates? The raw quadrature-weighted coefficient derivatives give the advertised log-coordinate vector by analytic sums. | Density/shear-modulus decomposition and chain-rule log-coordinate maps; numerical closure is checked. |
| 05 Taylor evidence | Does the gradient predict independently propagated perturbations? All declared epsilons show second-order remainders and progressively matching centered derivatives. | Two objectives and three parameter directions test the same bounded native gradient; thresholds and out-of-tolerance large steps stay visible. |

The archetype is an image plate with quantitative evidence for plates 01–04,
and a quantitative validation grid for plate 05. These are repository/report
figures, not a journal-specific submission package. Their physical sizes are
chosen for readable README/report display. Exports are 300 dpi PNG, editable
PDF and SVG; raster scientific fields are embedded, while text/axes remain
editable. The GIF uses a fixed scale and 44 native states at ten display frames
per second. Playback speed is illustrative; titles state the actual native index
and trace time.

No scientific rows, receiver samples or spatial nodes are discarded. Temporal
selection for the GIF/snapshot plate is explicit and retained in the source
manifest. MPI interface copies merge only after exact physical-coordinate
matching and bitwise equality of every selected displacement and acceleration
state. Rectilinear display axes are rounded to 12 decimal places in meters to
remove native element-arithmetic roundoff, changing coordinates by less than
10⁻¹² m and leaving the unique physical-node count unchanged. Field amplitudes
are never changed. The full 2 km square and reflecting boundaries remain shown.

The starting model is the heterogeneous two-layer reference. It is not a
Gaussian-blurred model. The four-coordinate interface does not support a new
arbitrary blurred full-field starting model. Spatial maps sum the 25 GLL
derivatives in each element; they are coefficient derivatives already including
quadrature/Jacobian, not continuous kernel densities or unweighted point values.
There is no smoothing, clipping, normalization or illumination correction.

Required QA: numerical assertions in the renderer, final physical plot-area
alignment at 1.5 pt, source validation, exported-PDF glyph-size audit,
exported-PDF collision audit, then visual inspection of each complete PNG.
The QA artifacts live under `figures/qa`; they are never substituted for the
scientific figures. The retained failed/coarse validation records remain part
of the numerical record and are described in the report.
