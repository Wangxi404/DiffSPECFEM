# Final rendered review

All five scientific plates were inspected as PNGs, including every panel and
the complete page. The final Taylor legend was moved above the panels after
the first PDF audit found gridlines crossing in-panel legend text. The final
PDFs have zero collision FAILs and zero WARNs. All five final plot-area alignment
reports pass at 1.5 pt; exported-PDF glyph minima exceed the 5 pt floor. Editable
PDF/SVG text and complete uncropped canvases are retained.

| Panel | Unique role checked | Summary/spread/unit | Labels, alignment and collisions | Pass |
|---|---|---|---|---|
| 01a | Starting density | Exact native regional coefficients; no replicate spread | kg/m³, full domain, shared ρ scale, aligned | Yes |
| 01b | Target density | Exact exponential parameter map | Same physical/color scales as 01a | Yes |
| 01c | Density contrast | Target minus starting | Symmetric zero-centered scale, physical coordinates | Yes |
| 01d | Starting Vs | Exact native regional coefficients | m/s, full domain, shared Vs scale, aligned | Yes |
| 01e | Target Vs | Exact exponential parameter map | Same physical/color scales as 01d | Yes |
| 01f | Vs contrast | Target minus starting | Symmetric zero-centered scale, physical coordinates | Yes |
| 02a | Early source state | Saved native n=200 displacement | Source/receivers/interface visible; common amplitude scale | Yes |
| 02b | Outward propagation | Saved native n=400 displacement | Correct post-step index and trace-label time | Yes |
| 02c | Boundary interaction | Saved native n=600 displacement | Complete domain retained; no amplitude normalization | Yes |
| 02d | Later native state | Saved native n=800 displacement | Same coordinates and amplitude scale as 02a–c | Yes |
| 03a | Target data | Four native traces × 800 samples | Shared displacement color scale, station ordering | Yes |
| 03b | Starting data | Four native traces × 800 samples | Same scale as 03a, duplicate receiver retained | Yes |
| 03c | Data mismatch | Exact starting minus target | All samples, symmetric residual scale | Yes |
| 03d | L2 control source | Actual saved cotangent, asserted equal to residual | Scaled m units, no implicit dt/reversal | Yes |
| 03e | Processed AD source | Actual PyTorch data VJP | m⁻¹ units, symmetric independent scale | Yes |
| 03f | Waveform detail | Native R04 starting/target | Full record, common displacement extent | Yes |
| 04a | Density sensitivity | Raw per-element sum of 25 GLL derivatives | m³/kg, quadrature caveat stated | Yes |
| 04b | Shear-modulus sensitivity | Raw per-element sum of 25 GLL derivatives | Pa⁻¹, own symmetric scale | Yes |
| 04c | Log-density map | Exact analytic ρ/μ chain rule | Dimensionless J, no smoothing or clipping | Yes |
| 04d | Log-Vs map | Exact analytic 2μ chain rule | Map-to-region closure checked; full domain | Yes |
| 05a | L2 remainder | All 8 epsilons × 3 deterministic directions | Exact positive residuals, ε² guide, external legend | Yes |
| 05b | L2 central error | All 8 epsilons × 3 deterministic directions | Actual absolute/relative thresholds; no points excluded | Yes |
| 05c | Processed remainder | All 8 epsilons × 3 deterministic directions | Exact positive residuals, shared family colors | Yes |
| 05d | Processed central error | All 8 epsilons × 3 deterministic directions | Thresholds and large-step misses stay visible | Yes |

Every displayed quantity is deterministic for its fixed case; no replicate
distribution or inferential uncertainty is claimed. Axes use physical
coordinates and units, and different objective units are disclosed. The blue/red
maps have white at zero; reference/target material maps use a common sequential
scale. No salience claim depends solely on categorical color differences.

Numerical rendering assertions establish: 1,024 distinct physical elements;
25 raw coefficient rows per element; exact original-native/reference trace
equality; exact duplicate receiver equality; bitwise matching MPI interface
states before merging; 16,641 unique physical nodes; correct declared native
times; exact receiver reconstruction for selected frames; and regional processed
gradient closure with maximum absolute difference 2.22×10⁻¹⁶.

The 44-frame GIF is 1,860×1,830 pixels, with a 100 ms display interval per frame.
Its color limits are identical to the snapshot plate. Native times appear in
the title because temporal extraction includes four additional snapshot times
and is therefore not a constant-time-step movie. All field values are native
output; temporal selection is fully disclosed and no spatial nodes are dropped.

The source validator's three advisory warnings were reviewed: no TIFF export,
300 dpi rather than the optional 600 dpi default, and report widths above common
journal column widths. These are deliberate repository/report deliverables;
editable PDF/SVG and 300 dpi PNG are the export contract. No journal submission
claim is made. All required source and final-PDF checks exit zero; the source
validator is run without promoting these declared export warnings to errors.

Portability follow-up: the unmodified, Apache-licensed alignment helper is now
bundled under `scripts/figure_qa`. Rendering with `NATURE_FIGURE_QA` unset and
`--skip-animation` passed all five alignment gates and produced byte-identical
PNGs for all five plates. The existing final PDF/SVG files and their completed
PDF audits were retained because geometry and scientific content did not change.
The bundled helper's Git blob is
`48f6f2b477b2b12cd4f83004d2e83aad9ac7e4b4`, matching the attributed upstream
revision, and the original Apache license was copied byte-for-byte.
