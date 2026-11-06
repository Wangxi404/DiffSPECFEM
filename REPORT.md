# Validation report

The final serial and two-rank suites passed. The supported configuration is the fixed two-layer SH/Newmark case in README.md; this is not an all-physics SPECFEM2D differentiation claim.

| Ranks | Gate | L2 | FIR/tanh/mixed/weighted |
|---|---|---:|---:|
| 1 | Model-adjoint dot relative error | 6.440e-10 | 7.181e-10 |
| 1 | rho Taylor, all 8 epsilons | pass; order 2.0001–2.0081 | pass; order 2.0001–2.0077 |
| 1 | vs Taylor, all 8 epsilons | pass; order 2.0003–2.0082 | pass; order 2.0002–2.0047 |
| 1 | mixed Taylor, all 8 epsilons | pass; order 2.0002–2.0103 | pass; order 2.0002–2.0086 |
| 1 | Data-objective VJP Taylor | True | True |
| 2 | Model-adjoint dot relative error | 6.436e-10 | 7.183e-10 |
| 2 | rho Taylor, all 8 epsilons | pass; order 2.0001–2.0081 | pass; order 2.0001–2.0077 |
| 2 | vs Taylor, all 8 epsilons | pass; order 2.0003–2.0082 | pass; order 2.0002–2.0047 |
| 2 | mixed Taylor, all 8 epsilons | pass; order 2.0002–2.0103 | pass; order 2.0002–2.0086 |
| 2 | Data-objective VJP Taylor | True | True |

For each rank count the original native and extension traces are bitwise identical. The compiled native arbitrary-state sampling/scatter dot relative error is 1.214e-16. Reapplying native sampling to saved post-step history matches emitted traces with relative error 1.367e-16 (serial) and 1.393e-16 (two ranks). The two-rank source-assembly route also matched the original serial smoke traces at relative 1.672e-15 and the unprocessed 4-coordinate gradient at 9.032e-16.

The upstream source-resolution diagnostic recommends f0<=2.048 Hz for the coarse 8×8 mesh, while this test uses a 6 Hz Ricker source. Its high-frequency tail is underresolved. These results establish the gradient of the fixed discrete native solver, not continuum-converged seismogram accuracy.

The failed first Taylor sweep is retained in `evidence/validation-attempt1.json`: clean second-order Vs remainders were present, but the grid ended after only one central-match epsilon. Smaller steps were appended without dropping points or changing tolerances. The native adjoint did not need a fitted scale or parameter-gradient repair. Independent review strengthened the original algebra-only sampling probe with actual compiled native operators. The first history comparison had a test-only array extent error, recorded in `evidence/validation-attempt3.log` and repaired before acceptance.

Each production gradient executes one native forward and one exact native discrete adjoint, while all model finite differences are confined to the independent validation scripts. Native raw GLL density/shear-modulus coefficient gradients include SEM spatial quadrature and are converted analytically to the four advertised regional log coordinates. No temporal AD graph, smoothing, normalization or illumination correction supplies the production result.

Serial full-history storage is 13,939,212 bytes; two-rank storage totals 14,464,024 bytes because interface nodes are replicated. Both displacement and acceleration are saved at every post-step time, read in reverse, and no replay occurs. Full native source/build/runtime logs and numerical arrays are retained; heavy histories/build outputs are excluded from the Git repository and remain in the remote run/archive with checksums.

The complete attempt/intervention record is `application_record.json`. Actual hands-on development time was not measured. There was no no-skill control and the application is not author-blinded; no universal or causal adaptation-effort claim follows from these results.

Measured phase observations on the shared CPU host (not a speedup study):

| Ranks | Native forward | Extension, history off | Extension, full history | Native adjoint | Complete processed gradient |
|---|---:|---:|---:|---:|---:|
| 1 | 0.123 s | 0.138 s | 0.123 s | 0.117 s | 0.363 s |
| 2 | 0.489 s | 0.480 s | 0.498 s | 0.468 s | 1.490 s |

Serial native-adjoint maximum process RSS and two-rank independent process RSS peaks are recorded in the corresponding resource JSON/time files. The complete scope includes meshing, history writes, data objective/VJP, source conversion, native adjoint and material-gradient conversion; setup/build/imports are excluded. MPI startup is included. Per-rank RSS peaks are not simultaneous aggregate RAM.
