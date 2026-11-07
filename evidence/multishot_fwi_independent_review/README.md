# Independent full-shot review evidence

These reports describe independent source, numerical-array and control-flow checks. Their individual scope and limitations remain authoritative. Tiny analytic control-flow tests do not substitute for actual wave-solver derivative checks, and an accepted update does not certify completion or geological recovery.

Public JSON paths are logical identifiers. `<TEMP>` denotes an ephemeral diagnostic directory; `env:NAME/site-packages/...` retains only the environment and relative module path. They are not literal executable paths. Original private report bytes are retained locally under the Git-ignored `.cache/private-review/multishot_fwi_independent_review/`. Every public report records `original_private_report_sha256`; original numerical values and numerical-source SHA fields are unchanged. Portable scripts have their own current hash in `portable_public_reproducer`, while existing original-reader hash fields still identify the originally executed script.

The scripts infer their repository root from their installed location and accept `--repo-root`. Output defaults to a new system temporary directory; `--output` must name a new directory. The two loop tests also accept `--skill-root`, which must point to `skills/differentiate-seismic-solver` in a SeismicDiffSkill checkout. They use NumPy and CPU PyTorch; the policy test additionally uses psutil and a POSIX process-group API. No GPU is needed.

The path/CLI portability edits were checked by syntax parsing and `--help`; the historical numerical reports were not regenerated. The newly requested process-group policy check was actually rerun with harmless owned sleep sessions. No production solver, GPU task, numerical environment, or frozen numerical source was changed by publication cleanup.

From the repository root:

```sh
python evidence/multishot_fwi_independent_review/independent_native_full_loop.py --repo-root . --skill-root ../SeismicDiffSkill/skills/differentiate-seismic-solver
python evidence/multishot_fwi_independent_review/audit_native_actual_fwi.py --repo-root . --case psv_fwi_unit_20261108_final --case psv_fwi_roi_20261108
```

The first command replaces only the native forward/gradient with a small analytic CPU oracle and does not launch MPI or a native solver. The actual-array audit is read-only and requires the separately retained prepared inputs, targets, immutable shot caches and runtime files; these private inputs are not bundled here. Its default expected numerical driver SHA is the historical audited version; use `--expected-driver-sha256` only when deliberately auditing another explicitly identified version. Unit FD LHS/baseline archive limitations remain recorded; the audit does not fabricate unavailable original traces.

`audit_native_raw_fd.py --repo-root .` independently recomputes both sides from the separately retained twenty original native FD receiver arrays at the first coarse ROI anchor. `native_roi_raw_fd_audit.json` records this stronger raw-array check. It does not alter the historical unit's missing-trace limitation or certify later stages/models. This reader also accepts `--fd-run`, `--anchor-run`, `--prepared`, and `--epoch`.

The v2 controller review covers a smaller natural quasi-Newton direction, strict target-reuse guards and interrupted-run control regressions. `native_v2_initial_anchor_audit.json` independently compares every fresh initial shot array with the original anchor and binds the already retained raw FD proof only at this point. Run `audit_transition_bindings.py --native-root .` to repeat this read-only binding. Later iterates and fine-stage derivatives remain outside this initial-point scope.
