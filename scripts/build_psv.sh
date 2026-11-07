#!/usr/bin/env bash
set -euo pipefail
# SPDX-License-Identifier: GPL-3.0-or-later
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p vendor evidence/psv_build
pin=abf3196e3192af3343f2b402c82f34205577e672
if [[ ! -d vendor/specfem2d-psv/.git ]]; then
  git clone https://github.com/SPECFEM/specfem2d.git vendor/specfem2d-psv
fi
[[ -z "$(git -C vendor/specfem2d-psv status --porcelain --untracked-files=no)" ]] || { echo 'Requires clean official source'; exit 1; }
git -C vendor/specfem2d-psv checkout --detach "$pin"
(cd vendor/specfem2d-psv && nice -n 10 ./configure --with-mpi --enable-double-precision FC=gfortran MPIFC=mpif90 CC=gcc > ../../evidence/psv_build/configure.log 2>&1 && nice -n 10 make -j1 all > ../../evidence/psv_build/baseline_build.log 2>&1)
cp vendor/specfem2d-psv/bin/xspecfem2D psv-baseline-xspecfem2D
cp vendor/specfem2d-psv/bin/xmeshfem2D psv-baseline-xmeshfem2D
sha256sum psv-baseline-* > evidence/psv_build/baseline_sha256.txt
python3 scripts/patch_psv.py vendor/specfem2d-psv
(cd vendor/specfem2d-psv && nice -n 10 make -j1 all > ../../evidence/psv_build/extension_build.log 2>&1)
sha256sum vendor/specfem2d-psv/bin/xspecfem2D > evidence/psv_build/extension_sha256.txt
