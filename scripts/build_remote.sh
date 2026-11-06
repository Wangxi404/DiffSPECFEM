#!/usr/bin/env bash
set -euo pipefail
# Execute on the chosen Linux CPU host inside the repository.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GIT_TERMINAL_PROMPT=0
mkdir -p vendor evidence
pin=abf3196e3192af3343f2b402c82f34205577e672
if [[ ! -d vendor/specfem2d/.git ]]; then
  git clone https://github.com/SPECFEM/specfem2d.git vendor/specfem2d
fi
[[ -z "$(git -C vendor/specfem2d status --porcelain --untracked-files=no)" ]] || { echo 'upstream tree has changes; use fresh clone'; exit 1; }
git -C vendor/specfem2d checkout --detach "$pin"
(cd vendor/specfem2d && nice -n 10 ./configure --without-mpi --enable-double-precision FC=gfortran CC=gcc > ../../evidence/configure-reproduce.log 2>&1 && nice -n 10 make -j1 all > ../../evidence/build-baseline-reproduce.log 2>&1)
cp vendor/specfem2d/bin/xspecfem2D native-baseline-xspecfem2D
cp vendor/specfem2d/bin/xmeshfem2D native-baseline-xmeshfem2D
mkdir -p baseline/OUTPUT_FILES
cp -r examples/sh_two_layer/DATA baseline/
(cd baseline && nice -n 10 ../native-baseline-xmeshfem2D > ../evidence/baseline-mesh-reproduce.log 2>&1 && nice -n 10 ../native-baseline-xspecfem2D > ../evidence/baseline-forward-reproduce.log 2>&1)
python3 scripts/patch_native.py vendor/specfem2d
(cd vendor/specfem2d && nice -n 10 make -j1 all > ../../evidence/build-extension-reproduce.log 2>&1)
