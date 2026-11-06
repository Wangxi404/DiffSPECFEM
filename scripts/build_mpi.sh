#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p vendor evidence
pin=abf3196e3192af3343f2b402c82f34205577e672
if [[ ! -d vendor/specfem2d-mpi/.git ]]; then
  git clone https://github.com/SPECFEM/specfem2d.git vendor/specfem2d-mpi
fi
[[ -z "$(git -C vendor/specfem2d-mpi status --porcelain --untracked-files=no)" ]] || { echo 'upstream tree has changes; use fresh clone'; exit 1; }
git -C vendor/specfem2d-mpi checkout --detach "$pin"
(cd vendor/specfem2d-mpi && nice -n 10 ./configure --with-mpi --enable-double-precision FC=gfortran MPIFC=mpif90 CC=gcc > ../../evidence/configure-mpi-reproduce.log 2>&1 && nice -n 10 make -j1 all > ../../evidence/build-mpi-baseline-reproduce.log 2>&1)
cp vendor/specfem2d-mpi/bin/xspecfem2D native-baseline-mpi-xspecfem2D
cp vendor/specfem2d-mpi/bin/xmeshfem2D native-baseline-mpi-xmeshfem2D
python3 scripts/patch_native.py vendor/specfem2d-mpi
(cd vendor/specfem2d-mpi && nice -n 10 make -j1 all > ../../evidence/build-mpi-extension-reproduce.log 2>&1)
