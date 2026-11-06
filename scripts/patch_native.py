#!/usr/bin/env python3
"""Idempotently add native hooks; force/time stepping kernels remain unchanged."""
from pathlib import Path
import sys
root=Path(sys.argv[1]); here=Path(__file__).resolve().parents[1]
p=root/'src/specfem2D/iterate_time.F90'
s=p.read_text()
if 'module diffspecfem_discrete' in s: raise SystemExit('already patched')
s=s.replace('  subroutine iterate_time()', (here/'native/discrete_adjoint.f90').read_text()+'\n\n  subroutine iterate_time()',1)
s=s.replace('  use specfem_par_movie, only: MOVIE_SIMULATION','  use specfem_par_movie, only: MOVIE_SIMULATION\n  use diffspecfem_discrete',1)
s=s.replace('  ! local parameters','  logical :: ds_skip\n\n  ! local parameters',1)
s=s.replace('  ! count elapsed wall-clock time','  call ds_begin(ds_skip)\n  if (ds_skip) return\n\n  ! count elapsed wall-clock time',1)
s=s.replace('    call write_seismograms()','    call write_seismograms()\n    call ds_save()',1)
p.write_text(s)
