#!/usr/bin/env python3
"""Apply independent P-SV recording/reverse hooks to clean official source.

SPDX-License-Identifier: GPL-3.0-or-later
The original time loop and all native numerical kernels remain unchanged.
"""
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1]);here=Path(__file__).resolve().parents[1]
path=root/'src/specfem2D/iterate_time.F90'
text=path.read_text()
if 'module diffspecfem_' in text:
    raise SystemExit('Requires clean official iterate_time.F90, not a previous extension')
replacements=[
    ('  subroutine iterate_time()', (here/'native/psv_adjoint.f90').read_text()+'\n\n  subroutine iterate_time()'),
    ('  use specfem_par_movie, only: MOVIE_SIMULATION','  use specfem_par_movie, only: MOVIE_SIMULATION\n  use diffspecfem_psv'),
    ('  ! local parameters','  logical :: psv_skip\n\n  ! local parameters'),
    ('  ! count elapsed wall-clock time','  call psv_begin(psv_skip)\n  if (psv_skip) return\n\n  ! count elapsed wall-clock time'),
    ('    call write_seismograms()','    call write_seismograms()\n    call psv_save()'),
]
for before,after in replacements:
    if text.count(before)!=1:
        raise RuntimeError(f'Pinned hook pattern not unique: {before!r}')
    text=text.replace(before,after,1)
path.write_text(text)
patch=subprocess.check_output(['git','-C',str(root),'diff','--','src/specfem2D/iterate_time.F90'])
(here/'native/psv_iterate_time.patch').write_bytes(patch)
