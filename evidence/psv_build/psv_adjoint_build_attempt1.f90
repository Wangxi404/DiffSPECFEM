! DiffSPECFEM exact coupled acoustic / P-SV Newmark transpose.
! SPDX-License-Identifier: GPL-3.0-or-later
! Uses unchanged official SPECFEM2D spatial force and fluid-solid kernels.
! Supported active parameters: elastic rho, lambda, mu only; water is fixed.
! Full native post-step history is streamed from disk in reverse, no replay.
module diffspecfem_psv
  use specfem_par
  implicit none
  character(16) :: psv_mode = ''
  integer :: psv_history_unit = 81
contains
  function psv_path(suffix) result(path)
    character(*),intent(in) :: suffix
    character(256) :: path
    character(6) :: rankstr
    write(rankstr,'(i6.6)') myrank
    path='OUTPUT_FILES/psv_rank'//rankstr//'_'//suffix
  end function

  subroutine psv_begin(skip_forward)
    logical,intent(out) :: skip_forward
    integer :: status,r,e,u,k
    call get_environment_variable('DIFFSPECFEM_PSV_MODE',psv_mode,status=status)
    if (status /= 0) psv_mode=''
    skip_forward=.false.
    if (len_trim(psv_mode) == 0) return
    if (psv_mode /= 'record' .and. psv_mode /= 'adjoint') stop 'invalid P-SV mode'
    if (GPU_MODE .or. .not. P_SV .or. AXISYM) stop 'P-SV extension requires CPU planar P-SV'
    if (any_poroelastic .or. any(ispec_is_anisotropic)) stop 'P-SV extension requires isotropic acoustic/elastic'
    if (.not. any_acoustic .or. .not. any_elastic .or. .not. coupled_acoustic_elastic) &
      stop 'P-SV extension requires a coupled acoustic/elastic partition on every rank'
    if (size(potential_acoustic) /= nglob .or. size(displ_elastic,2) /= nglob) &
      stop 'P-SV history requires full native acoustic/elastic state allocation'
    if (ATTENUATION_VISCOELASTIC .or. ATTENUATION_VISCOACOUSTIC .or. &
        PML_BOUNDARY_CONDITIONS .or. STACEY_ABSORBING_CONDITIONS) stop 'P-SV extension: no attenuation or absorbers'
    if (time_stepping_scheme /= 1 .or. SIMULATION_TYPE /= 1 .or. initialfield) stop 'P-SV requires zero-init Newmark'
    if (NTSTEP_BETWEEN_OUTPUT_SAMPLE /= 1 .or. NSIGTYPE /= 1 .or. seismotypeVec(1) /= 1) &
      stop 'P-SV requires full-rate displacement receivers'
    if (SOURCE_IS_MOVING .or. NOISE_TOMOGRAPHY /= 0 .or. ADD_PERIODIC_CONDITIONS .or. ACOUSTIC_FORCING) &
      stop 'P-SV unsupported source/boundary setting'
    if (abs(anglerec) > 1.d-14 .or. rec_normal_to_surface) stop 'P-SV requires Cartesian receiver axes'
    do r=1,nrecloc
      e=ispec_selected_rec_loc(r)
      if (.not. ispec_is_elastic(e)) stop 'P-SV differentiable receivers must be in elastic elements'
    enddo
    if (psv_mode == 'adjoint') then
      call psv_reverse()
      skip_forward=.true.
    else
      open(psv_history_unit,file=psv_path('history.bin'),access='stream',form='unformatted',status='replace')
      write(psv_history_unit) nglob,NSTEP,nrecloc
      call psv_sampling_export()
      call psv_sampling_probe()
      call psv_coupling_probe()
      open(newunit=u,file=psv_path('nodes.txt'),status='replace')
      do k=1,nglob
        write(u,'(i10,2es25.16)') k,coord(:,k)
      enddo
      close(u)
    endif
  end subroutine

  subroutine psv_coupling_probe()
    real(kind=CUSTOM_REAL),allocatable :: vf(:,:),q(:),bvf(:),btq(:,:),me(:,:),ma(:)
    integer :: k,u
    allocate(vf(NDIM,nglob),q(nglob),bvf(nglob),btq(NDIM,nglob),me(NDIM,nglob),ma(nglob))
    do k=1,nglob
      vf(1,k)=sin(.0037d0*coord(1,k))+.1d0*cos(.0013d0*coord(2,k))
      vf(2,k)=cos(.0017d0*coord(1,k))-.3d0*sin(.0023d0*coord(2,k))
      q(k)=.4d0+sin(.0021d0*coord(1,k))+.2d0*cos(.0011d0*coord(2,k))
    enddo
    displ_elastic=vf;potential_dot_dot_acoustic=0
    call compute_coupling_acoustic_el()
#ifdef WITH_MPI
    if (NPROC > 1) call assemble_MPI_scalar_ac_blocking(potential_dot_dot_acoustic)
#endif
    bvf=potential_dot_dot_acoustic
    potential_dot_dot_acoustic=-q;accel_elastic=0
    call compute_coupling_viscoelastic_ac()
#ifdef WITH_MPI
    if (NPROC > 1) call assemble_MPI_vector_el_blocking(accel_elastic)
#endif
    btq=accel_elastic
    ma=1;me=1
#ifdef WITH_MPI
    if (NPROC > 1) then
      call assemble_MPI_scalar_ac_blocking(ma)
      call assemble_MPI_vector_el_blocking(me)
    endif
#endif
    open(newunit=u,file=psv_path('coupling_probe.txt'),status='replace')
    do k=1,nglob
      write(u,'(i10,8es25.16)') k,q(k),bvf(k),vf(:,k),btq(:,k),ma(k),me(1,k)
    enddo
    close(u)
    ! psv_begin supports zero-initial-state forward only. Restore exactly.
    displ_elastic=0;accel_elastic=0;potential_dot_dot_acoustic=0
    deallocate(vf,q,bvf,btq,me,ma)
  end subroutine

  subroutine psv_sampling_export()
    integer :: r,i,j,e,k,u
    open(newunit=u,file=psv_path('sampling.txt'),status='replace')
    do r=1,nrecloc
      e=ispec_selected_rec_loc(r)
      do j=1,NGLLZ
        do i=1,NGLLX
          k=ibool(i,j,e)
          write(u,'(2i10,es25.16)') recloc(r),k,xir_store_loc(r,i)*gammar_store_loc(r,j)
        enddo
      enddo
    enddo
    close(u)
  end subroutine

  subroutine psv_scatter(buffer,samples)
    real(kind=CUSTOM_REAL),intent(inout) :: buffer(NDIM,nglob)
    double precision,intent(in) :: samples(nrecloc,NDIM)
    real(kind=CUSTOM_REAL),allocatable :: local_source(:,:)
    integer :: r,e,j,i,k
    allocate(local_source(NDIM,nglob));local_source=0
    do r=1,nrecloc
      e=ispec_selected_rec_loc(r)
      do j=1,NGLLZ
        do i=1,NGLLX
          k=ibool(i,j,e)
          local_source(:,k)=local_source(:,k)+samples(r,:)*xir_store_loc(r,i)*gammar_store_loc(r,j)
        enddo
      enddo
    enddo
#ifdef WITH_MPI
    if (NPROC > 1) call assemble_MPI_vector_el_blocking(local_source)
#endif
    buffer=buffer+local_source
    deallocate(local_source)
  end subroutine

  subroutine psv_sampling_probe()
    real(kind=CUSTOM_REAL),allocatable :: vf(:,:),scatter(:,:),multiplicity(:,:)
    real(kind=CUSTOM_REAL) :: vec(NDIM,NGLLX,NGLLZ),pressure(NGLLX,NGLLZ),curl(NGLLX,NGLLZ)
    double precision,allocatable :: y(:,:),observed(:,:)
    double precision :: vcurl
    integer :: k,r,e,u
    allocate(vf(NDIM,nglob),scatter(NDIM,nglob),y(nrecloc,NDIM),observed(nrecloc,NDIM),multiplicity(NDIM,nglob))
    scatter=0;pressure=0;curl=0
    do k=1,nglob
      vf(1,k)=sin(.0037d0*coord(1,k))+.1d0*cos(.0013d0*coord(2,k))
      vf(2,k)=cos(.0017d0*coord(1,k))-.3d0*sin(.0023d0*coord(2,k))
    enddo
    do r=1,nrecloc
      y(r,1)=cos(.41d0*recloc(r));y(r,2)=sin(.31d0*recloc(r))
      e=ispec_selected_rec_loc(r)
      call compute_vector_one_element(potential_acoustic,vf,displs_poroelastic,e,vec)
      call compute_interpolated_dva(r,e,vec,pressure,curl,observed(r,1),observed(r,2),vcurl,1)
    enddo
    call psv_scatter(scatter,y)
    multiplicity=1
#ifdef WITH_MPI
    if (NPROC > 1) call assemble_MPI_vector_el_blocking(multiplicity)
#endif
    open(newunit=u,file=psv_path('sampling_native_nodes.txt'),status='replace')
    do k=1,nglob
      write(u,'(i10,7es25.16)') k,vf(:,k),scatter(:,k),multiplicity(1,k),coord(:,k)
    enddo
    close(u)
    open(newunit=u,file=psv_path('sampling_native_receivers.txt'),status='replace')
    do r=1,nrecloc
      write(u,'(i10,4es25.16)') recloc(r),y(r,:),observed(r,:)
    enddo
    close(u)
    deallocate(vf,scatter,y,observed,multiplicity)
  end subroutine

  subroutine psv_save()
    if (psv_mode /= 'record') return
    write(psv_history_unit) displ_elastic,accel_elastic,potential_acoustic,potential_dot_dot_acoustic
    if (it == NSTEP) close(psv_history_unit)
  end subroutine

  subroutine psv_reverse()
    real(kind=CUSTOM_REAL),allocatable :: uh(:,:),ah(:,:),ch(:),cah(:)
    real(kind=CUSTOM_REAL),allocatable :: bu(:,:),bv(:,:),ba(:,:),we(:,:),fe(:,:),ze(:,:)
    real(kind=CUSTOM_REAL),allocatable :: bc(:),bcv(:),bca(:),wa(:),fa(:),za(:)
    double precision,allocatable :: cot(:,:,:),cot_global(:,:,:),gm(:,:,:),gl(:,:,:),gr(:,:,:)
    double precision :: duxi(2),duga(2),dwxi(2),dwga(2),dux(2),duz(2),dwx(2),dwz(2),weight
    integer :: n,nn,ng,nr,history_unit,cot_unit,u,e,i,j,k,r,l,iph
    integer(kind=8) :: position
    allocate(uh(NDIM,nglob),ah(NDIM,nglob),ch(nglob),cah(nglob))
    allocate(bu(NDIM,nglob),bv(NDIM,nglob),ba(NDIM,nglob),we(NDIM,nglob),fe(NDIM,nglob),ze(NDIM,nglob))
    allocate(bc(nglob),bcv(nglob),bca(nglob),wa(nglob),fa(nglob),za(nglob))
    allocate(cot(nrecloc,NDIM,NSTEP),cot_global(nrec,NDIM,NSTEP))
    allocate(gm(NGLLX,NGLLZ,nspec),gl(NGLLX,NGLLZ,nspec),gr(NGLLX,NGLLZ,nspec))
    open(newunit=history_unit,file=psv_path('history.bin'),access='stream',form='unformatted',status='old')
    read(history_unit) ng,nn,nr
    if (ng /= nglob .or. nn /= NSTEP .or. nr /= nrecloc) stop 'P-SV history shape mismatch'
    open(newunit=cot_unit,file='DATA/psv_cotangent.bin',access='stream',form='unformatted',status='old')
    read(cot_unit) cot_global
    close(cot_unit)
    do r=1,nrecloc
      cot(r,:,:)=cot_global(recloc(r),:,:)
    enddo
    bu=0;bv=0;ba=0;bc=0;bcv=0;bca=0;ze=0;za=0;gm=0;gl=0;gr=0
    do n=NSTEP,1,-1
      position=13_8+int(n-1,8)*6_8*int(nglob,8)*8_8
      read(history_unit,pos=position) uh,ah,ch,cah
      ! Elastic stage follows acoustic stage in the native forward.
      ! Reverse its velocity corrector, inverse mass, and unchanged stiffness.
      ba=ba+deltatover2*bv
      we=rmass_inverse_elastic*ba
      do e=1,nspec
        if (.not. ispec_is_elastic(e)) cycle
        do j=1,NGLLZ
          do i=1,NGLLX
            k=ibool(i,j,e)
            duxi=0;duga=0;dwxi=0;dwga=0
            do l=1,NGLLX
              duxi=duxi+hprime_xx(i,l)*uh(:,ibool(l,j,e))
              dwxi=dwxi+hprime_xx(i,l)*we(:,ibool(l,j,e))
            enddo
            do l=1,NGLLZ
              duga=duga+hprime_zz(j,l)*uh(:,ibool(i,l,e))
              dwga=dwga+hprime_zz(j,l)*we(:,ibool(i,l,e))
            enddo
            dux=xix(i,j,e)*duxi+gammax(i,j,e)*duga
            duz=xiz(i,j,e)*duxi+gammaz(i,j,e)*duga
            dwx=xix(i,j,e)*dwxi+gammax(i,j,e)*dwga
            dwz=xiz(i,j,e)*dwxi+gammaz(i,j,e)*dwga
            weight=jacobian(i,j,e)*wxgll(i)*wzgll(j)
            gl(i,j,e)=gl(i,j,e)-weight*(dux(1)+duz(2))*(dwx(1)+dwz(2))
            gm(i,j,e)=gm(i,j,e)-weight*(2*dux(1)*dwx(1)+2*duz(2)*dwz(2)+ &
                                      (duz(1)+dux(2))*(dwz(1)+dwx(2)))
            gr(i,j,e)=gr(i,j,e)-weight*sum(we(:,k)*ah(:,k))
          enddo
        enddo
      enddo
      fe=0
      do iph=1,2
        call compute_forces_viscoelastic(fe,ze,we,ze,dux_dxl_old,duz_dzl_old,dux_dzl_plus_duz_dxl_old, &
                                        .false.,e1,e11,e13,iph)
      enddo
#ifdef WITH_MPI
      if (NPROC > 1) call assemble_MPI_vector_el_blocking(fe)
#endif
      bu=bu+fe
      ! Elastic coupling is -B^T chi_ddot, whose transpose is -B we.
      displ_elastic=we;potential_dot_dot_acoustic=0
      if (coupled_acoustic_elastic) call compute_coupling_acoustic_el()
#ifdef WITH_MPI
      if (NPROC > 1) call assemble_MPI_scalar_ac_blocking(potential_dot_dot_acoustic)
#endif
      bca=bca-potential_dot_dot_acoustic
      ! Native post-acoustic-stage free-surface projection is self transpose.
      call enforce_acoustic_free_surface(bca,bcv,bc)
      bca=bca+deltatover2*bcv
      wa=rmass_inverse_acoustic*bca
      fa=0
      do iph=1,2
        call compute_forces_viscoacoustic(fa,za,wa,.false.,za,iph,e1_acous_sf,sum_forces_old)
      enddo
#ifdef WITH_MPI
      if (NPROC > 1) call assemble_MPI_scalar_ac_blocking(fa)
#endif
      bc=bc+fa
      ! Acoustic coupling is B u, whose transpose is B^T wa. Use the native
      ! -B^T kernel with -wa, retaining its actual normals/edge quadrature.
      potential_dot_dot_acoustic=-wa;accel_elastic=0
      if (coupled_acoustic_elastic) call compute_coupling_viscoelastic_ac()
#ifdef WITH_MPI
      if (NPROC > 1) call assemble_MPI_vector_el_blocking(accel_elastic)
#endif
      bu=bu+accel_elastic
      ! Native pre-acoustic-force projection, and C^T at post-step samples.
      call enforce_acoustic_free_surface(za,za,bc)
      call psv_scatter(bu,cot(:,:,n))
      ! Reverse the two independent predictors; initial states are fixed zero.
      ba=deltatsquareover2*bu+deltatover2*bv
      bv=bv+deltat*bu
      bca=deltatsquareover2*bc+deltatover2*bcv
      bcv=bcv+deltat*bc
    enddo
    close(history_unit)
    open(newunit=u,file=psv_path('raw_kernels.txt'),status='replace')
    do e=1,nspec
      do j=1,NGLLZ
        do i=1,NGLLX
          k=ibool(i,j,e)
          write(u,'(4i10,8es25.16,i10)') e,i,j,k,coord(:,k),rhostore(i,j,e), &
            kappastore(i,j,e)-mustore(i,j,e),mustore(i,j,e),gr(i,j,e),gl(i,j,e),gm(i,j,e), &
            merge(1,0,ispec_is_elastic(e))
        enddo
      enddo
    enddo
    close(u)
    open(newunit=u,file=psv_path('route.txt'),status='replace')
    write(u,*) 'coupled native P-SV discrete adjoint; temporal AD=0; parameter FD=0; replay=0'
    write(u,*) 'fixed acoustic coefficients; elastic rho lambda mu coefficient gradients'
    write(u,*) 'history policy: six native arrays/step on disk, reverse stream reads; O(nglob) live state'
    write(u,*) 'steps,nglob,receivers,history_bytes=',NSTEP,nglob,nrecloc,6_8*nglob*NSTEP*8_8+12
    write(u,*) 'native force assembly calls per domain=',2*NSTEP
    close(u)
    deallocate(uh,ah,ch,cah,bu,bv,ba,we,fe,ze,bc,bcv,bca,wa,fa,za,cot,cot_global,gm,gl,gr)
  end subroutine
end module
