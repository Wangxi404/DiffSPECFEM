! DiffSPECFEM exact discrete SH/Newmark transpose extension.
! SPDX-License-Identifier: GPL-3.0-or-later
! This module calls the unchanged SPECFEM2D force assembly.
module diffspecfem_discrete
  use specfem_par
  implicit none
  character(16) :: ds_mode = ''
  integer :: ds_history_unit = 81
contains
  subroutine ds_begin(skip_forward)
    logical, intent(out) :: skip_forward
    integer :: s
    call get_environment_variable('DIFFSPECFEM_MODE',ds_mode,status=s)
    if (s /= 0) ds_mode = ''
    skip_forward = .false.
    if (len_trim(ds_mode) == 0) return
    if (ds_mode /= 'record' .and. ds_mode /= 'adjoint') stop 'invalid DIFFSPECFEM_MODE'
    if (NPROC /= 1 .or. GPU_MODE .or. P_SV .or. AXISYM) stop 'DiffSPECFEM supports serial CPU planar SH only'
    if (any_acoustic .or. any_poroelastic .or. any(ispec_is_anisotropic)) stop 'DiffSPECFEM requires isotropic elastic only'
    if (ATTENUATION_VISCOELASTIC .or. PML_BOUNDARY_CONDITIONS .or. STACEY_ABSORBING_CONDITIONS) &
      stop 'DiffSPECFEM supports reflecting/free boundaries without attenuation only'
    if (time_stepping_scheme /= 1 .or. SIMULATION_TYPE /= 1 .or. initialfield) stop 'DiffSPECFEM requires zero-init Newmark forward'
    if (NTSTEP_BETWEEN_OUTPUT_SAMPLE /= 1 .or. NSIGTYPE /= 1 .or. seismotypeVec(1) /= 1) &
      stop 'DiffSPECFEM requires full-rate displacement samples'
    if (SOURCE_IS_MOVING .or. NOISE_TOMOGRAPHY /= 0 .or. ADD_PERIODIC_CONDITIONS) stop 'unsupported source/boundary setting'
    if (ds_mode == 'adjoint') then
      call ds_reverse()
      skip_forward = .true.
    else
      open(ds_history_unit,file='OUTPUT_FILES/ds_history.bin',access='stream',form='unformatted',status='replace')
      write(ds_history_unit) nglob,NSTEP,nrecloc
      call ds_sampling_export()
      call ds_sampling_probe()
    endif
  end subroutine

  subroutine ds_sampling_export()
    integer :: r,i,j,e,k,u
    open(newunit=u,file='OUTPUT_FILES/ds_sampling.txt',status='replace')
    do r=1,nrecloc
      e=ispec_selected_rec_loc(r)
      do j=1,NGLLZ
        do i=1,NGLLX
          k=ibool(i,j,e)
          write(u,'(2i10,es25.16)') r,k,xir_store_loc(r,i)*gammar_store_loc(r,j)
        enddo
      enddo
    enddo
    close(u)
  end subroutine


  subroutine ds_scatter(buffer,samples)
    real(kind=CUSTOM_REAL),intent(inout) :: buffer(nglob)
    double precision,intent(in) :: samples(nrecloc)
    integer :: r,e,j,i,k
    do r=1,nrecloc
      e=ispec_selected_rec_loc(r)
      do j=1,NGLLZ
        do i=1,NGLLX
          k=ibool(i,j,e)
          buffer(k)=buffer(k)+samples(r)*xir_store_loc(r,i)*gammar_store_loc(r,j)
        enddo
      enddo
    enddo
  end subroutine

  subroutine ds_sampling_probe()
    real(kind=CUSTOM_REAL),allocatable :: vf(:,:),scatter(:)
    real(kind=CUSTOM_REAL) :: vec(NDIM,NGLLX,NGLLZ),pressure(NGLLX,NGLLZ),curl(NGLLX,NGLLZ)
    double precision,allocatable :: y(:),observed(:)
    double precision :: vz,vcurl
    integer :: k,r,e,u
    allocate(vf(NDIM,nglob),scatter(nglob),y(nrecloc),observed(nrecloc))
    vf=0;scatter=0;pressure=0;curl=0
    do k=1,nglob
      vf(1,k)=sin(.37d0*k)+.1d0*cos(.013d0*k)
    enddo
    do r=1,nrecloc
      y(r)=cos(.41d0*r)
      e=ispec_selected_rec_loc(r)
      call compute_vector_one_element(potential_acoustic,vf,displs_poroelastic,e,vec)
      call compute_interpolated_dva(r,e,vec,pressure,curl,observed(r),vz,vcurl,1)
    enddo
    call ds_scatter(scatter,y)
    open(newunit=u,file='OUTPUT_FILES/ds_sampling_native_nodes.txt',status='replace')
    do k=1,nglob
      write(u,'(i10,2es25.16)') k,vf(1,k),scatter(k)
    enddo
    close(u)
    open(newunit=u,file='OUTPUT_FILES/ds_sampling_native_receivers.txt',status='replace')
    do r=1,nrecloc
      write(u,'(i10,2es25.16)') r,y(r),observed(r)
    enddo
    close(u)
    deallocate(vf,scatter,y,observed)
  end subroutine

  subroutine ds_save()
    if (ds_mode /= 'record') return
    write(ds_history_unit) displ_elastic(1,:),accel_elastic(1,:)
    if (it == NSTEP) close(ds_history_unit)
  end subroutine

  subroutine ds_reverse()
    real(kind=CUSTOM_REAL),allocatable :: uh(:,:),ah(:,:),bu(:),bv(:),ba(:),w(:)
    real(kind=CUSTOM_REAL),allocatable :: vf(:,:),force(:,:),zeros(:,:)
    double precision,allocatable :: cot(:,:),gm(:,:,:),gr(:,:,:)
    double precision :: ux,uz,wx,wz,uxi,uga,wxi,wga,weight
    integer :: n,nn,ng,nr,u,e,i,j,k,l,iph
    allocate(uh(nglob,NSTEP),ah(nglob,NSTEP),bu(nglob),bv(nglob),ba(nglob),w(nglob))
    allocate(vf(NDIM,nglob),force(NDIM,nglob),zeros(NDIM,nglob),cot(nrecloc,NSTEP))
    allocate(gm(NGLLX,NGLLZ,nspec),gr(NGLLX,NGLLZ,nspec))
    open(newunit=u,file='OUTPUT_FILES/ds_history.bin',access='stream',form='unformatted',status='old')
    read(u) ng,nn,nr
    if (ng /= nglob .or. nn /= NSTEP .or. nr /= nrecloc) stop 'history shape mismatch'
    do n=1,NSTEP
      read(u) uh(:,n),ah(:,n)
    enddo
    close(u)
    open(newunit=u,file='DATA/ds_cotangent.bin',access='stream',form='unformatted',status='old')
    read(u) cot
    close(u)
    bu=0;bv=0;ba=0;zeros=0;gm=0;gr=0
    do n=NSTEP,1,-1
      ! Reverse corrector v_n = v_half + dt/2 a_n.
      ba = ba + deltatover2*bv
      w = rmass_inverse_elastic(1,:)*ba
      ! Reverse a_n=M^-1(-K u_n+f_n), including local coefficient VJPs.
      do e=1,nspec
        do j=1,NGLLZ
          do i=1,NGLLX
            k=ibool(i,j,e)
            uxi=0;uga=0;wxi=0;wga=0
            do l=1,NGLLX
              uxi=uxi+hprime_xx(i,l)*uh(ibool(l,j,e),n)
              wxi=wxi+hprime_xx(i,l)*w(ibool(l,j,e))
            enddo
            do l=1,NGLLZ
              uga=uga+hprime_zz(j,l)*uh(ibool(i,l,e),n)
              wga=wga+hprime_zz(j,l)*w(ibool(i,l,e))
            enddo
            ux=xix(i,j,e)*uxi+gammax(i,j,e)*uga
            uz=xiz(i,j,e)*uxi+gammaz(i,j,e)*uga
            wx=xix(i,j,e)*wxi+gammax(i,j,e)*wga
            wz=xiz(i,j,e)*wxi+gammaz(i,j,e)*wga
            weight=jacobian(i,j,e)*wxgll(i)*wzgll(j)
            gm(i,j,e)=gm(i,j,e)-weight*(ux*wx+uz*wz)
            gr(i,j,e)=gr(i,j,e)-weight*w(k)*ah(k,n)
          enddo
        enddo
      enddo
      vf=0;vf(1,:)=w;force=0
      do iph=1,2
        call compute_forces_viscoelastic(force,zeros,vf,zeros,dux_dxl_old,duz_dzl_old,dux_dzl_plus_duz_dxl_old, &
                                        .false.,e1,e11,e13,iph)
      enddo
      bu=bu+force(1,:)
      ! C transpose at the same post-step sample as native write_seismograms.
      call ds_scatter(bu,cot(:,n))
      ! Reverse predictor; initial u,v,a are fixed zero.
      ba=deltatsquareover2*bu+deltatover2*bv
      bv=bv+deltat*bu
    enddo
    open(newunit=u,file='OUTPUT_FILES/ds_raw_kernels.txt',status='replace')
    do e=1,nspec
      do j=1,NGLLZ
        do i=1,NGLLX
          k=ibool(i,j,e)
          write(u,'(4i10,6es25.16)') e,i,j,k,coord(1,k),coord(2,k), &
            rhostore(i,j,e),mustore(i,j,e),gr(i,j,e),gm(i,j,e)
        enddo
      enddo
    enddo
    close(u)
    open(newunit=u,file='OUTPUT_FILES/ds_route.txt',status='replace')
    write(u,*) 'native exact discrete adjoint; temporal AD=0; parameter FD=0; replay=0'
    write(u,*) 'steps,nglob,receivers,history_bytes=',NSTEP,nglob,nrecloc,2_8*nglob*NSTEP*8_8+12
    write(u,*) 'native force assembly calls=',2*NSTEP
    close(u)
    deallocate(uh,ah,bu,bv,ba,w,vf,force,zeros,cot,gm,gr)
  end subroutine
end module
