subroutine extern_init(name,namelen) bind(C, name="extern_init")
    ! initialize the external runtime parameters in extern_probin_module
    !
    ! Binds to C function 'extern_init'

  use amrex_fort_module, only: rt => amrex_real

  implicit none

  integer, intent(in) :: namelen
  integer, intent(in) :: name(namelen)

  call runtime_init(name,namelen)

end subroutine extern_init
