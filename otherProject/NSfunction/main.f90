program main
    use operator
    implicit none

    integer, parameter :: nx = 3, ny = 3
    real, parameter :: lx = 1, ly = 1
    real :: L(nx*ny, nx*ny)

    integer :: i, j

    call laplacian(L, lx, ly, nx, ny)

    write(*, '(9(f7.2, 1x))') ((L(i, j), j=1, nx*ny), i=1, nx*ny)

end program main
