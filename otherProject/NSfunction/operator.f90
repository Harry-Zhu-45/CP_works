module operator
    implicit none
    
    contains

    subroutine laplacian(L, lx, ly, nx, ny)
        integer, intent(in) :: nx, ny
        real, intent(in) :: lx, ly
        real, intent(out) :: L(nx*ny, nx*ny)

        real :: dxi2, dyi2
        integer :: i, j, ii, jj

        dxi2 = (lx/nx)**(-2);dyi2 = (ly/ny)**(-2)

        L = 0.

        do j = 1, ny
            do i = 1, nx
                L((j-1)*nx+i, (j-1)*nx+i) = (-2)*(dxi2 + dyi2)
                do ii = i-1, i+1, 2
                    if (ii.ge.1.and.ii.le.nx) then
                        L((j-1)*nx+i, (j-1)*nx+ii) = dxi2
                    else
                        L((j-1)*nx+i, (j-1)*nx+i) = L((j-1)*nx+i, (j-1)*nx+i) + dxi2
                    end if
                end do

                do jj = j-1, j+1, 2
                    if (jj.ge.1.and.jj.le.ny) then
                        L((j-1)*nx+i, (jj-1)*nx+i) = dyi2
                    else
                        L((j-1)*nx+i, (j-1)*nx+i) = L((j-1)*nx+i, (j-1)*nx+i) + dyi2
                    end if
                end do

            end do
        end do

    end subroutine laplacian

end module operator
