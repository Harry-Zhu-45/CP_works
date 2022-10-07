module linear
    implicit none

    contains

    subroutine jacobi(A, x, b, n, k)
        implicit none
        integer, intent(in) :: n, k
        real, intent(in) :: A(n, n), b(n)
        real, intent(out) :: x(n)

        integer :: iter, i, j
        real :: s

        x = 0.

        do iter = 1, k
            do i = 1, n
                s = 0.
                do j = 1, n
                    if (i .ne. j) then
                        s = s + A(i, i)**(-1) * A(i, j) * x(j)
                    end if
                end do
                x(i) = -s + A(i, i)**(-1) * b(i)
            end do
        end do
    end subroutine jacobi
end module linear
