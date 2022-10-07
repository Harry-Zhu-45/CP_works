program main
    use operator
    use linear
    implicit none

    integer, parameter :: nx = 10, ny = 10, nt = 30
    real, parameter :: nu = 0.5, rho = 1, dt = 0.1, lx = 10., ly = 10., uB = 0., uT = 5., vL = 0., vR = 0.
    integer :: i, j, n, iter
    real :: dti, dxi, dx, dyi, dy, dxi2, dyi2, v_here, u_adv, u_shear, u_here, v_adv, v_shear, U_div,       &
            u(ny+2,nx+2), v(ny+2,nx+2), u_star(ny+2,nx+2) = 0., v_star(ny+2,nx+2) = 0., p(nx+2,ny+2) = 0.,  &
            r(ny*nx), L(ny*nx, ny*nx), p_prime(ny*nx), um(ny, nx), vm(ny, nx)

    ! dx, dy are the grid spacings
    dx = lx / nx
    dy = ly / ny

    dti = dt**(-1)
    dxi = dx**(-1)
    dyi = dy**(-1)
    dxi2 = dx**(-2)
    dyi2 = dy**(-2)

    ! randomize the initial velocity field
    call random_seed()
    call random_number(u)
    call random_number(v)

    ! set the velocity boundary conditions
    u(:, 2) = 0
    u(:, nx+2) = 0
    v(2, :) = 0
    v(ny+2, :) = 0

    u(1, :) = 2*uB - u(2, :)
    u(ny+2, :) = 2*uT - u(ny+1, :)
    v(:, 1) = 2*vL - v(:, 2)
    v(:, nx+2) = 2*vR - v(:, nx+1)

    ! iter means the number of iterations of the time-stepping loop
    do iter = 1, nt
        do j = 2, ny+1
            do i = 3, nx+1
                v_here = 0.25 * (v(j, i) + v(j+1, i) + v(j+1, i-1) + v(j, i-1))
                u_adv = u(j,i) * 0.5 * (u(j, i+1)-u(j, i-1)) * dxi + v_here *0.5 * (u(j+1, i)-u(j-1, i)) * dyi
                u_shear = nu * ((u(j, i-1)-2*u(j, i)+u(j, i+1)) * dxi2 + (u(j-1, i)-2*u(j, i)+u(j+1, i)) * dyi2)
                u_star(j, i) = u(j, i) + dt * (u_shear - u_adv)
            end do
        end do

        do j = 3, ny+1
            do i = 2, nx+1
                u_here = 0.25 * (u(j, i) + u(j, i+1) + u(j-1, i+1) + u(j-1, i))
                v_adv = u_here * 0.5 * (v(j, i+1)-v(j, i-1)) * dxi + v(j, i) * 0.5 * (v(j+1, i)-v(j-1, i)) * dyi
                v_shear = nu * ((v(j, i-1)-2*v(j, i)+v(j, i+1)) * dxi2 + (v(j-1, i)-2*v(j, i)+v(j+1, i)) * dyi2)
                v_star(j, i) = v(j, i) + dt * (v_shear - v_adv)
            end do
        end do

        write (*, "('u_star:'/5(f10.4, 1x))") ((u_star(i, j), j = 1, nx+2), i = 1, ny+2)
        write (*, "('v_star:'/5(f10.4, 1x))") ((v_star(i, j), j = 1, nx+2), i = 1, ny+2)

        n = 0
        do j = 2, ny+1
            do i = 2, nx+1
                n = n + 1
                r(n) = ((u_star(j, i+1) - u_star(j, i)) * dxi + (v_star(j+1, i) - v_star(j, i)) * dyi) * rho * dti
            end do
        end do

        write(*, "('r:'/f10.3, 1x)") r

        call laplacian(L, lx, ly, nx, ny)
        write(*, "('L:'/100(f5.1, 1x))") ((L(i, j), j = 1, nx*ny), i = 1, nx*ny)

        call jacobi(L, p_prime, r, ny*nx, 50)
        write(*, "('p_prime:'/f8.3)") p_prime

        n = 0
        do j = 2, ny+1
            do i = 2, nx+1
                n = n + 1
                p(j, i) = p_prime(n)
            end do
        end do

        write(*, "('p:'/10(f8.3, 1x))") ((p(i, j), j = 2, nx+1), i = 2, ny+1)

        open(10, file='./p.txt')
        write(10, '(10(f10.4, 1x))') ((p(i, j), j = 2, nx+1), i = 2, ny+1)

        do j = 2, ny+1
            do i = 3, nx+1
                u(j, i) = u_star(j, i) - dt / rho * (p(j, i) - p(j, i-1)) * dxi
            end do
        end do

        do j = 3, ny+1
            do i = 2, nx+1
                v(j, i) = v_star(j, i) - dt / rho * (p(j, i) - p(j-1, i)) * dyi
            end do
        end do

        ! constraint the velocity field to the boundary conditions
        u(1, :) = 2*uB - u(2, :)
        u(ny+2, :) = 2*uT - u(ny+1, :)
        v(:, 1) = 2*vL - v(:, 2)
        v(:, nx+2) = 2*vR - v(:, nx+1)

        write (*, "('u_n+1:',/,12(f10.4, 1x))") ((u(i, j), j = 1, nx+2), i = 1, ny+2)
        write (*, "('v_n+1:',/,12(f10.4, 1x))") ((v(i, j), j = 1, nx+2), i = 1, ny+2)

        do i = 1, nx
            um(:, i) = (u(2:ny+1, i+1) + u(2:ny+1, i+2)) * 0.5
        end do

        do j = 1, ny
            vm(j, :) = (v(j+1, 2:nx+1) + v(j+2, 2:nx+1)) * 0.5
        end do

        open(20, file='./u.txt')
        open(30, file='./v.txt')

        write(20, '(10(f15.3, 1x))') ((um(i, j), j = 1, nx), i = 1, ny)
        write(30, '(10(f15.3, 1x))') ((vm(i, j), j = 1, nx), i = 1, ny)

        do j = 2, ny+1
            do i = 2, nx+1
                U_div = (u(j, i+1)-u(j, i)) * dxi + (v(j+1, i)-v(j, i)) * dyi
                write(*, "('div_Un+1 at (',i2,', ',i2,'): ',f8.3)") j, i, U_div
            end do
        end do

        write(*, "('*********************************************')")
    end do

end program main
