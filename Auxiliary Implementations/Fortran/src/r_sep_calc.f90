program vicsek_test
    real, parameter :: pi = 4.d0 * datan(1.d0)
    real, parameter :: L = 1.0, R_vic = 0.1, dt = 0.01, v_0 = 1.0, &
        t_max = 10 * dt
    integer, parameter :: N = 1000, D = 2

    real :: r(N, D), v(N, D), L_half, theta(N), t
    integer :: clock

    interface
        subroutine iterate(R_vic, L, dt, r, v)
            real, intent(in) :: R_vic, L, dt
            real, intent(inout) :: r(:, :), v(:, :)
        end subroutine iterate
    end interface

    L_half = L / 2.0

    call system_clock(count=clock)
    call random_seed(put=clock + 37 * (/ (i - 1, i = 1, n) /))

    call random_number(r)
    r = (r * L) - L_half

    call random_number(theta)
    theta = (theta * 2.0 * PI) - PI
    v(:, 1) = v_0 * cos(theta(:))
    v(:, 2) = v_0 * sin(theta(:))

    t = 0.0
    do while (t < t_max)
        call iterate(R_vic, L, dt, r, v)
        t = t + dt
!        write(*, *) ((sum(v(:, 1)) / N) ** 2 + (sum(v(:, 2)) / N) ** 2) / v_0
        write(*, *) v(1, 1), v(1, 2)
    end do

    end program vicsek_test

subroutine r_sep_calc(r, L, r_sep)
    implicit none
    real, intent(in) :: r(:, :), L
    real, intent(out) :: r_sep(:, :, :)

    integer :: N, D, i_1, i_2, i_dim
    real :: L_half

    N = size(r, 1)
    D = size(r, 2)
    L_half = L / 2.0

    do i_1 = 1, N
        do i_2 = 1, N
            r_sep(i_1, i_2, :) = r(i_2, :) - r(i_1, :)
            do i_dim = 1, D
                if (r_sep(i_1, i_2, i_dim) > L_half) then
                    r_sep(i_1, i_2, i_dim) = r_sep(i_1, i_2, i_dim) - L
                else if (r_sep(i_1, i_2, i_dim) .lt. -L_half) then
                    r_sep(i_1, i_2, i_dim) = r_sep(i_1, i_2, i_dim) + L
                end if
            end do
       end do
   end do
end subroutine r_sep_calc

subroutine collide(r_sep, R_c, v)
    implicit none

    real, intent(in) :: r_sep(:, :, :), R_c
    real, intent(inout) :: v(:, :)

    integer :: N, D, i_1, i_2
    real :: R_c_sq, R_sep_sq, v_1_dot_r_sep, v_2_dot_r_sep

    N = size(v, 1)
    D = size(v, 2)
    R_c_sq = R_c ** 2

    do i_1 = 1, N - 1
        do i_2 = i_1 + 1, N
            R_sep_sq = sum(r_sep(i_1, i_2, :) ** 2)
            if (R_sep_sq < R_c_sq) then
                v_1_dot_r_sep = sum(v(i_1, :) * r_sep(i_1, i_2, :))
                v_2_dot_r_sep = sum(v(i_2, :) * r_sep(i_2, i_1, :))
                if (v_1_dot_r_sep > 0.0) &
                    v(i_1, :) = v(i_1, :) - (2.0 * v_1_dot_r_sep * &
                        r_sep(i_1, i_2, :)) / R_sep_sq
                if (v_2_dot_r_sep > 0.0) &
                    v(i_2, :) = v(i_2, :) - (2.0 * v_2_dot_r_sep * &
                        r_sep(i_2, i_1, :)) / R_sep_sq
            end if
       end do
   end do
end subroutine collide

subroutine v_rho(r_sep, R, v_0, k, v)
    implicit none

    real, intent(in) :: r_sep(:, :, :), R, v_0, k
    real, intent(inout) :: v(:, :)

    integer :: i_1, i_2, n_neighbs
    real :: R_sq

    R_sq = R ** 2

    do i_1 = 1, size(v, 1)
        n_neighbs = 0
        do i_2 = 1, size(v, 1)
            if ((sum(r_sep(i_1, i_2, :) ** 2) < R_sq) .and. (i_1 /= i_2)) &
                n_neighbs = n_neighbs + 1
            v(i_1, :) = (v(i_1, :) / sqrt(sum(v(i_1, :) ** 2))) * &
                (v_0 * exp(-k * n_neighbs))
       end do
   end do
end subroutine v_rho

subroutine vicsek_align(r_sep, R, v)
    implicit none

    real, intent(in) :: r_sep(:, :, :), R
    real, intent(inout) :: v(:, :)

    integer :: i_1, i_2, n_neighbs
    real :: R_sq
    real, dimension(size(v, 1), size(v, 2)) :: v_temp

    R_sq = R ** 2
    v_temp = v

    do i_1 = 1, size(v, 1)
        n_neighbs = 1
        do i_2 = 1, size(v, 1)
            if ((sum(r_sep(i_1, i_2, :) ** 2) < R_sq) .and. (i_1 /= i_2)) &
                n_neighbs = n_neighbs + 1
                v(i_1, :) = v(i_1, :) + v_temp(i_2, :)
       end do
       v(i_1, :) = v(i_1, :) / n_neighbs
   end do
end subroutine vicsek_align

subroutine iterate_r(v, L, dt, r)
    implicit none

    real, intent(in) :: v(:, :), L, dt
    real, intent(inout) :: r(:, :)

    integer :: i, i_dim
    real :: L_half

    L_half = L / 2.0

    r = r + v * dt
    do i = 1, size(r, 1)
        do i_dim = 1, size(r, 2)
            if (r(i, i_dim) > +L_half) r(i, i_dim) = r(i, i_dim) - L
            if (r(i, i_dim) < -L_half) r(i, i_dim) = r(i, i_dim) + L
        end do
    end do
end subroutine iterate_r

subroutine iterate_v(r, R_vic, L, v)
    implicit none

    real, intent(in) :: r(:, :), R_vic, L
    real, intent(inout) :: v(:, :)

    real :: r_sep(size(r, 1), size(r, 1), size(r, 2))

    interface
        subroutine r_sep_calc(r, L, r_sep)
            real, intent(in) :: r(:, :), L
            real, intent(out) :: r_sep(:, :, :)
        end subroutine r_sep_calc
    end interface

    interface
        subroutine vicsek_align(v, r_sep, R)
            real, intent(in) :: r_sep(:, :, :), R
            real, intent(inout) :: v(:, :)
        end subroutine vicsek_align
    end interface

    call r_sep_calc(r, L, r_sep)
    call vicsek_align(v, r_sep, R_vic)
end subroutine iterate_v

subroutine iterate(R_vic, L, dt, r, v)
    implicit none

    real, intent(in) :: R_vic, L, dt
    real, intent(inout) :: r(:, :), v(:, :)

    interface
        subroutine iterate_v(r, R_vic, L, v)
            real, intent(in) :: r(:, :), R_vic, L
            real, intent(inout) :: v(:, :)
        end subroutine iterate_v
    end interface

    interface
        subroutine iterate_r(v, L, dt, r)
            real, intent(in) :: v(:, :), L, dt
            real, intent(inout) :: r(:, :)
        end subroutine iterate_r
    end interface

    call iterate_v(r, R_vic, L, v)
    call iterate_r(v, L, dt, r)
    end subroutine iterate
