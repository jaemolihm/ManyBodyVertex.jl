using IntervalSets
using ContinuumArrays
using QuasiArrays: domain
using QuadGK

# Each basis set `f` should define the following functions:
# `nbasis(f)`: number of basis functions
# `Base.getindex(f, x, n)`: value of the `n`-th basis function at `x`
# `support_bounds(f, n)`: support of the `n`-th basis function

abstract type AbstractRealFreqBasis{T} <: Basis{T}; end
Base.axes(f::AbstractRealFreqBasis) = (Inclusion(-Inf..Inf), 1:nbasis(f))

struct ConstantBasis{T} <: AbstractRealFreqBasis{T}; end
ConstantBasis(::Type{T}=Float64) where {T} = ConstantBasis{T}()
ntails(::ConstantBasis) = 1
nbasis(::ConstantBasis) = 1
Base.getindex(::ConstantBasis{T}, x::Number, n::Integer) where {T} = (n == 1 || throw(BoundsError()); one(T))

@inline function support_bounds(f::ConstantBasis, n::Integer)
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    -Inf .. Inf
end


"""
- `n = 1 ~ ntails/2`: ``(grid[end] / x)^{nmin+n-1}`` at x > grid[end] (right tail)
- `n = ntails/2 + 1 ~ ntails`: ``(grid[1] / x)^{nmin+n-1}`` at x < grid[1] (left tail)
- `n = ntail + 1 ~ end`: linear spline on `grid`
"""
struct LinearSplineAndTailBasis{T, FT} <: AbstractRealFreqBasis{T}
    nmin::Int
    nmax::Int
    grid::Vector{FT}
end
function LinearSplineAndTailBasis(::Type{T}, nmin, nmax, grid) where {T}
    if nmin > nmax
        nmin, nmax = 1, 0  # no tails
    end
    LinearSplineAndTailBasis{T, eltype(grid)}(nmin, nmax, Vector(grid))
end
LinearSplineAndTailBasis(nmin, nmax, grid) = LinearSplineAndTailBasis(Float64, nmin, nmax, grid)

ntails(f::LinearSplineAndTailBasis) = 2 * (f.nmax - f.nmin + 1)
nbasis(f::LinearSplineAndTailBasis) = ntails(f) + length(f.grid)

"""
    integral_divergent(f::LinearSplineAndTailBasis, n::Integer)
Return true if the integral ∫dx f_n(x) is divergent.
"""
function integral_divergent(f::LinearSplineAndTailBasis, n::Integer)
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    if f.nmin <= 1
        ntails_over_2 = f.nmax - f.nmin + 1
        if n <= ntails_over_2
            p = n - 1 + f.nmin
            p <= 1 && return true
        elseif n <= ntails_over_2 * 2
            p = n - ntails_over_2 - 1 + f.nmin
            p <= 1 && return true
        end
    end
    return false
end

@inline function support_bounds(f::LinearSplineAndTailBasis{T, FT}, n::Integer) where {T, FT}
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    if n <= div(ntails(f), 2)  # right tail
        nextfloat(f.grid[end]) .. FT(Inf)
    elseif n <= ntails(f)  # left tail
        FT(-Inf) .. prevfloat(f.grid[1])
    else
        k = n - ntails(f)
        f.grid[max(1, k-1)] .. f.grid[min(end, k+1)]
    end
end

function _linear_spline_and_tail(::Type{T}, x, n, nmin, nmax, grid) where {T}
    ntails_over_2 = nmax - nmin + 1
    if n <= ntails_over_2  # right tail
        p = n - 1 + nmin
        return T((grid[end] / x)^p)
    elseif n <= ntails_over_2 * 2  # left tail
        p = n - ntails_over_2 - 1 + nmin
        return T((grid[1] / x)^p)
    else  # Spline interpolation
        k = n - ntails_over_2 * 2
        if x == grid[k]
            return one(T)
        elseif x < grid[k]
            k > 1 ? (x-grid[k-1])/(grid[k]-grid[k-1]) : one(T)
        else  # x > grid[k]
            k < length(grid) ? T((x-grid[k+1])/(grid[k]-grid[k+1])) : one(T)
        end
    end
end

@inline function Base.getindex(f::LinearSplineAndTailBasis{T}, x::Number, n::Integer) where {T}
    x ∈ axes(f, 1) || throw(BoundsError())
    n ∈ axes(f, 2) || throw(BoundsError())
    if x ∈ support_bounds(f, n)
        _linear_spline_and_tail(T, x, n, f.nmin, f.nmax, f.grid)
    else
        return zero(T)
    end
end

function ContinuumArrays.grid(f::LinearSplineAndTailBasis, length=11)
    vcat(range(2 * f.grid[1], prevfloat(f.grid[1]), length=length), f.grid, range(nextfloat(f.grid[end]), 2 * f.grid[end], length=length))
end

# Customize printing
function Base.summary(io::IO, f::LinearSplineAndTailBasis)
    print(io, typeof(f))
    print(io, "(nmin=$(f.nmin), nmax=$(f.nmax), ")
    print(io, "grid=$(f.grid[1])..$(f.grid[end]), $(length(f.grid)) points)")
end

"""
    get_fitting_points(basis::LinearSplineAndTailBasis)
Points to be used for fitting the basis coefficients.
"""
function get_fitting_points(basis::LinearSplineAndTailBasis)
    coeffs_extrap = ntails(basis) > 0 ? (1:(div(ntails(basis), 2)+1)) : (1:0)
    # If the extrapolation grid points are too close to the interpolation grid points,
    # numerical instability can occur. So we multiply by 1 + sqrt(eps).
    left = prevfloat(basis.grid[1]) * (1 + sqrt(eps(eltype(basis))))
    right = nextfloat(basis.grid[end]) * (1 + sqrt(eps(eltype(basis))))
    vcat(left .* reverse(coeffs_extrap), basis.grid, right .* coeffs_extrap)
end


"""
    basis_for_bubble(basis_v::LinearSplineAndTailBasis, basis_w::LinearSplineAndTailBasis)
For the bosonic frequency of the bubble, only interpolation can be done. So, the tail part
is added as additional coarse interpolation grids that spans up to `± w_ex_max`.

For the fermionic frequency, one uses (i) dense interpolation for the grids of `basis_v`,
(ii) coarse interpolation up to `± w_ex_max / 2 (+ buffer)`, and (iii) tails.

# Inputs
- `basis_v`: Fermionic frequency containing the interpolation grid and tail specificiation
for the bubble. Not the basis of the bubble - some coarse interpolation points may be added.
- `basis_w`: Bosonic frequency basis with which the bubble will be used. Not the basis of
the bubble - tails will be converted to coarse interpolation points.
"""
function basis_for_bubble(basis_v::LinearSplineAndTailBasis, basis_w::LinearSplineAndTailBasis)
    w_coarse_max = maximum(get_fitting_points(basis_w))
    w_coarse_num = ntails(basis_w)
    basis_for_bubble(basis_v, basis_w.grid, w_coarse_max, w_coarse_num)
end

"""
# Inputs
- `w_grid`: Grid to use for dense interpolation of `w`.
- `w_coarse_max`: Maximum `w` for coarse interpolation (that is used instead of tails).
- `w_coarse_num`: Number of points (for each side of the real axis) for coarse interpolation.
"""
function basis_for_bubble(basis_v::LinearSplineAndTailBasis, w_grid, w_coarse_max, w_coarse_num)
    v_coarse_max = w_coarse_max * 3 / 4  # 1/2 + buffer of 1/4
    v_coarse_step = maximum(diff(basis_v.grid)) * 2
    vs_bubble = vcat(
        range(basis_v.grid[1], -v_coarse_max, step=-v_coarse_step)[end:-1:2],
        basis_v.grid,
        range(basis_v.grid[end], v_coarse_max, step=v_coarse_step)[2:end],
    )
    basis_v_bubble = LinearSplineAndTailBasis(basis_v.nmin, basis_v.nmax, vs_bubble)

    ws_bubble = vcat(
        range(-w_coarse_max, minimum(w_grid), length=w_coarse_num+1)[1:end-1],
        sort(w_grid),
        range(maximum(w_grid), w_coarse_max, length=w_coarse_num+1)[2:end],
    )
    basis_w_bubble = LinearSplineAndTailBasis(1, 0, ws_bubble)

    basis_v_bubble, basis_w_bubble
end
