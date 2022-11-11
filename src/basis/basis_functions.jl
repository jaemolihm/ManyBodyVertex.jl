using ContinuumArrays
using QuasiArrays: domain
using QuadGK

# Each basis set `f` should define the following functions:
# `Base.axes(f)`: domain of the basis, and the allowed indices of the basis functions
# `Base.getindex(f, x, n)`: value of the `n`-th basis function at `x`
# `support_domain(f, n)`: support of the `n`-th basis function

# FIXME: multiply some constant tail to make its norm not so small.

struct ConstantBasis{T} <: Basis{T}
end
ConstantBasis(::Type{T}=Float64) where {T} = ConstantBasis{T}()
Base.axes(::ConstantBasis) = (Inclusion(-Inf..Inf), 1:1)
Base.getindex(::ConstantBasis{T}, x::Number, n::Integer) where {T} = (n == 1 || throw(BoundsError()); one(T))
support_domain(::ConstantBasis, n::Integer) = (n == 1 || throw(BoundsError()); -Inf..Inf)
ntails(f::ConstantBasis) = 1

"""
- `n = 1 ~ ntail`: polynomial of 1/x ``(x0 / x)^{nmin + n - 1}`` where ``x0 = maximum(abs(grid))``
- `n = ntail+1 ~ 2*ntail`: polynomial of 1/x multiplied by sign ``sign(x) * (x0 / x)^{nmin + n - 1}``
- `n = 2*ntail+1 ~ end`: linear spline on `grid`
"""
struct LinearSplineAndTailBasis{T, FT} <: Basis{T}
    nmin::Int
    nmax::Int
    grid::Vector{FT}
end
function LinearSplineAndTailBasis(::Type{T}, nmin, nmax, grid) where {T}
    LinearSplineAndTailBasis{T, eltype(grid)}(nmin, nmax, Vector(grid))
end
LinearSplineAndTailBasis(nmin, nmax, grid) = LinearSplineAndTailBasis(Float64, nmin, nmax, grid)

ntails(f::LinearSplineAndTailBasis) = 2 * (f.nmax - f.nmin + 1)

Base.axes(f::LinearSplineAndTailBasis) = (Inclusion(-Inf..Inf), 1:(ntails(f) + length(f.grid)))

"""
    support_domain(f::LinearSplineAndTailBasis, n::Integer)
Domain of nonzero value of the `n`-th basis function.
"""
@inline function support_domain(f::LinearSplineAndTailBasis, n::Integer)
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    if n <= ntails(f)
        domain(Inclusion(-Inf..prevfloat(f.grid[1])) ∪ Inclusion(nextfloat(f.grid[end])..Inf))
    else
        k = n - ntails(f)
        # domain(Inclusion(f.grid[max(1, k-1)] .. f.grid[min(end, k+1)]))
        f.grid[max(1, k-1)] .. f.grid[min(end, k+1)]
    end
end

@inline function Base.getindex(f::LinearSplineAndTailBasis{T}, x::Number, n::Integer) where {T}
    x ∈ axes(f, 1) || throw(BoundsError())
    n ∈ axes(f, 2) || throw(BoundsError())
    if x ∈ support_domain(f, n)
        if n <= div(ntails(f), 2)
            # Tail 1: polynomial of 1 / x
            x0 = max(abs(f.grid[1]), abs(f.grid[end]))
            return T((x0 / x)^(n - 1 + f.nmin))
        elseif n <= ntails(f)
            # Tail 2: polynomial of 1 / x times sign(x)
            n_ = n - div(ntails(f), 2)
            x0 = max(abs(f.grid[1]), abs(f.grid[end]))
            return T(sign(x) * (x0 / x)^(n_ - 1 + f.nmin))
        else
            # Spline interpolation
            p = f.grid
            k = n - ntails(f)
            x == p[k] && return one(T)
            x < p[k] && return (x-p[k-1])/(p[k]-p[k-1])
            return T((x-p[k+1])/(p[k]-p[k+1])) # x > p[k]
        end
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
