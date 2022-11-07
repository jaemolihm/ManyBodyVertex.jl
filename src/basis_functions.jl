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
            return T(1 / x^(n - 1 + f.nmin))
        elseif n <= ntails(f)
            # Tail 2: polynomial of 1 / x times sign(x)
            n_ = n - div(ntails(f), 2)
            return T(sign(x) / x^(n_ - 1 + f.nmin))
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
Points to be used for fitting the basis coefficients
"""
function get_fitting_points(basis::LinearSplineAndTailBasis)
    coeffs_extrap = ntails(basis) > 0 ? (1:(div(ntails(basis), 2)+1)) : (1:0)
    left = prevfloat(basis.grid[1])
    right = nextfloat(basis.grid[end])
    vcat(left .* reverse(coeffs_extrap), basis.grid, right .* coeffs_extrap)
end
