using ContinuumArrays
using QuasiArrays: domain
using QuadGK

# Each basis set `f` should define the following functions:
# `Base.axes(f)`: domain of the basis, and the allowed indices of the basis functions
# `Base.getindex(f, x, n)`: value of the `n`-th basis function at `x`
# `support_domain(f, n)`: support of the `n`-th basis function

struct ConstantBasis{T} <: Basis{T}
end
ConstantBasis(::Type{T}=Float64) where {T} = ConstantBasis{T}()
Base.axes(::ConstantBasis) = (Inclusion(-Inf..Inf), 1:1)
Base.getindex(::ConstantBasis{T}, x::Number, n::Integer) where {T} = (n == 1 || throw(BoundsError()); one(T))
support_domain(::ConstantBasis, n::Integer) = (n == 1 || throw(BoundsError()); -Inf..Inf)


struct LinearSplineAndTailBasis{T, FT} <: Basis{T}
    nmin::Int
    nmax::Int
    grid::Vector{FT}
end
function LinearSplineAndTailBasis(::Type{T}, nmin, nmax, grid) where {T}
    LinearSplineAndTailBasis{T, eltype(grid)}(nmin, nmax, Vector(grid))
end
LinearSplineAndTailBasis(nmin, nmax, grid) = LinearSplineAndTailBasis(Float64, nmin, nmax, grid)

ntails(f::LinearSplineAndTailBasis) = f.nmax - f.nmin + 1

Base.axes(w::LinearSplineAndTailBasis) = (Inclusion(-Inf..Inf), 1:(w.nmax - w.nmin + 1 + length(w.grid)))

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

function Base.getindex(f::LinearSplineAndTailBasis{T}, x::Number, n::Integer) where {T}
    x ∈ axes(f, 1) || throw(BoundsError())
    n ∈ axes(f, 2) || throw(BoundsError())
    if n <= ntails(f)
        # Tail: polynomial of 1 / x
        if x ∈ support_domain(f, n)
            return 1 / x^(n - 1 + f.nmin)
        else
            return zero(T)
        end
    else
        # Spline interpolation
        if x ∈ support_domain(f, n)
            p = f.grid
            k = n - ntails(f)
            x == p[k] && return one(T)
            x < p[k] && return (x-p[k-1])/(p[k]-p[k-1])
            return (x-p[k+1])/(p[k]-p[k+1]) # x ≥ p[k]
        else
            return zero(T)
        end
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
