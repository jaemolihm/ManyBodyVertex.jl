# FIXME: Allow odd tails
using Infinities: InfiniteCardinal
using ContinuumArrays

struct InfRange <: AbstractUnitRange{Int} end

Base.show(io::IO, ::InfRange) = print(io, "InfRange()")
Base.in(x::Real, ::InfRange) = false
Base.in(x::Integer, ::InfRange) = true
Base.length(::InfRange) = InfiniteCardinal{0}()
Base.checkindex(::Type{Bool}, ::InfRange, index::Integer) = true
Base.:(==)(x::InfRange, y::InfRange) = true
Base.:(==)(x::InfRange, y::AbstractUnitRange) = false
Base.:(==)(x::AbstractUnitRange, y::InfRange) = false
Base.:(==)(x::InfRange, y) = false
Base.:(==)(x, y::InfRange) = false


"""
    ImagConstantBasis(::Type{T}=Float64)
Constant imaginary-freqency basis.
"""
struct ImagConstantBasis{T} <: Basis{T} end
ImagConstantBasis(::Type{T}=Float64) where {T} = ImagConstantBasis{T}()

Base.axes(::ImagConstantBasis) = (InfRange(), 1:1)
Base.getindex(::ImagConstantBasis{T}, x::Integer, n::Integer) where {T} = (n == 1 || throw(BoundsError()); one(T))
ntails(::ImagConstantBasis) = 1


"""
    ImagGridAndTailBasis(::Type{T}=Float64, nmin, nmax, xmin, xmax)
Polynomial tails of order `nmin` to `nmax` outside `[xmin, xmax]`, discrete basis inside.
"""
struct ImagGridAndTailBasis{T} <: Basis{T}
    nmin::Int
    nmax::Int
    xmin::Int
    xmax::Int
    function ImagGridAndTailBasis(::Type{T}, nmin, nmax, xmin, xmax) where {T}
        new{T}(nmin, nmax, xmin, xmax)
    end
end
ImagGridAndTailBasis(nmin, nmax, xmin, xmax) = ImagGridAndTailBasis(Float64, nmin, nmax, xmin, xmax)

ntails(f::ImagGridAndTailBasis) = f.nmax - f.nmin + 1

Base.axes(w::ImagGridAndTailBasis) = (InfRange(), 1:(w.nmax - w.nmin + 1 + w.xmax - w.xmin + 1))

function Base.getindex(f::ImagGridAndTailBasis{T}, x::Integer, n::Integer) where {T}
    x ∈ axes(f, 1) || throw(BoundsError())
    n ∈ axes(f, 2) || throw(BoundsError())
    if n <= ntails(f)
        # Tail: polynomial of 1 / x
        if x < f.xmin || x > f.xmax
            T(1 / x^(n - 1 + f.nmin))
        else
            zero(T)
        end
    else
        # f(x, n) = δ(x - xmin + 1, n - ntails)
        x - f.xmin + 1 == n - ntails(f) ? one(T) : zero(T)
    end
end

# FIXME: Make this symmetric for both Fermionic and Bosonic bases.
function get_fitting_points(basis::ImagGridAndTailBasis)
    x_bound = max(-basis.xmin, basis.xmax)
    vcat(-x_bound .* (ntails(basis)+1:-1:2), basis.xmin:basis.xmax, x_bound .* (2:ntails(basis)+1))
end


"""
    frequency_index_bounds(nmax, particle_type)
The bounds for the index (that should be passed as `xmin` and `xmax` to ImagGridAndTailBasis)
so that frequency up to |v| <= nmax * 2π/β can be sampled for given `particle_type`
(`:Fermion` or `:Boson`).
"""
function frequency_index_bounds(nmax, particle_type)
    if particle_type === :Fermion
        ceil(Int, -nmax - 1/2), floor(Int, nmax - 1/2)
    elseif particle_type === :Boson
        -nmax, nmax
    else
        error("Wrong particle_type $particle_type")
    end
end
