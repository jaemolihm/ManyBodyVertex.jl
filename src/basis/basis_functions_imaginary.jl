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


abstract type AbstractImagBasis{T} <: Basis{T} end
Base.axes(f::AbstractImagBasis) = (InfRange(), 1:nbasis(f))

"""
    ImagConstantBasis(::Type{T}=Float64)
Constant imaginary-freqency basis.
"""
struct ImagConstantBasis{T} <: AbstractImagBasis{T} end
ImagConstantBasis(::Type{T}=Float64) where {T} = ImagConstantBasis{T}()
Base.getindex(::ImagConstantBasis{T}, x::Integer, n::Integer) where {T} = (n == 1 || throw(BoundsError()); one(T))
ntails(::ImagConstantBasis) = 1
nbasis(::ImagConstantBasis) = 1

@inline function support_bounds(f::ImagConstantBasis, n::Integer)
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    typemin(Int) .. typemax(Int)
end


"""
    ImagGridAndTailBasis(::Type{T}=Float64, nmin, nmax, wmax)
Polynomial tails of order `nmin` to `nmax` outside `[xmin, wmax]`, discrete basis inside.
- `n = 1 ~ ntail/2`: ``(x0 / x)^{nmin + n - 1}`` at x > grid[end] (right tail)
- `n = ntail/2+1 ~ ntail`: `(x0 / x)^{nmin + n - 1}`` at x < grid[1] (left tail)
- `n = ntail+1 ~ end`: linear spline on `grid`

### particle_type == :Boson
- `w = 2π/β * m`
- Tails for `|m| >= wmax+1`, order-n tail is `((wmax + 1) / m)^n`
- explicit grid for `-wmax:wmax`, which corresponds to `w = 2π/β * (-wmax : wmax)`.

### particle_type == :Fermion
- `w = 2π/β * (m + 1/2)`
- Tails for `m >= wmax` or `m <= -wmax-1`, order-n tail is `((wmax + 1/2) / (m + 1/2))^n`
- explicit grid for `-wmax:wmax-1`, which corresponds to `v = 2π/β * (-wmax+1/2 : wmax-1/2)`.

"""
struct ImagGridAndTailBasis{T} <: AbstractImagBasis{T}
    particle_type::Symbol
    nmin::Int
    nmax::Int
    wmax::Int
    function ImagGridAndTailBasis(::Type{T}, particle_type, nmin, nmax, wmax) where {T}
        if nmin > nmax
            nmin, nmax = 1, 0  # no tails
        end
        if particle_type !== :Boson && particle_type !== :Fermion
            error("Invalid particle type $particle_type. Must be :Boson or :Fermion.")
        end
        new{T}(particle_type, nmin, nmax, wmax)
    end
end
ImagGridAndTailBasis(particle_type, nmin, nmax, wmax) = ImagGridAndTailBasis(Float64, particle_type, nmin, nmax, wmax)

ntails(f::ImagGridAndTailBasis) = (f.nmax - f.nmin + 1) * 2
nbasis(f::ImagGridAndTailBasis) = ntails(f) + length(f.grid)

function Base.getproperty(f::ImagGridAndTailBasis, s::Symbol)
    if s === :grid
        wmax = getfield(f, :wmax)
        getfield(f, :particle_type) === :Boson ? (-wmax:wmax) : (-wmax:(wmax-1))
    else
        getfield(f, s)
    end
end

# Customize printing
function Base.summary(io::IO, f::ImagGridAndTailBasis)
    print(io, typeof(f))
    print(io, "(particle_type=$(f.particle_type), nmin=$(f.nmin), nmax=$(f.nmax), ")
    print(io, "grid=$(f.grid), $(length(f.grid)) points)")
end

"""
    integral_divergent(f::ImagGridAndTailBasis, n::Integer)
Return true if the integral ∑_x f_n(x) is divergent.
"""
function integral_divergent(f::ImagGridAndTailBasis, n::Integer)
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

@inline function support_bounds(f::ImagGridAndTailBasis, n::Integer)
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    if n <= div(ntails(f), 2)  # right tail
        last(f.grid)+1 .. typemax(f.wmax)
    elseif n <= ntails(f)  # left tail
        typemin(f.wmax) .. first(f.grid)-1
    else
        k = n - ntails(f)
        f.grid[k] .. f.grid[k]
    end
end

@inline function Base.getindex(f::ImagGridAndTailBasis{T}, x::Integer, n::Integer) where {T}
    x ∈ axes(f, 1) || throw(BoundsError())
    n ∈ axes(f, 2) || throw(BoundsError())
    if x ∈ support_bounds(f, n)
        if n <= ntails(f)  # Left and right tails
            pow = mod1(n, div(ntails(f), 2)) - 1 + f.nmin
            if f.particle_type === :Boson
                T((sign(x) * (f.wmax + 1) / x)^pow)
            else
                T((sign(x) * (f.wmax + 1/2) / (x + 1/2))^pow)
            end
        else  # Grid: 1 if x == f.grid[n - ntails]
            x == f.grid[n - ntails(f)] ? one(T) : zero(T)
        end
    else
        zero(T)
    end
end

# FIXME: Make this symmetric for both Fermionic and Bosonic bases.
function get_fitting_points(f::ImagGridAndTailBasis)
    coeffs_extrap = ntails(f) > 0 ? (1:(div(ntails(f), 2)+1)) : (1:0)
    if f.particle_type === :Fermion && f.wmax == 0  # Special case: only tails
        return vcat(.-reverse(coeffs_extrap), coeffs_extrap .- 1)
    end
    if f.particle_type === :Boson
        extrap_points = (f.wmax + 1) .* coeffs_extrap
        vcat(.-reverse(extrap_points), f.grid, extrap_points)
    else
        extrap_points = floor.(Int, (f.wmax + 1/2) .* coeffs_extrap .- 1/2)
        # v' = (n' + 1/2) = -v = -(n + 1/2) => n' = -n - 1
        vcat(.-reverse(extrap_points) .- 1, f.grid, extrap_points)
    end
end
