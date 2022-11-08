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
    ImagGridAndTailBasis(::Type{T}=Float64, nmin, nmax, wmax)
Polynomial tails of order `nmin` to `nmax` outside `[xmin, wmax]`, discrete basis inside.
- `n = 1 ~ ntail`: polynomial of 1/x ``(x0 / x)^{nmin + n - 1}`` where ``x0 = maximum(abs(grid))``
- `n = ntail+1 ~ 2*ntail`: polynomial of 1/x multiplied by sign ``sign(x) * (x0 / x)^{nmin + n - 1}``
- `n = 2*ntail+1 ~ end`: linear spline on `grid`

### particle_type == :Boson
- `w = 2π/β * m`
- Tails for `|m| >= wmax+1`, order-n tail is `((wmax + 1) / m)^n`
- explicit grid for `-wmax:wmax`, which corresponds to `w = 2π/β * (-wmax : wmax)`.

### particle_type == :Fermion
- `w = 2π/β * (m + 1/2)`
- Tails for `m >= wmax` or `m <= -wmax-1`, order-n tail is `((wmax + 1/2) / (m + 1/2))^n`
- explicit grid for `-wmax:wmax-1`, which corresponds to `v = 2π/β * (-wmax+1/2 : wmax-1/2)`.

"""
struct ImagGridAndTailBasis{T} <: Basis{T}
    particle_type::Symbol
    nmin::Int
    nmax::Int
    wmax::Int
    function ImagGridAndTailBasis(::Type{T}, particle_type, nmin, nmax, wmax) where {T}
        if particle_type !== :Boson && particle_type !== :Fermion
            error("Invalid particle type $particle_type. Must be :Boson or :Fermion.")
        end
        new{T}(particle_type, nmin, nmax, wmax)
    end
end
ImagGridAndTailBasis(particle_type, nmin, nmax, wmax) = ImagGridAndTailBasis(Float64, particle_type, nmin, nmax, wmax)

@inline ntails(f::ImagGridAndTailBasis) = (f.nmax - f.nmin + 1) * 2

function Base.getproperty(f::ImagGridAndTailBasis, s::Symbol)
    if s === :grid
        wmax = getfield(f, :wmax)
        getfield(f, :particle_type) === :Boson ? (-wmax:wmax) : (-wmax:(wmax-1))
    else
        getfield(f, s)
    end
end

@inline Base.axes(f::ImagGridAndTailBasis) = (InfRange(), 1:(ntails(f) + length(f.grid)))

# Customize printing
function Base.summary(io::IO, f::ImagGridAndTailBasis)
    print(io, typeof(f))
    print(io, "(particle_type=$(f.particle_type), nmin=$(f.nmin), nmax=$(f.nmax), ")
    print(io, "grid=$(f.grid), $(length(f.grid)) points)")
end

@inline function Base.getindex(f::ImagGridAndTailBasis{T}, x::Integer, n::Integer) where {T}
    x ∈ axes(f, 1) || throw(BoundsError())
    n ∈ axes(f, 2) || throw(BoundsError())
    if n <= ntails(f)  # Tail: polynomial of 1 / x
        if n <= div(ntails(f), 2)
            s = 1
            pow = n - 1 + f.nmin
        else  # For the latter half, multiply by sign(x)
            s = sign(x)
            pow = n - div(ntails(f), 2) - 1 + f.nmin
        end
        if f.particle_type === :Boson
            x ∈ f.grid ? zero(T) : T(((f.wmax + 1) / x)^pow) * s
        else
            x ∈ f.grid ? zero(T) : T(((f.wmax + 1/2) / (x + 1/2))^pow) * s
        end
    else  # Grid: f(x, n) = δ(x - first(f.grid), n - ntails - 1)
        x - first(f.grid) == n - ntails(f) - 1 ? one(T) : zero(T)
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
