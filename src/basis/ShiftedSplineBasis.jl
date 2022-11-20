# TODO: Add interpolation points or tails between the grids for large w
# TODO: imaginary frequency

"""
LinearSplineAndTailBasis with doubled grid ``vcat(grid - w/2, grid + w/2)``
# ``basis[v, n, w] = f_n(v - w/2)`` for n = 1, ..., nbasis/2
# ``basis[v, n, w] = f_{n - nbasis/2}(v + w/2)`` for n = nbasis/2 + 1, ..., nbasis
# `nbasis = 2 * (ntails + length(grid))`

# - `n = 1 ~ ntails/2`: ``(grid[end] / x)^{nmin+n-1}`` at x > grid[end] (right tail)
# - `n = ntails/2 + 1 ~ ntails`: ``(grid[1] / x)^{nmin+n-1}`` at x < grid[1] (left tail)
# - `n = ntails+1 ~ end`: linear spline on `grid`
"""
mutable struct ShiftedSplineBasis{T, FT} <: AbstractRealFreqBasis{T}
    const nmin::Int
    const nmax::Int
    const grid::Vector{FT}
    const basis::LinearSplineAndTailBasis{T, FT}
    w::FT
end
function ShiftedSplineBasis(::Type{T}, nmin, nmax, grid) where {T}
    w = zero(eltype(grid))
    grid_doubled = deduplicate_knots!(sort!(vcat(grid, grid)))
    basis = LinearSplineAndTailBasis(T, nmin, nmax, grid_doubled)
    ShiftedSplineBasis{T, eltype(grid)}(nmin, nmax, grid, basis, w)
end
ShiftedSplineBasis(nmin, nmax, grid) = ShiftedSplineBasis(Float64, nmin, nmax, grid)

set_basis_shift!(::Union{AbstractRealFreqBasis,AbstractImagBasis}, w) = w
function set_basis_shift!(f::ShiftedSplineBasis, w)
    f.w = w
    f.basis.grid .= vcat(f.grid .- f.w/2, f.grid .+ f.w/2)
    deduplicate_knots!(sort!(f.basis.grid))
    f
end

ntails(f::ShiftedSplineBasis) = 2 * (f.nmax - f.nmin + 1)
nbasis(f::ShiftedSplineBasis) = ntails(f) + 2 * length(f.grid)

@inline function support_bounds(f::ShiftedSplineBasis{T, FT}, n::Integer) where {T, FT}
    n ∈ axes(f, 2) || throw(BoundsError())
    support_bounds(f.basis, n)
end

@inline function Base.getindex(f::ShiftedSplineBasis{T}, x::Number, n::Integer) where {T}
    x ∈ axes(f, 1) || throw(BoundsError())
    n ∈ axes(f, 2) || throw(BoundsError())
    f.basis[x, n]
end

function get_fitting_points(f::ShiftedSplineBasis)
    points = get_fitting_points(f.basis)

    # Add points in the intermediate region if w is large so that the grids centered at
    # +w/2 and -w/2 do not overlap.
    x1 = f.grid[end] - abs(f.w/2) + sqrt(eps(f.w))
    x2 = f.grid[1] + abs(f.w/2) - sqrt(eps(f.w))
    if x1 < x2
        append!(points, range(x1, x2, length=5))
    end
    deduplicate_knots!(sort!(points))
end
