"""
``basis[v, n, w] = f_n(v - w/2)`` for n = 1, ..., nbasis/2
``basis[v, n, w] = f_{n - nbasis/2}(v + w/2)`` for n = nbasis/2 + 1, ..., nbasis
`nbasis = 2 * (ntails + length(grid))`

- `n = 1 ~ ntails/2`: ``(grid[end] / x)^{nmin+n-1}`` at x > grid[end] (right tail)
- `n = ntails/2 + 1 ~ ntails`: ``(grid[1] / x)^{nmin+n-1}`` at x < grid[1] (left tail)
- `n = ntails+1 ~ end`: linear spline on `grid`
"""
mutable struct ShiftedSplineBasis{T, FT} <: AbstractRealFreqBasis{T}
    nmin::Int
    nmax::Int
    grid::Vector{FT}
    w::FT
end
function ShiftedSplineBasis(::Type{T}, nmin, nmax, grid) where {T}
    ShiftedSplineBasis{T, eltype(grid)}(nmin, nmax, grid, zero(eltype(grid)))
end
ShiftedSplineBasis(nmin, nmax, grid) = ShiftedSplineBasis(Float64, nmin, nmax, grid)

ntails(f::ShiftedSplineBasis) = 2 * (f.nmax - f.nmin + 1)
nbasis(f::ShiftedSplineBasis) = 2 * (ntails(f) + length(f.grid))

@inline function support_bounds(f::ShiftedSplineBasis{T, FT}, n::Integer) where {T, FT}
    n ∈ axes(f, 2) || throw(BoundsError())
    n_ = mod1(n, div(nbasis(f), 2))
    shift = (n <= div(nbasis(f), 2)) ? 1 : -1
    wshift = shift * abs(f.w/2)
    if n_ <= div(ntails(f), 2)  # right tail
        if shift < 0
            nextfloat(f.grid[end] + wshift) .. prevfloat(f.grid[1] - wshift)
        else
            nextfloat(f.grid[end] + wshift) .. FT(Inf)
        end
    elseif n_ <= ntails(f)  # left tail
        if shift > 0
            nextfloat(f.grid[end] - wshift) .. prevfloat(f.grid[1] + wshift)
        else
            T(-Inf) .. prevfloat(f.grid[1] + wshift)
        end
    else
        k = n_ - ntails(f)
        (f.grid[max(1, k-1)] + wshift) .. (f.grid[min(end, k+1)] + wshift)
    end
end

@inline function Base.getindex(f::ShiftedSplineBasis{T}, x::Number, n::Integer) where {T}
    x ∈ axes(f, 1) || throw(BoundsError())
    n ∈ axes(f, 2) || throw(BoundsError())
    n_ = mod1(n, div(nbasis(f), 2))
    shift = (n <= div(nbasis(f), 2)) ? 1 : -1
    wshift = shift * abs(f.w/2)
    if x ∈ support_bounds(f, n)
        _linear_spline_and_tail(T, x - wshift, n_, f.nmin, f.nmax, f.grid)
    else
        zero(T)
    end
end

function get_fitting_points(f::ShiftedSplineBasis)
    coeffs_extrap = ntails(f) > 0 ? (1:(div(ntails(f), 2)+1)) : (1:0)
    # If the extrapolation grid points are too close to the interpolation grid points,
    # numerical instability can occur. So we multiply by 1 + sqrt(eps).
    left = prevfloat(f.grid[1]) * (1 + sqrt(eps(eltype(f))))
    right = nextfloat(f.grid[end]) * (1 + sqrt(eps(eltype(f))))
    points = vcat(left .* reverse(coeffs_extrap) .- abs(f.w/2), f.grid .- abs(f.w/2),
                  f.grid .+ abs(f.w/2), right .* coeffs_extrap .+ abs(f.w/2))

    x1 = f.grid[end] - abs(f.w/2) + sqrt(eps(eltype(f)))
    x2 = f.grid[1] + abs(f.w/2) - sqrt(eps(eltype(f)))
    if x1 < x2
        # The two grids do not overlap. Need to sample the place in between.
        append!(points, range(x1, x2, length=ntails(f)))
    end
    sort!(unique!(points))
end
