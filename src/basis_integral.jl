"""
    basis_integral(bases...)
Compute overlap between basis functions.
``S_{i_1,..., i_N} = ∫dx basis1_{i_1}(x) * ... * basisN_{i_N}(x)``
"""
basis_integral(bases...) = basis_integral(bases)

function basis_integral(bases::Tuple)
    sizes = size.(bases, 2)
    overlap = zeros(eltype(bases[1]), sizes)
    for inds in CartesianIndices(sizes)
        # Divide the function support by intervals and integrate each interval.
        # Directly integrating over (-Inf, Inf) is inefficient, and may even become
        # inaccurate when the support is narrow.
        intervals = integration_intervals(bases, inds.I)

        f(x) = prod(getindex.(bases, x, inds.I))
        for interval in intervals
            l, r = interval
            l >= r && continue
            res, err = quadgk(f, l, r)
            overlap[inds] += res
        end
    end
    overlap
end

@inline function support_bounds(f::LinearSplineAndTailBasis, n::Integer)
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    if n <= ntails(f)
        prevfloat(f.grid[1]), nextfloat(f.grid[end])
    else
        k = n - ntails(f)
        f.grid[max(1, k-1)], f.grid[min(end, k+1)]
    end
end
@inline istail(f::LinearSplineAndTailBasis, n::Integer) = n <= ntails(f)

@inline function integration_intervals(bases, inds)
    T = eltype(first(bases))
    lb_tail = T(Inf)
    rb_tail = T(-Inf)
    lb_inte = T(-Inf)
    rb_inte = T(Inf)
    has_tail = false
    for (b, i) in zip(bases, inds)
        b isa ConstantBasis && continue
        lb, rb = support_bounds(b, i)
        if istail(b, i)
            has_tail = true
            lb_tail = min(lb_tail, lb)
            rb_tail = max(rb_tail, rb)
        else
            lb_inte = max(lb_inte, lb)
            rb_inte = min(rb_inte, rb)
        end
    end
    if has_tail
        # ---- lb_tail]  [rb_tail ----
        #    [lb_inte ---- rb_inte]
        ((lb_inte, lb_tail), (rb_tail, rb_inte))
    else
        #    [lb_inte ---- rb_inte]
        # Add dummy empty interval (zero(T), zero(T)) for type stability
        ((lb_inte, rb_inte), (zero(T), zero(T)))
    end
end
