function basis_integral_imag(bases::Tuple)
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
            l > r && continue
            res, err = integrate_imag(f, l, r)
            overlap[inds] += res
        end
    end
    overlap
end

function integrate_imag(f::Function, l::Integer, r::Integer)
    # Constant chosen to get absolute error below 1E-14 for f(x) = 1/x^n (n >= 2).
    N = 2000
    if l == typemin(l) || r == typemax(r)
        # Cubic extrapolation of summation for x = 1/N to 1/4N and evaluate at x=0.
        l == typemin(l) && r < -N && error("extrapolation for r below -N not implemented, r=$r")
        r == typemax(r) && l < -N && error("extrapolation for l above +N not implemented, l=$l")
        x1 = 1 / N
        x2 = 1 / (2 * N)
        x3 = 1 / (3 * N)
        x4 = 1 / (4 * N)
        if l == typemin(l)
            y1 = sum(f, -N:r)
            y2 = y1 + sum(f, (-2N):(-N-1))
            y3 = y2 + sum(f, (-3N):(-2N-1))
            y4 = y3 + sum(f, (-4N):(-3N-1))
        else
            y1 = sum(f, l:N)
            y2 = y1 + sum(f, (N+1):2N)
            y3 = y2 + sum(f, (2N+1):3N)
            y4 = y3 + sum(f, (3N+1):4N)
        end
        res = if abs(y4 - y3) < eps(y4)
            y4
        else
            ( - y1 * x2 * x3 * x4 / (x1 - x2) / (x1 - x3) / (x1 - x4)
              - y2 * x3 * x4 * x1 / (x2 - x3) / (x2 - x4) / (x2 - x1)
              - y3 * x4 * x1 * x2 / (x3 - x4) / (x3 - x1) / (x3 - x2)
              - y4 * x1 * x2 * x3 / (x4 - x1) / (x4 - x2) / (x4 - x3) )
        end
        # For debugging
        # ff(x) = ( y1 * (x-x2) * (x-x3) * (x-x4) / (x1 - x2) / (x1 - x3) / (x1 - x4)
        #         + y2 * (x-x3) * (x-x4) * (x-x1) / (x2 - x3) / (x2 - x4) / (x2 - x1)
        #         + y3 * (x-x4) * (x-x1) * (x-x2) / (x3 - x4) / (x3 - x1) / (x3 - x2)
        #         + y4 * (x-x1) * (x-x2) * (x-x3) / (x4 - x1) / (x4 - x2) / (x4 - x3) )
    else
        res = sum(x -> f(x), l:r)
    end
    err = zero(res)  # Dummy value to match the return type of integrate_real
    res, err
end

@inline function support_bounds(f::ImagGridAndTailBasis, n::Integer)
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    if n <= ntails(f)
        first(f.grid)-1, last(f.grid)+1
    else
        f.grid[n - ntails(f)], f.grid[n - ntails(f)]
    end
end
@inline istail(f::ImagGridAndTailBasis, n::Integer) = n <= ntails(f)

@inline function integration_intervals_imag(bases, inds)
    lb_tail = typemax(Int)
    rb_tail = typemin(Int)
    lb_inte = typemin(Int)
    rb_inte = typemax(Int)
    has_tail = false
    for (b, i) in zip(bases, inds)
        b isa ImagConstantBasis && continue
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
        ((lb_inte, rb_inte), (0, -1))
    end
end
