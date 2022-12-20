function basis_integral_imag(bases::Tuple; skip_divergence=false)
    skip_divergence && length(bases) > 1 && error("skip_divergence is allowed only for a single basis")
    sizes = size.(bases, 2)
    overlap = zeros(eltype(bases[1]), sizes)
    for inds in CartesianIndices(sizes)
        skip_divergence && integral_divergent(bases[1], inds[1]) && continue
        # Integrate the function only on the interval where the function is nonzero.
        # Directly integrating over (-Inf, Inf) is inefficient, and may even become
        # inaccurate when the support is narrow.
        interval = integration_interval(bases, inds.I)
        isempty(interval) && continue

        f(x) = prod(getindex.(bases, x, inds.I))
        l, r = interval.left, interval.right

        res = integrate_imag(f, l, r)
        res !== nothing && (overlap[inds] += res.val)
    end
    overlap
end

function integrate_imag(f::Function, l::Integer, r::Integer)
    l > r && return nothing
    # Constant chosen to get absolute error below 2E-14 for f(x) = 1/x^n (n >= 2).
    N = 1200
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
        val = if norm(y4 - y3) < eps(real(eltype(y4)))
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
        val = sum(x -> f(x), l:r)
    end
    err = zero(val)  # Dummy value to match the return type of integrate_real
    (; val, err)
end
