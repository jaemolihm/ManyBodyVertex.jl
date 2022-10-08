"""
    basis_integral(bases...)
Compute overlap between basis functions.
``S_{i_1,..., i_N} = ∫dx basis1_{i_1}(x) * ... * basisN_{i_N}(x)``
"""
function basis_integral(bases...)
    sizes = size.(bases, 2)
    overlap = zeros(eltype(bases[1]), sizes)
    for inds in CartesianIndices(sizes)
        # Divide the function support by intervals and integrate each interval.
        # Directly integrating over (-Inf, Inf) is inefficient, and may even become
        # inaccurate when the support is narrow.
        if length(bases) == 1
            intervals = interval_iterable(support_domain(bases[1], inds.I[1]))
        else
            intervals = interval_iterable(intersect(support_domain.(bases, inds.I)...))
        end
        intervals === () && continue

        f(x) = prod(getindex.(bases, x, inds.I))
        for interval in intervals
            isempty(interval) && continue
            @assert leftendpoint(interval) <= rightendpoint(interval)
            res, err = quadgk(f, leftendpoint(interval), rightendpoint(interval))
            overlap[inds] += res
        end
    end
    overlap
end
