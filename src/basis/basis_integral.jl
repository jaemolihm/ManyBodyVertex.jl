"""
    basis_integral(bases...)
Compute overlap between basis functions.
``S_{i_1,..., i_N} = ∫dx basis1_{i_1}(x) * ... * basisN_{i_N}(x)``
"""
basis_integral(bases...) = basis_integral(bases)

basis_integral(bases::Tuple) = (bases[1] isa AbstractImagBasis ? basis_integral_imag(bases)
                                                               : basis_integral_real(bases))

function basis_integral_real(bases::Tuple)
    sizes = size.(bases, 2)
    overlap = zeros(eltype(bases[1]), sizes)
    for inds in CartesianIndices(sizes)
        # Integrate the function only on the interval where the function is nonzero.
        # Directly integrating over (-Inf, Inf) is inefficient, and may even become
        # inaccurate when the support is narrow.
        interval = integration_interval(bases, inds.I)
        isempty(interval) && continue

        f(x) = prod(getindex.(bases, x, inds.I))
        l, r = interval.left, interval.right
        (l >= r || l ≈ r) && continue
        res, err = quadgk(f, l, r)
        overlap[inds] += res
    end
    overlap
end

@inline function integration_interval(bases, inds)
    bounds = support_bounds.(bases, inds)
    maximum(leftendpoint.(bounds)) .. minimum(rightendpoint.(bounds))
end

