"""
    basis_integral(bases...; skip_divergence=false)
Compute overlap between basis functions.
``S_{i_1,..., i_N} = ∫dx basis1_{i_1}(x) * ... * basisN_{i_N}(x)``
If `skip_divergence=true`, set the integral to zero if the integral is divergent. This option
is allowed only when `N = 1`.
"""
basis_integral(bases...; skip_divergence=false) = basis_integral(bases; skip_divergence)

basis_integral(bases::Tuple; skip_divergence=false) = (
    bases[1] isa AbstractImagBasis ? basis_integral_imag(bases; skip_divergence)
                                   : basis_integral_real(bases; skip_divergence))

function basis_integral_real(bases::Tuple; skip_divergence=false)
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

