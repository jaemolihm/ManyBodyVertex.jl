# TODO: Optimize by reducing allocation inside function call

"""
    compute_bubble_smoothed(G, basis_f, basis_b, valC::Val{C}, temperature=nothing)
Improved version of `compute_bubble`.

The information Π should store: data on v1 & v2 grid. (for simple case: separable as G1 and
G2, for k-integrated case: not separable.)

What we need in an actual calculation: `∫dv Π(v, w) f(v)` for given `w`.

We approximate this integral by first computing `cᵢ(w) = ∫dv Π(v, w) bᵢ(v)` so that the
bubble is approximated as `Π(v, w) ≈ ∑ᵢ bᵢ(v) * cᵢ(w)`. Then, the integral is approximated
as `∫dv Π(v, w) f(v) ≈ ∑ᵢ cᵢ(w) * ∫dv bᵢ(v) f(v)`.
This is equivalent to using the truncated-unity approximation `δ(v, v') ≈ ∑ᵢ bᵢ(v) bᵢ(v')`.
This approximation is accurate when `f(v)` is spanned by `bᵢ(v)`.

(FIXME: In the above we assumed `bᵢ(v')` are orthonormal. But, we actually do deal with
non-orthogonality of `bᵢ(v')`.)

The basis for `w` of `Π` should be chosen such that it is as dense as the `v`-grid for
`|w| < 2 * max(v)`, and coarse but spanning up to largest bosonic frequency of interest
(e.g. largest fitting grid point for `w` used in BSE) for larger `|w|`.
"""
function compute_bubble_smoothed(G::AbstractLazyGreen2P{F}, basis_f, basis_b, valC::Val{C};
                                 temperature=nothing) where {F, C}
    ntails(basis_b) > 0 && error("tails cannot be used for `basis_b` in compute_bubble_smoothed")
    nind = get_nind(G)
    valF = Val(F)

    # Compute ∫dv b_f(v) * b_1p(v1) * b_1p(v2) where v1 and v2 are bubble frequencies for (v, w)
    ws = get_fitting_points(basis_b)
    Π_data = zeros(ComplexF64, nbasis(basis_f), nind^4, length(ws))
    # Base.Threads.@threads
    for iw in axes(ws, 1)
        w = ws[iw]
        set_basis_shift!(basis_f, w)
        overlap_f = basis_integral(basis_f, basis_f)

        bubble_value_integral = zeros(ComplexF64, size(basis_f, 2), nind^4)
        for i_f in axes(basis_f, 2)
            interval_v = integration_interval((basis_f,), (i_f,))
            isempty(interval_v) && continue

            function f(v)
                v1, v2 = _bubble_frequencies(valF, valC, v, w)
                coeff_f = basis_f[v, i_f]
                g1 = vec(G(v1))
                g2 = vec(G(v2))
                (g1 * transpose(g2)) .* coeff_f
            end

            l_v, r_v = endpoints(interval_v)
            if F === :MF
                res = integrate_imag(f, l_v, r_v)
            else
                l_v >= r_v && continue
                l_v ≈ r_v && continue
                res = quadgk(f, l_v, r_v)
            end

            value = reshape(res[1], nind, nind, nind, nind)
            for (ik, ks) in enumerate(Iterators.product(1:nind, 1:nind, 1:nind, 1:nind))
                k11, k12, k21, k22 = _bubble_indices(valC, ks)
                bubble_value_integral[i_f, ik] += value[k11, k12, k21, k22]
            end
        end
        Π_data[:, :, iw] .= qr(overlap_f, Val(true)) \ bubble_value_integral
    end

    Π = Bubble{F, C}(basis_f, basis_b; temperature)
    Π.data .= fit_basis_coeff(reshape(Π_data, :, nind^2, nind^2, length(ws)), basis_b, ws, 4)
    Π.data .*= _bubble_prefactor(valC)

    Π
end


function compute_bubble_smoothed(G::Green2P{F}, basis_f, basis_b, valC::Val{C}, temperature=nothing) where {F, C}
    # Same as above, but G is a Green2P, not a AbstractLazyGreen2P.
    # Use preallocated buffers: coeff_g1, coeff_g2, g1, g2.

    ntails(basis_b) > 0 && error("tails cannot be used for `basis_b` in compute_bubble_smoothed")
    valF = Val(F)
    nind = get_nind(G)
    G_data_reshaped = Base.ReshapedArray(G.data, (nind^2, nbasis(G.basis)), ())

    # Compute ∫dv b_f(v) * b_1p(v1) * b_1p(v2) where v1 and v2 are bubble frequencies for (v, w)
    ws = get_fitting_points(basis_b)
    Π_data = zeros(ComplexF64, nbasis(basis_f), nind^4, length(ws))
    # Base.Threads.@threads
    for iw in axes(ws, 1)
        w = ws[iw]
        set_basis_shift!(basis_f, w)
        overlap_f = basis_integral(basis_f, basis_f)

        coeff_g1 = zeros(eltype(G.basis), size(G.basis, 2))
        coeff_g2 = zeros(eltype(G.basis), size(G.basis, 2))
        g1 = zeros(eltype(G), nind^2)
        g2 = zeros(eltype(G), nind^2)

        bubble_value_integral = zeros(ComplexF64, size(basis_f, 2), nind^4)
        for i_f in axes(basis_f, 2)
            interval_v = integration_interval((basis_f,), (i_f,))
            isempty(interval_v) && continue

            function f(v)
                v1, v2 = _bubble_frequencies(valF, valC, v, w)
                coeff_f = basis_f[v, i_f]
                @views coeff_g1 .= G.basis[v1, :]
                @views coeff_g2 .= G.basis[v2, :]
                mul!(g1, G_data_reshaped, coeff_g1)
                mul!(g2, G_data_reshaped, coeff_g2)
                (g1 * transpose(g2)) .* coeff_f
            end

            l_v, r_v = endpoints(interval_v)
            if F === :MF
                res = integrate_imag(f, l_v, r_v)
            else
                l_v >= r_v && continue
                l_v ≈ r_v && continue
                res = quadgk(f, l_v, r_v)
            end

            value = reshape(res[1], nind, nind, nind, nind)
            for (ik, ks) in enumerate(Iterators.product(1:nind, 1:nind, 1:nind, 1:nind))
                k11, k12, k21, k22 = _bubble_indices(valC, ks)
                bubble_value_integral[i_f, ik] += value[k11, k12, k21, k22]
            end
        end
        Π_data[:, :, iw] .= qr(overlap_f, Val(true)) \ bubble_value_integral
    end

    Π = Bubble{F, C}(basis_f, basis_b; temperature)
    Π.data .= fit_basis_coeff(reshape(Π_data, :, nind^2, nind^2, length(ws)), basis_b, ws, 4)
    Π.data .*= _bubble_prefactor(valC)

    Π
end
